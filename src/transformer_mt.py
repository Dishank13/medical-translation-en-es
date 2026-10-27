"""Basic encoder-decoder Transformer trained from scratch.

Run from the repository root:  python -m src.transformer_mt
"""
import argparse
import math
import re
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn

from . import config
from .common import count_params, save_run, save_translations, set_seed, load_probes
from .data import load

SYSTEM = "Transformer"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPM_PREFIX = config.CHECKPOINT_DIR / "transformer" / "spm"


def train_tokenizer(train):
    SPM_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    corpus = SPM_PREFIX.parent / "spm_corpus.txt"
    corpus.write_text("\n".join(s for pair in train for s in pair), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(SPM_PREFIX), vocab_size=config.SPM_VOCAB,
        model_type="bpe", character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        input_sentence_size=100000, shuffle_input_sentence=True, minloglevel=2)
    return spm.SentencePieceProcessor(model_file=f"{SPM_PREFIX}.model")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Seq2SeqTransformer(nn.Module):
    """Shared source/target embeddings (joint BPE vocabulary) tied to the output layer."""

    def __init__(self, vocab_size):
        super().__init__()
        d = config.TF_D_MODEL
        self.scale = math.sqrt(d)
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        # small init: the matrix is shared with the output layer, so N(0, 1) would give huge logits
        nn.init.normal_(self.embed.weight, mean=0.0, std=d ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()
        self.pos = PositionalEncoding(d)
        self.dropout = nn.Dropout(config.TF_DROPOUT)
        self.transformer = nn.Transformer(
            d_model=d, nhead=config.TF_HEADS, num_encoder_layers=config.TF_LAYERS,
            num_decoder_layers=config.TF_LAYERS, dim_feedforward=config.TF_FF,
            dropout=config.TF_DROPOUT, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.embed.weight

    def embed_tokens(self, ids):
        return self.dropout(self.pos(self.embed(ids) * self.scale))

    def encode(self, src):
        return self.transformer.encoder(self.embed_tokens(src), src_key_padding_mask=src == PAD)

    def decode(self, tgt, memory, src_pad_mask):
        causal = self.transformer.generate_square_subsequent_mask(tgt.size(1), dtype=torch.bool)
        hidden = self.transformer.decoder(self.embed_tokens(tgt), memory, tgt_mask=causal,
                                          tgt_key_padding_mask=tgt == PAD,
                                          memory_key_padding_mask=src_pad_mask)
        return self.out(hidden)

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src == PAD)


def pad_batch(seqs):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs])


def encode_pairs(sp, pairs):
    limit = config.TF_MAX_TOKENS
    src = [sp.encode(en)[: limit - 1] + [EOS] for en, _ in pairs]
    tgt = [[BOS] + sp.encode(es)[: limit - 2] + [EOS] for _, es in pairs]
    return src, tgt


def batches(src, tgt, batch_size, shuffle, generator=None):
    """Length-bucketed batches: sentences of similar length are grouped to avoid wasted padding."""
    order = torch.randperm(len(src), generator=generator).tolist() if shuffle else list(range(len(src)))
    pool = batch_size * 50
    groups = []
    for i in range(0, len(order), pool):
        chunk = sorted(order[i:i + pool], key=lambda j: len(src[j]))
        groups += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    if shuffle:
        groups = [groups[k] for k in torch.randperm(len(groups), generator=generator).tolist()]
    for idx in groups:
        yield pad_batch([src[j] for j in idx]), pad_batch([tgt[j] for j in idx])


@torch.no_grad()
def evaluate_loss(model, src, tgt, criterion):
    model.eval()
    total, n = 0.0, 0
    for s, t in batches(src, tgt, 128, shuffle=False):
        logits = model(s, t[:, :-1])
        mask = t[:, 1:] != PAD
        total += criterion(logits[mask], t[:, 1:][mask]).item() * mask.sum().item()
        n += mask.sum().item()
    return total / n


@torch.no_grad()
def translate(model, sp, sentences, temperature=0.0, batch_size=64, generator=None):
    """Greedy decoding (temperature 0) or temperature sampling, batched."""
    model.eval()
    outputs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        src = pad_batch([sp.encode(s)[: config.TF_MAX_TOKENS - 1] + [EOS] for s in chunk])
        memory, src_mask = model.encode(src), src == PAD
        ys = torch.full((len(chunk), 1), BOS)
        done = torch.zeros(len(chunk), dtype=torch.bool)
        for _ in range(config.TF_MAX_TOKENS):
            logits = model.decode(ys, memory, src_mask)[:, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == EOS
            if done.all():
                break
        for row in ys[:, 1:].tolist():
            row = row[: row.index(EOS)] if EOS in row else row
            outputs.append(sp.decode([t for t in row if t != PAD]))
    return outputs


def lr_at(step):
    """Linear warm-up followed by inverse square-root decay."""
    step = max(step, 1)
    return config.TF_LR * min(step / config.TF_WARMUP, math.sqrt(config.TF_WARMUP / step))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path,
                        help="log of an interrupted run; training continues after its last completed epoch")
    args = parser.parse_args()

    set_seed()
    train, val, test = load("train"), load("val"), load("test")
    sp = spm.SentencePieceProcessor(model_file=f"{SPM_PREFIX}.model") if args.resume else train_tokenizer(train)
    model = Seq2SeqTransformer(sp.get_piece_size())
    print(f"parameters: {count_params(model):,}", flush=True)

    tr_src, tr_tgt = encode_pairs(sp, train)
    va_src, va_tgt = encode_pairs(sp, val)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TF_LR, betas=(0.9, 0.98), eps=1e-9)
    generator = torch.Generator().manual_seed(config.SEED)
    checkpoint = config.CHECKPOINT_DIR / "transformer" / "model.pt"
    history, best, step, offset = [], float("inf"), 0, 0.0
    if args.resume:
        # Resume from the checkpoint written at the end of the last completed epoch. The batch order is
        # replayed so later epochs see the same data order as an uninterrupted run, and the learning-rate
        # schedule continues from the same step. Adam's moment estimates were not saved, so they restart.
        log = args.resume.read_text(encoding="utf-8")
        history = [{"epoch": int(m[1]), "train_loss": float(m[2]), "val_loss": float(m[3]), "elapsed_s": float(m[4])}
                   for m in re.finditer(r"^epoch\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)\s+(\d+)s", log, re.M)]
        best, offset = min(h["val_loss"] for h in history), history[-1]["elapsed_s"]
        assert best == history[-1]["val_loss"], "the saved checkpoint must be the last completed epoch"
        model.load_state_dict(torch.load(checkpoint))
        for _ in history:
            step += sum(1 for _ in batches(tr_src, tr_tgt, config.TF_BATCH, shuffle=True, generator=generator))
        print(f"resumed after epoch {len(history)} at step {step}", flush=True)

    resumed_after = len(history) or None
    start = time.perf_counter() - offset
    for epoch in range(len(history) + 1, config.TF_EPOCHS + 1):
        model.train()
        loss_sum, tokens = 0.0, 0
        for s, t in batches(tr_src, tr_tgt, config.TF_BATCH, shuffle=True, generator=generator):
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            logits = model(s, t[:, :-1])
            mask = t[:, 1:] != PAD
            loss = criterion(logits[mask], t[:, 1:][mask])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * mask.sum().item()
            tokens += mask.sum().item()
            if step % 100 == 0:
                print(f"  step {step}  loss {loss.item():.3f}  {time.perf_counter() - start:.0f}s", flush=True)
        val_loss = evaluate_loss(model, va_src, va_tgt, criterion)
        history.append({"epoch": epoch, "train_loss": loss_sum / tokens, "val_loss": val_loss,
                        "elapsed_s": time.perf_counter() - start})
        print(f"epoch {epoch:2d}  train {loss_sum / tokens:.3f}  val {val_loss:.3f}  "
              f"{time.perf_counter() - start:.0f}s", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), checkpoint)
    train_time = time.perf_counter() - start

    model.load_state_dict(torch.load(checkpoint))
    sources = [en for en, _ in test]
    start = time.perf_counter()
    hyps = translate(model, sp, sources)
    infer_time = time.perf_counter() - start
    save_translations(SYSTEM, hyps)
    save_translations(SYSTEM, translate(model, sp, [s for _, s in load_probes()]), kind="probes")

    subset = sources[: config.TEMP_SUBSET]
    for temperature in config.TEMPERATURES:
        gen = torch.Generator().manual_seed(config.SEED)
        save_translations(SYSTEM, translate(model, sp, subset, temperature, generator=gen),
                          kind=f"temperature/T{temperature}")

    save_run(SYSTEM, {
        "system": SYSTEM, "model": "Transformer (from scratch)", "architecture": "encoder-decoder",
        "adaptation": f"trained from scratch on {len(train):,} pairs, {config.TF_EPOCHS} epochs",
        "pretrained": False, "tokenizer": f"SentencePiece BPE, joint {config.SPM_VOCAB} vocab",
        "params_total": count_params(model), "params_trained": count_params(model),
        "train_time_s": train_time, "inference_time_s": infer_time, "train_pairs": len(train),
        "history": history, "resumed_after_epoch": resumed_after,
    })


if __name__ == "__main__":
    main()
