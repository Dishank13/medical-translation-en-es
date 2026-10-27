"""Fine-tune the pretrained multilingual mT5-small encoder-decoder.

mT5 has a 250k-token vocabulary covering 101 languages; its embedding and output layers hold most of
the parameters and most of the compute on a CPU. Before fine-tuning, the vocabulary is trimmed to the
tokens that occur in the English-Spanish training and validation data (test data is not used), and the
matching rows of the embedding and output matrices are kept. The network itself is unchanged.

Run from the repository root:  python -m src.mt5_finetune
"""
import json
import random
import time

import torch
from torch import nn
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from transformers.optimization import Adafactor

from . import config
from .common import count_params, load_probes, save_run, save_translations, set_seed
from .data import load
from .download_models import local_path

SYSTEM = "mT5"


class TrimmedVocab:
    """Maps between the original mT5 token ids and the ids of the trimmed vocabulary."""

    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        used = set()
        for i in range(0, len(texts), 2000):
            for ids in tokenizer(texts[i:i + 2000]).input_ids:
                used.update(ids)
        # pad (0), eos (1) and unk (2) keep their ids, so generation settings stay valid
        specials = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]
        assert specials == [0, 1, 2]
        self.keep = specials + sorted(used - set(specials))
        self.old_to_new = torch.full((len(tokenizer),), tokenizer.unk_token_id, dtype=torch.long)
        self.old_to_new[torch.tensor(self.keep)] = torch.arange(len(self.keep))
        self.new_to_old = torch.tensor(self.keep)

    def __len__(self):
        return len(self.keep)

    def encode(self, texts, target=False):
        kwargs = {"text_target": texts} if target else {"text": texts}
        enc = self.tokenizer(max_length=config.MT5_MAX_TOKENS, truncation=True, padding=True,
                             return_tensors="pt", **kwargs)
        enc["input_ids"] = self.old_to_new[enc["input_ids"]]
        return enc

    def decode(self, ids):
        return [t.strip() for t in self.tokenizer.batch_decode(self.new_to_old[ids], skip_special_tokens=True)]


def trim_model(model, vocab):
    idx = vocab.new_to_old
    d = model.config.d_model
    embed = nn.Embedding(len(vocab), d)
    embed.weight.data = model.get_input_embeddings().weight.data[idx].clone()
    model.set_input_embeddings(embed)
    head = nn.Linear(d, len(vocab), bias=False)
    head.weight.data = model.get_output_embeddings().weight.data[idx].clone()
    model.set_output_embeddings(head)
    model.config.vocab_size = len(vocab)
    return model


def tokenize(vocab, pairs):
    enc = vocab.encode([config.MT5_PREFIX + en for en, _ in pairs])
    labels = vocab.encode([es for _, es in pairs], target=True)["input_ids"]
    labels[labels == 0] = -100
    enc["labels"] = labels
    return enc


@torch.no_grad()
def val_loss(model, vocab, pairs):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(pairs), 32):
        batch = tokenize(vocab, pairs[i:i + 32])
        tokens = (batch["labels"] != -100).sum().item()
        total += model(**batch).loss.item() * tokens
        n += tokens
    return total / n


@torch.no_grad()
def translate(model, vocab, sentences, temperature=0.0, batch_size=32):
    model.eval()
    # sort by length so each batch needs little padding, then restore the order
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    outputs = [None] * len(sentences)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = vocab.encode([config.MT5_PREFIX + sentences[j] for j in idx])
        kwargs = {"do_sample": True, "temperature": temperature, "top_k": 0} if temperature > 0 else {"num_beams": 1}
        out = model.generate(**enc, max_new_tokens=config.MT5_MAX_TOKENS, **kwargs)
        for j, text in zip(idx, vocab.decode(out)):
            outputs[j] = text
    return outputs


def main():
    set_seed()
    all_train = load("train")
    train = all_train[: config.MT5_TRAIN]
    val, test = load("val"), load("test")
    tokenizer = AutoTokenizer.from_pretrained(local_path(config.MT5_NAME))
    model = MT5ForConditionalGeneration.from_pretrained(local_path(config.MT5_NAME))
    original_params = count_params(model)

    texts = [config.MT5_PREFIX] + [s for pair in all_train + val for s in pair]
    vocab = TrimmedVocab(tokenizer, texts)
    model = trim_model(model, vocab)
    test_tokens = [t for ids in tokenizer([s for p in test for s in p]).input_ids for t in ids]
    coverage = sum(vocab.old_to_new[t].item() != 2 or t == 2 for t in test_tokens) / len(test_tokens)
    print(f"parameters: {original_params:,} -> {count_params(model):,} after trimming the vocabulary "
          f"from {len(tokenizer):,} to {len(vocab):,} tokens (test-token coverage {coverage:.4f})", flush=True)

    optimizer = Adafactor(model.parameters(), lr=config.MT5_LR, scale_parameter=False,
                          relative_step=False, warmup_init=False)
    sources = [en for en, _ in test]
    history = [{"step": 0, "val_loss": val_loss(model, vocab, val[:200])}]
    print(history[-1], flush=True)
    zero_shot = translate(model, vocab, sources[:5])  # shows what the pretrained model does before fine-tuning
    print("before fine-tuning:", zero_shot, flush=True)

    rng = random.Random(config.SEED)
    steps_per_epoch = len(train) // config.MT5_BATCH
    start = time.perf_counter()
    step = 0
    for epoch in range(config.MT5_EPOCHS):
        order = list(range(len(train)))
        rng.shuffle(order)
        model.train()
        running = 0.0
        for b in range(steps_per_epoch):
            batch = tokenize(vocab, [train[i] for i in order[b * config.MT5_BATCH:(b + 1) * config.MT5_BATCH]])
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running += loss.item()
            if step % 25 == 0:
                print(f"step {step}/{steps_per_epoch * config.MT5_EPOCHS}  train loss {running / 25:.3f}  "
                      f"{time.perf_counter() - start:.0f}s", flush=True)
                running = 0.0
            if step % 125 == 0:
                history.append({"step": step, "val_loss": val_loss(model, vocab, val[:200]),
                                "elapsed_s": time.perf_counter() - start})
                print(history[-1], flush=True)
    train_time = time.perf_counter() - start
    history.append({"step": step, "val_loss": val_loss(model, vocab, val), "full_val": True})
    out_dir = config.CHECKPOINT_DIR / "mt5"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "vocab_keep.json").write_text(json.dumps(vocab.keep))

    start = time.perf_counter()
    hyps = translate(model, vocab, sources)
    infer_time = time.perf_counter() - start
    save_translations(SYSTEM, hyps)
    save_translations(SYSTEM, translate(model, vocab, [s for _, s in load_probes()]), kind="probes")
    for temperature in config.TEMPERATURES:
        torch.manual_seed(config.SEED)
        save_translations(SYSTEM, translate(model, vocab, sources[: config.TEMP_SUBSET], temperature),
                          kind=f"temperature/T{temperature}")

    save_run(SYSTEM, {
        "system": SYSTEM, "model": config.MT5_NAME, "architecture": "encoder-decoder",
        "adaptation": f"full fine-tuning on {len(train):,} pairs, {config.MT5_EPOCHS} epoch "
                      f"(vocabulary trimmed to {len(vocab):,} tokens)",
        "pretrained": True,
        "tokenizer": f"SentencePiece unigram, 250k multilingual vocab trimmed to {len(vocab):,}",
        "params_total": count_params(model), "params_original": original_params,
        "params_trained": count_params(model), "vocab_original": len(tokenizer), "vocab_trimmed": len(vocab),
        "test_token_coverage": coverage, "before_finetuning_examples": zero_shot,
        "train_time_s": train_time, "inference_time_s": infer_time, "train_pairs": len(train),
        "history": history,
    })


if __name__ == "__main__":
    main()
