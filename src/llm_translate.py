"""Decoder-only LLMs (LLaMA, Gemma): zero-shot prompting, retrieved few-shot prompting and LoRA fine-tuning.

Run from the repository root, one model at a time:
    python -m src.llm_translate --model LLaMA
    python -m src.llm_translate --model Gemma
"""
import argparse
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from . import config
from .download_models import local_path
from .common import clean_output, count_params, load_probes, save_run, save_translations, set_seed
from .data import load

INSTRUCTION = ("Translate the following English sentence from a medicine package leaflet into Spanish. "
               "Reply with the Spanish translation only.\n\n{sentence}")


def chat(tokenizer, sentence, examples=()):
    """Build the prompt with the model's own chat template; few-shot examples become earlier turns."""
    messages = []
    for en, es in examples:
        messages += [{"role": "user", "content": INSTRUCTION.format(sentence=en)},
                     {"role": "assistant", "content": es}]
    messages.append({"role": "user", "content": INSTRUCTION.format(sentence=sentence)})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Retriever:
    """Pick the k training pairs whose English side is most similar (TF-IDF cosine) to the input."""

    def __init__(self, train):
        self.train = train
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform([en for en, _ in train])

    def __call__(self, sentence, k=config.FEW_SHOT_K):
        scores = (self.matrix @ self.vectorizer.transform([sentence]).T).toarray().ravel()
        best = scores.argsort()[::-1][:k]
        return [self.train[i] for i in best[::-1]]  # most similar example closest to the query


@torch.no_grad()
def generate(model, tokenizer, prompts, temperature=0.0, batch_size=8):
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(prompts)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = tokenizer([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        kwargs = ({"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0}
                  if temperature > 0 else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
        out = model.generate(**enc, max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                             pad_token_id=tokenizer.pad_token_id, **kwargs)
        texts = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for j, text in zip(idx, texts):
            outputs[j] = clean_output(text)
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outputs


def lora_finetune(model, tokenizer, pairs):
    """Train LoRA adapters on prompt -> translation, with the loss on the translation tokens only."""
    lora = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = count_params(model, trainable_only=True)
    print(f"LoRA trainable parameters: {trainable:,}")

    examples = []
    for en, es in pairs:
        prompt_text = chat(tokenizer, en)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION.format(sentence=en)}, {"role": "assistant", "content": es}],
            tokenize=False)
        assert full_text.startswith(prompt_text)
        prompt = tokenizer(prompt_text, add_special_tokens=False).input_ids
        # the answer includes the template's end-of-turn marker, so the model learns when to stop
        answer = tokenizer(full_text[len(prompt_text):], add_special_tokens=False).input_ids
        examples.append((prompt + answer, [-100] * len(prompt) + answer))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.LORA_LR)
    steps = config.LORA_EPOCHS * len(examples) // config.LORA_BATCH
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * steps), steps)
    rng = random.Random(config.SEED)
    model.train()
    start, step, running = time.perf_counter(), 0, 0.0
    for _ in range(config.LORA_EPOCHS):
        rng.shuffle(examples)
        for b in range(0, len(examples) - config.LORA_BATCH + 1, config.LORA_BATCH):
            batch = examples[b:b + config.LORA_BATCH]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([x + [tokenizer.pad_token_id] * (width - len(x)) for x, _ in batch])
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch])
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch])
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
            if step % 10 == 0:
                print(f"  LoRA step {step}/{steps}  loss {running / 10:.3f}  {time.perf_counter() - start:.0f}s",
                      flush=True)
                running = 0.0
    train_time = time.perf_counter() - start
    adapter_dir = config.CHECKPOINT_DIR / "lora" / Path(tokenizer.name_or_path).name
    model.save_pretrained(adapter_dir)
    return model.merge_and_unload(), trainable, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(config.LLMS), required=True)
    parser.add_argument("--modes", nargs="+", default=["zero", "few", "lora"])
    parser.add_argument("--extras", default="lora", help="mode used for the probe and temperature runs")
    args = parser.parse_args()

    set_seed()
    name = args.model
    repo = config.LLMS[name]
    tokenizer = AutoTokenizer.from_pretrained(local_path(repo))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma 3's model card recommends eager attention for fine-tuning
    attention = "eager" if name == "Gemma" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(local_path(repo), torch_dtype=torch.float32,
                                                 attn_implementation=attention)
    total_params = count_params(model)
    print(f"{name}: {total_params:,} parameters")

    train, test = load("train"), load("test")
    sources = [en for en, _ in test]
    probes = [s for _, s in load_probes()]
    base_info = {"model": repo, "architecture": "decoder-only", "pretrained": True,
                 "tokenizer": f"{type(tokenizer).__name__}, {len(tokenizer):,} vocab",
                 "params_total": total_params}

    def run(mode, prompts_for):
        system = f"{name}-{mode}"
        start = time.perf_counter()
        hyps = generate(model, tokenizer, prompts_for(sources))
        save_translations(system, hyps)
        info = dict(base_info, system=system, inference_time_s=time.perf_counter() - start)
        if mode == args.extras:
            save_translations(system, generate(model, tokenizer, prompts_for(probes)), kind="probes")
            subset = prompts_for(sources[: config.TEMP_SUBSET])
            for temperature in config.TEMPERATURES:
                torch.manual_seed(config.SEED)
                save_translations(system, generate(model, tokenizer, subset, temperature),
                                  kind=f"temperature/T{temperature}")
        return system, info

    if "zero" in args.modes:
        system, info = run("zero", lambda xs: [chat(tokenizer, x) for x in xs])
        save_run(system, dict(info, adaptation="zero-shot prompting", params_trained=0, train_time_s=0))
    if "few" in args.modes:
        retriever = Retriever(train)
        system, info = run("few", lambda xs: [chat(tokenizer, x, retriever(x)) for x in xs])
        save_run(system, dict(info, adaptation=f"{config.FEW_SHOT_K}-shot prompting, examples retrieved "
                                               "from the training split by TF-IDF similarity",
                              params_trained=0, train_time_s=0))
    if "lora" in args.modes:
        model, trainable, train_time = lora_finetune(model, tokenizer, train[: config.LORA_TRAIN])
        system, info = run("lora", lambda xs: [chat(tokenizer, x) for x in xs])
        save_run(system, dict(info, adaptation=f"LoRA (r={config.LORA_R}) on {config.LORA_TRAIN:,} pairs, "
                                               f"{config.LORA_EPOCHS} epoch, then zero-shot prompt",
                              params_trained=trainable, train_time_s=train_time,
                              train_pairs=config.LORA_TRAIN))


if __name__ == "__main__":
    main()
