"""Paired bootstrap resampling over the test sentences for selected pairs of systems.

Run from the repository root (after src.evaluate):  python -m src.significance
"""
import csv
import random

from sacrebleu.metrics import BLEU, CHRF

from . import config
from .data import load

PAIRS = [("Gemma-few", "LLaMA-few"), ("Gemma-lora", "LLaMA-lora"), ("Transformer", "mT5"),
         ("Gemma-few", "Gemma-lora"), ("Gemma-lora", "Gemma-zero")]
RESAMPLES = 1000


def main():
    refs = [es for _, es in load("test")]
    systems = {s for pair in PAIRS for s in pair}
    hyps = {s: (config.TRANSLATIONS_DIR / "test" / f"{s}.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
            for s in systems}
    rng = random.Random(config.SEED)
    samples = [[rng.randrange(len(refs)) for _ in refs] for _ in range(RESAMPLES)]
    rows = []
    for a, b in PAIRS:
        row = {"system_a": a, "system_b": b}
        for name, metric in (("BLEU", BLEU()), ("chrF", CHRF())):
            wins = 0
            for idx in samples:
                r = [[refs[i] for i in idx]]
                wins += (metric.corpus_score([hyps[a][i] for i in idx], r).score
                         > metric.corpus_score([hyps[b][i] for i in idx], r).score)
            row[f"a_better_{name}"] = wins / RESAMPLES
        rows.append(row)
        print(row, flush=True)
    with open(config.RESULTS_DIR / "significance.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
