"""Score every system on the shared test set and build the comparison tables and plots.

Run from the repository root after the model scripts:  python -m src.evaluate
"""
import csv
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU, CHRF

from . import config
from .common import load_probes
from .data import load

ORDER = ["Transformer", "mT5", "LLaMA-zero", "LLaMA-few", "LLaMA-lora", "Gemma-zero", "Gemma-few", "Gemma-lora"]
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
EN_NEGATION = re.compile(r"\b(not|never|no|nor|without|cannot)\b|n't\b", re.I)
ES_NEGATION = re.compile(r"\b(no|nunca|jamás|ni|sin|ningún|ninguna|ninguno|nada|tampoco)\b", re.I)


def numbers(text):
    return sorted(n.replace(",", ".") for n in NUMBER_RE.findall(text))


def fidelity(sources, hyps):
    """Simple checks for content that must survive translation in a medicine leaflet."""
    with_numbers = [(s, h) for s, h in zip(sources, hyps) if NUMBER_RE.search(s)]
    with_negation = [(s, h) for s, h in zip(sources, hyps) if EN_NEGATION.search(s)]
    return {
        "number_fidelity": sum(numbers(s) == numbers(h) for s, h in with_numbers) / len(with_numbers),
        "negation_kept": sum(bool(ES_NEGATION.search(h)) for _, h in with_negation) / len(with_negation),
        "empty_or_copied": sum((not h.strip()) or h.strip().lower() == s.strip().lower()
                               for s, h in zip(sources, hyps)) / len(hyps),
        "n_with_numbers": len(with_numbers),
        "n_with_negation": len(with_negation),
    }


def read(path):
    return path.read_text(encoding="utf-8").rstrip("\n").split("\n")


def score(hyps, refs):
    return BLEU().corpus_score(hyps, [refs]).score, CHRF().corpus_score(hyps, [refs]).score


def systems():
    found = [p.stem for p in (config.RESULTS_DIR / "runs").glob("*.json")]
    return [s for s in ORDER if s in found] + sorted(s for s in found if s not in ORDER)


def main_table(test):
    sources, refs = [en for en, _ in test], [es for _, es in test]
    rows = []
    for system in systems():
        run = json.loads((config.RESULTS_DIR / "runs" / f"{system}.json").read_text())
        hyps = read(config.TRANSLATIONS_DIR / "test" / f"{system}.txt")
        assert len(hyps) == len(refs), system
        bleu, chrf = score(hyps, refs)
        row = {
            "system": system, "architecture": run["architecture"], "adaptation": run["adaptation"],
            "params_total_M": round(run["params_total"] / 1e6, 1),
            "params_trained_M": round(run["params_trained"] / 1e6, 2),
            "BLEU": round(bleu, 2), "chrF": round(chrf, 2),
            "train_time_min": round(run["train_time_s"] / 60, 1),
            "inference_time_s": round(run["inference_time_s"], 1),
            "inference_s_per_sentence": round(run["inference_time_s"] / len(hyps), 2),
        }
        row.update({k: round(v, 3) for k, v in fidelity(sources, hyps).items() if not k.startswith("n_")})
        rows.append(row)
        print(row, flush=True)
    # the human references through the same checks give a ceiling for the automatic checks
    reference = fidelity(sources, refs)
    (config.RESULTS_DIR / "reference_fidelity.json").write_text(json.dumps(reference, indent=2))
    print(f"reference fidelity: {reference}")
    with open(config.RESULTS_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_main(rows):
    names = [r["system"] for r in rows]
    colors = ["#4C72B0" if r["architecture"] == "encoder-decoder" else "#DD8452" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, key in zip(axes, ["BLEU", "chrF"]):
        bars = ax.bar(names, [r[key] for r in rows], color=colors)
        ax.bar_label(bars, fmt="%.1f")
        ax.set_title(f"{key} on the test set (blue: encoder-decoder, orange: decoder-only)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "quality_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, key, title in zip(axes, ["params_total_M", "train_time_min", "inference_s_per_sentence"],
                              ["Total parameters (millions, log scale)", "Training / fine-tuning time (min)",
                               "Inference time per sentence (s, log scale)"]):
        bars = ax.bar(names, [r[key] for r in rows], color=colors)
        ax.bar_label(bars, fmt="%g")
        ax.set_title(title)
        if "log" in title:
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "cost_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r, c in zip(rows, colors):
        ax.scatter(r["inference_s_per_sentence"], r["chrF"], s=40 + r["params_total_M"] / 4, color=c, alpha=0.8)
        ax.annotate(r["system"], (r["inference_s_per_sentence"], r["chrF"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("inference time per sentence (s, log scale)")
    ax.set_ylabel("chrF")
    ax.set_title("Quality vs inference cost (marker size ~ parameters)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "quality_vs_cost.png", dpi=150)
    plt.close(fig)


def distinct_2(lines):
    grams = [tuple(t[i:i + 2]) for line in lines for t in [line.split()] for i in range(len(t) - 1)]
    return len(set(grams)) / max(len(grams), 1)


def temperature_table(test):
    subset = test[: config.TEMP_SUBSET]
    sources, refs = [en for en, _ in subset], [es for _, es in subset]
    rows = []
    for system in systems():
        greedy = read(config.TRANSLATIONS_DIR / "test" / f"{system}.txt")[: config.TEMP_SUBSET]
        settings = [("greedy", greedy)]
        for t in config.TEMPERATURES:
            path = config.TRANSLATIONS_DIR / f"temperature/T{t}" / f"{system}.txt"
            if path.exists():
                settings.append((t, read(path)))
        if len(settings) == 1:
            continue
        for t, hyps in settings:
            bleu, chrf = score(hyps, refs)
            fid = fidelity(sources, hyps)
            rows.append({
                "system": system, "temperature": t, "BLEU": round(bleu, 2), "chrF": round(chrf, 2),
                "distinct_2": round(distinct_2(hyps), 3),
                "changed_vs_greedy": round(sum(h != g for h, g in zip(hyps, greedy)) / len(hyps), 3),
                "number_fidelity": round(fid["number_fidelity"], 3),
                "length_ratio": round(sum(len(h) for h in hyps) / sum(len(r) for r in refs), 3),
            })
    with open(config.RESULTS_DIR / "temperature_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    xs = [0.0] + config.TEMPERATURES
    for system in dict.fromkeys(r["system"] for r in rows):
        sys_rows = [r for r in rows if r["system"] == system]
        for ax, key in zip(axes, ["chrF", "number_fidelity", "distinct_2"]):
            ax.plot(xs[: len(sys_rows)], [r[key] for r in sys_rows], marker="o", label=system)
    for ax, title in zip(axes, ["chrF vs temperature", "Number fidelity vs temperature",
                                "Lexical diversity (distinct-2) vs temperature"]):
        ax.set_title(title)
        ax.set_xlabel("temperature (0 = greedy)")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "temperature_analysis.png", dpi=150)
    plt.close(fig)
    return rows


def write_examples(test):
    names = systems()
    outputs = {s: read(config.TRANSLATIONS_DIR / "test" / f"{s}.txt") for s in names}
    lines = ["# Sample test-set translations", "",
             "The first 15 test sentences, with the reference and the greedy output of every system.", ""]
    for i, (en, es) in enumerate(test[:15]):
        lines += [f"## {i + 1}", "", f"- **English:** {en}", f"- **Reference:** {es}"]
        lines += [f"- **{s}:** {outputs[s][i]}" for s in names] + [""]
    (config.RESULTS_DIR / "sample_translations.md").write_text("\n".join(lines), encoding="utf-8")

    probe_systems = [s for s in names if (config.TRANSLATIONS_DIR / "probes" / f"{s}.txt").exists()]
    probe_out = {s: read(config.TRANSLATIONS_DIR / "probes" / f"{s}.txt") for s in probe_systems}
    lines = ["# Responsible-AI probe translations", "",
             "Hand-written probe sentences (not from the dataset) covering gender, negation and dosage, "
             "sensitive medical content, cultural conventions and idioms.", ""]
    for i, (category, sentence) in enumerate(load_probes()):
        lines += [f"## {i + 1}. {category}", "", f"- **English:** {sentence}"]
        lines += [f"- **{s}:** {probe_out[s][i]}" for s in probe_systems] + [""]
    (config.RESULTS_DIR / "ethics_probes.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    test = load("test")
    rows = main_table(test)
    plot_main(rows)
    temperature_table(test)
    write_examples(test)


if __name__ == "__main__":
    main()
