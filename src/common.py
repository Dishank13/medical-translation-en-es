"""Helpers shared by the model scripts: saving outputs and run information."""
import json
import random

import numpy as np
import torch

from . import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def clean_output(text):
    """Keep the translation line of an LLM answer.

    Chat models sometimes add notes after the translation, or open with a preamble such as
    "The translation of the sentence into Spanish is:". Notes are dropped, and a first line ending
    in a colon is treated as a preamble and skipped.
    """
    lines = [line.strip().strip('"').strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    if len(lines) > 1 and lines[0].endswith(":"):
        return lines[1]
    return lines[0]


def save_translations(system, lines, kind="test"):
    """kind: 'test', 'probes' or 'temperature/T<value>'."""
    path = config.TRANSLATIONS_DIR / kind / f"{system}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.replace("\n", " ") for line in lines) + "\n", encoding="utf-8")


def save_run(system, info):
    """Record timing, size and adaptation details for one system; merged later by evaluate.py."""
    path = config.RESULTS_DIR / "runs" / f"{system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(info)
    path.write_text(json.dumps(existing, indent=2))


def load_probes():
    path = config.ROOT / "data" / "ethics_probes.tsv"
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    return [(category, sentence) for category, sentence in rows]
