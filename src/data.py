"""Download, clean, deduplicate and split the EMEA English-Spanish parallel corpus.

Run from the repository root:  python -m src.data
"""
import json
import random
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter

from . import config

SPLITS = ("train", "val", "test")


def download():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR / "en-es.txt.zip"
    if not zip_path.exists():
        print(f"downloading {config.DATASET_URL}")
        urllib.request.urlretrieve(config.DATASET_URL, zip_path)
    en_path = config.RAW_DIR / "EMEA.en-es.en"
    if not en_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config.RAW_DIR)
    return en_path, config.RAW_DIR / "EMEA.en-es.es"


def normalise(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def keep(en, es):
    """Filter out headings, fragments, untranslated lines and badly aligned pairs."""
    n_en, n_es = len(en.split()), len(es.split())
    if not (config.MIN_WORDS <= n_en <= config.MAX_WORDS and config.MIN_WORDS <= n_es <= config.MAX_WORDS):
        return False
    if max(n_en, n_es) / min(n_en, n_es) > config.MAX_LEN_RATIO:
        return False
    if en.lower() == es.lower():
        return False
    if sum(c.isalpha() for c in en) < 0.6 * len(en):  # tables, codes, numbers
        return False
    if not en[0].isupper() or en[-1] not in ".!?:)":  # keep complete sentences only
        return False
    return True


def build():
    en_path, es_path = download()
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    es_lines = es_path.read_text(encoding="utf-8").splitlines()
    assert len(en_lines) == len(es_lines)

    stats = {"raw_pairs": len(en_lines)}
    pairs, seen = [], set()
    for en, es in zip(en_lines, es_lines):
        en, es = normalise(en), normalise(es)
        if not en or not es or not keep(en, es):
            continue
        key = en.lower()
        if key in seen:  # EMEA repeats the same leaflet sentences many times
            continue
        seen.add(key)
        pairs.append((en, es))
    stats["clean_unique_pairs"] = len(pairs)

    # Deduplicating on the English side before splitting means no test sentence occurs in training.
    random.Random(config.SEED).shuffle(pairs)
    test = pairs[:config.N_TEST]
    val = pairs[config.N_TEST:config.N_TEST + config.N_VAL]
    train = pairs[config.N_TEST + config.N_VAL:config.N_TEST + config.N_VAL + config.N_TRAIN]

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in zip(SPLITS, (train, val, test)):
        with open(config.DATA_DIR / f"{name}.tsv", "w", encoding="utf-8", newline="\n") as f:
            for en, es in split:
                f.write(f"{en}\t{es}\n")
    stats.update(describe(train, val, test))
    (config.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def load(split):
    with open(config.DATA_DIR / f"{split}.tsv", encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f]


def describe(train, val, test):
    stats = {}
    for name, split in zip(SPLITS, (train, val, test)):
        en_words = [len(en.split()) for en, _ in split]
        es_words = [len(es.split()) for _, es in split]
        stats[name] = {
            "pairs": len(split),
            "avg_en_words": round(sum(en_words) / len(split), 2),
            "avg_es_words": round(sum(es_words) / len(split), 2),
        }
    en_vocab = Counter(w.lower().strip(".,;:()") for en, _ in train for w in en.split())
    es_vocab = Counter(w.lower().strip(".,;:()") for _, es in train for w in es.split())
    stats["train_en_word_types"] = len(en_vocab)
    stats["train_es_word_types"] = len(es_vocab)
    stats["train_sentences_with_digits"] = sum(any(c.isdigit() for c in en) for en, _ in train)
    stats["most_common_en_words"] = [w for w, _ in en_vocab.most_common(25)]
    return stats


if __name__ == "__main__":
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    build()
