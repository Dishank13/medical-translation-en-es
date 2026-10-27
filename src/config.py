"""Shared settings for every model and script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
TRANSLATIONS_DIR = RESULTS_DIR / "translations"
CHECKPOINT_DIR = ROOT / "checkpoints"   # large weights, not committed

SRC_LANG, TGT_LANG = "English", "Spanish"
SEED = 42

# Dataset (OPUS-EMEA v3, en-es)
DATASET_URL = "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-es.txt.zip"
MIN_WORDS, MAX_WORDS = 4, 40
MAX_LEN_RATIO = 2.0
N_TRAIN, N_VAL, N_TEST = 30000, 500, 200

# Basic Transformer (trained from scratch)
SPM_VOCAB = 8000
TF_D_MODEL, TF_HEADS, TF_LAYERS, TF_FF, TF_DROPOUT = 256, 4, 3, 1024, 0.1
TF_BATCH, TF_EPOCHS, TF_LR, TF_WARMUP = 64, 6, 7e-4, 800
TF_MAX_TOKENS = 80

# mT5 (fine-tuned)
MT5_NAME = "google/mt5-small"
MT5_PREFIX = "translate English to Spanish: "
MT5_TRAIN = 8000       # subset of the training split used for fine-tuning on CPU
MT5_BATCH, MT5_EPOCHS, MT5_LR = 16, 1, 1e-3
MT5_MAX_TOKENS = 80

# Decoder-only LLMs (prompting and LoRA)
LLMS = {
    "LLaMA": "unsloth/Llama-3.2-1B-Instruct",
    "Gemma": "unsloth/gemma-3-1b-it",
}
FEW_SHOT_K = 3
LORA_TRAIN = 400       # training pairs used for LoRA on CPU
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_BATCH, LORA_EPOCHS, LORA_LR = 4, 1, 2e-4
LLM_MAX_NEW_TOKENS = 96

# Sampling-temperature analysis
TEMPERATURES = [0.3, 0.7, 1.0, 1.5]
TEMP_SUBSET = 40       # test sentences used for the temperature study
