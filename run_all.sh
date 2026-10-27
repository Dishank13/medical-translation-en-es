#!/usr/bin/env bash
# Runs the whole pipeline in order. Each step writes its log to results/logs/.
set -e
cd "$(dirname "$0")"
mkdir -p results/logs
python -m src.data                                      > results/logs/data.log 2>&1
python -m src.download_models                           > results/logs/download.log 2>&1
python -m src.transformer_mt                            > results/logs/transformer.log 2>&1
python -m src.mt5_finetune                              > results/logs/mt5.log 2>&1
python -m src.llm_translate --model LLaMA               > results/logs/llama.log 2>&1
python -m src.llm_translate --model Gemma               > results/logs/gemma.log 2>&1
python -m src.evaluate                                  > results/logs/evaluate.log 2>&1
python -m src.significance                              > results/logs/significance.log 2>&1
