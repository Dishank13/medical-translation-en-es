"""Download the pretrained models into checkpoints/pretrained/ (resumes after network errors).

Run from the repository root:  python -m src.download_models
"""
import time

from huggingface_hub import list_repo_files, snapshot_download

from . import config

REPOS = [config.MT5_NAME] + list(config.LLMS.values())


def local_path(repo):
    return config.CHECKPOINT_DIR / "pretrained" / repo.split("/")[-1]


def main():
    for repo in REPOS:
        for attempt in range(1, 500):
            try:
                files = list_repo_files(repo)
                weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["pytorch_model.bin"]
                snapshot_download(repo, local_dir=local_path(repo), ignore_patterns=["onnx/*"],
                                  allow_patterns=["*.json", "*.model", "*.jinja"] + weights)
                print(f"OK {repo}", flush=True)
                break
            except Exception as e:
                print(f"retry {attempt} {repo}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                time.sleep(15)


if __name__ == "__main__":
    main()
