"""Publish the MiniDance weights / dataset to the Hugging Face Hub (needs `hf auth login` with a write token).

    python scripts/upload_hf.py model   --repo JuniverseLi/MiniDance
    python scripts/upload_hf.py dataset --repo JuniverseLi/minidance-scenery --src /content/drive/MyDrive/minidance
    python scripts/upload_hf.py dataset --repo JuniverseLi/minidance-scenery --card_only   # card only (data from Colab)

model:   out/minidance-base{,-sft}.safetensors + hf/MODEL_CARD.md + logo / sample clips
dataset: <src>/latents/shard_*.{safetensors,parquet} + <src>/short_captions.jsonl + hf/DATASET_CARD.md
         (resumable: rerun after an interruption and finished files are skipped)
"""
import argparse
import os

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = ["beach_sunset", "waterfall", "aerial_river", "sea_sunset_zh"]


def upload_model(api, repo, weights_dir):
    files = {f"{n}.safetensors": os.path.join(weights_dir, f"{n}.safetensors")
             for n in ("minidance-base-sft", "minidance-base")}
    files["README.md"] = os.path.join(ROOT, "hf", "MODEL_CARD.md")
    files["assets/logo.png"] = os.path.join(ROOT, "images", "logo.png")
    for s in SAMPLES:
        files[f"assets/{s}.webp"] = os.path.join(ROOT, "images", "gallery", f"{s}.webp")
    for dst, src in files.items():
        assert os.path.exists(src), f"missing {src}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for dst, src in files.items()]
    api.create_commit(repo, ops, commit_message="Upload MiniDance weights and model card")
    print(f"https://huggingface.co/{repo}")


def upload_dataset_card(api, repo):
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "hf", "DATASET_CARD.md"), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset", commit_message="Add dataset card")


def upload_dataset(api, repo, src):
    shards = [f for f in os.listdir(os.path.join(src, "latents")) if f.startswith("shard_")]
    assert os.path.exists(os.path.join(src, "short_captions.jsonl")), f"missing {src}/short_captions.jsonl"
    print(f"{len(shards)} shard files + short_captions.jsonl from {src}")
    upload_dataset_card(api, repo)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=src,
                            allow_patterns=["latents/shard_*.safetensors", "latents/shard_*.parquet",
                                            "short_captions.jsonl"])
    print(f"https://huggingface.co/datasets/{repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "dataset"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights_dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--src", help="dataset: folder holding latents/ and short_captions.jsonl")
    ap.add_argument("--card_only", action="store_true", help="dataset: only create the repo and its card")
    args = ap.parse_args()
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    if args.what == "model":
        upload_model(api, args.repo, args.weights_dir)
    elif args.card_only:
        upload_dataset_card(api, args.repo)
    else:
        upload_dataset(api, args.repo, args.src)


if __name__ == "__main__":
    main()
