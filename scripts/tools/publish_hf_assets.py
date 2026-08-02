#!/usr/bin/env python3
"""Publish MammoGen cards / optional binaries to Hugging Face (TraitYoung/*).

Requires:
  export HF_TOKEN=hf_xxx   # write token

Optional local paths (skipped with a warning when missing):
  outputs/lora/mammo_sd15_v6_allMLO/final_lora/
  datasets/CBIS_CLEAN_V2/metadata_clean.csv

Usage:
  python3 scripts/tools/publish_hf_assets.py --all
  python3 scripts/tools/publish_hf_assets.py --space
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HF_DIR = ROOT / "hf"

MODEL_ID = "TraitYoung/mammo-sd15-lora-v6"
DATASET_ID = "TraitYoung/cbis-clean-v2"
SPACE_ID = "TraitYoung/mammo-gallery"

DEFAULT_LORA = ROOT / "outputs/lora/mammo_sd15_v6_allMLO/final_lora"
DEFAULT_META = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"
DEMO_SRC_CANDIDATES = (
    ROOT / "docs" / "assets",
    ROOT / "medi-diff-demonstration",
)


def _hub():
    try:
        from huggingface_hub import HfApi, login
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required: pip install 'huggingface_hub>=0.23'"
        ) from exc
    return HfApi, login


def _ensure_token(login) -> None:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (write access) before publishing.")
    login(token=token)


def _sync_gallery_assets() -> None:
    dest = HF_DIR / "mammo-gallery" / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    demo_src = next((p for p in DEMO_SRC_CANDIDATES if p.is_dir()), None)
    if demo_src is None:
        print(f"[WARN] demo screenshots missing (tried: {list(DEMO_SRC_CANDIDATES)})")
        return
    for src in sorted(demo_src.glob("*.png")):
        shutil.copy2(src, dest / src.name)
        print(f"[OK] gallery asset ← {src.name}")


def publish_model(api, *, lora_dir: Path, private: bool) -> None:
    card_dir = HF_DIR / "mammo-sd15-lora-v6"
    api.create_repo(MODEL_ID, repo_type="model", exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=str(card_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=MODEL_ID,
        repo_type="model",
    )
    print(f"[OK] model card → {MODEL_ID}")
    if lora_dir.is_dir():
        api.upload_folder(
            folder_path=str(lora_dir),
            repo_id=MODEL_ID,
            repo_type="model",
            commit_message="Upload MammoGen LoRA v6 weights",
        )
        print(f"[OK] LoRA weights ← {lora_dir}")
    else:
        print(f"[SKIP] LoRA dir not found: {lora_dir} (card only)")


def publish_dataset(api, *, meta_csv: Path, private: bool) -> None:
    card_dir = HF_DIR / "cbis-clean-v2"
    api.create_repo(DATASET_ID, repo_type="dataset", exist_ok=True, private=private)
    for name in ("README.md", "SCHEMA.md"):
        api.upload_file(
            path_or_fileobj=str(card_dir / name),
            path_in_repo=name,
            repo_id=DATASET_ID,
            repo_type="dataset",
        )
    print(f"[OK] dataset card → {DATASET_ID}")
    if meta_csv.is_file():
        api.upload_file(
            path_or_fileobj=str(meta_csv),
            path_in_repo="metadata_clean.csv",
            repo_id=DATASET_ID,
            repo_type="dataset",
            commit_message="Upload CBIS_CLEAN_V2 metadata_clean.csv",
        )
        print(f"[OK] metadata ← {meta_csv}")
    else:
        print(f"[SKIP] metadata CSV not found: {meta_csv} (card only)")


def publish_space(api, *, private: bool) -> None:
    _sync_gallery_assets()
    space_dir = HF_DIR / "mammo-gallery"
    api.create_repo(SPACE_ID, repo_type="space", exist_ok=True, private=private, space_sdk="gradio")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Publish MammoGen static gallery Space",
    )
    print(f"[OK] space → https://huggingface.co/spaces/{SPACE_ID}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="Publish model + dataset + space")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--space", action="store_true")
    ap.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA)
    ap.add_argument("--metadata-csv", type=Path, default=DEFAULT_META)
    ap.add_argument("--private", action="store_true", help="Create private repos")
    ap.add_argument("--sync-assets-only", action="store_true", help="Only copy screenshots into hf/mammo-gallery/assets")
    args = ap.parse_args()

    if args.sync_assets_only:
        _sync_gallery_assets()
        return 0

    do_model = args.all or args.model
    do_dataset = args.all or args.dataset
    do_space = args.all or args.space
    if not (do_model or do_dataset or do_space):
        ap.print_help()
        return 2

    HfApi, login = _hub()
    _ensure_token(login)
    api = HfApi()

    if do_model:
        publish_model(api, lora_dir=args.lora_dir, private=args.private)
    if do_dataset:
        publish_dataset(api, meta_csv=args.metadata_csv, private=args.private)
    if do_space:
        publish_space(api, private=args.private)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
