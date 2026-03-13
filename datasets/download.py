"""
Dataset downloader for CUBE experiments.

Downloads and preprocesses multimodal reasoning datasets from HuggingFace Hub.
Datasets used in CUBE experiments (VLM-RL benchmarks):

  Visual QA / Reasoning:
    - HuggingFaceM4/NoCaps                    (image captioning)
    - HuggingFaceM4/VQAv2                     (visual question answering)
    - Multimodal-Fatima/OK-VQA_train          (knowledge-based VQA)
    - merve/vqav2-small                       (VQA v2 small split)

  Math / Multimodal Reasoning:
    - AI4Math/MathVista                       (multimodal math reasoning)
    - HuggingFaceM4/ChartQA                   (chart understanding)
    - lmms-lab/MMBench                        (multi-task benchmark)
    - Lin-Chen/MMStar                         (hard multimodal reasoning)
    - lmms-lab/MMMU_Pro                       (college-level multi-discipline)

  Science / OCR:
    - HuggingFaceM4/ScienceQAImg_Modif        (science with images)
    - Yelp/yelp_review_full                   (text-only baseline)

Usage:
    python datasets/download.py --dataset mathvista --split test --output_dir datasets/

    from datasets.download import download_dataset
    data = download_dataset("mathvista", split="test")
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


DATASET_REGISTRY = {
    "mathvista": {
        "hf_name": "AI4Math/MathVista",
        "splits": ["testmini", "test"],
        "task": "math_reasoning",
        "modality": "image+text",
    },
    "mmstar": {
        "hf_name": "Lin-Chen/MMStar",
        "splits": ["val"],
        "task": "multimodal_reasoning",
        "modality": "image+text",
    },
    "chartqa": {
        "hf_name": "HuggingFaceM4/ChartQA",
        "splits": ["train", "val", "test"],
        "task": "chart_qa",
        "modality": "image+text",
    },
    "mmbench": {
        "hf_name": "lmms-lab/MMBench",
        "splits": ["dev_en", "test_en"],
        "task": "multi_task",
        "modality": "image+text",
    },
    "scienceqa": {
        "hf_name": "HuggingFaceM4/ScienceQAImg_Modif",
        "splits": ["train", "validation", "test"],
        "task": "science_qa",
        "modality": "image+text",
    },
    "mmmu_pro": {
        "hf_name": "lmms-lab/MMMU_Pro",
        "splits": ["validation", "test"],
        "task": "college_reasoning",
        "modality": "image+text",
    },
    "vqav2": {
        "hf_name": "HuggingFaceM4/VQAv2",
        "splits": ["train", "validation"],
        "task": "visual_qa",
        "modality": "image+text",
    },
    "okvqa": {
        "hf_name": "Multimodal-Fatima/OK-VQA_train",
        "splits": ["train"],
        "task": "knowledge_vqa",
        "modality": "image+text",
    },
}


def download_dataset(
    name: str,
    split: str = "test",
    output_dir: str = "datasets/",
    cache_dir: Optional[str] = None,
) -> "datasets.Dataset":  # type: ignore
    """Download a dataset by registry name.

    Args:
        name       : registry key (see DATASET_REGISTRY)
        split      : dataset split name
        output_dir : where to save a local copy (JSON)
        cache_dir  : HuggingFace cache directory

    Returns:
        HuggingFace Dataset object
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the `datasets` package: pip install datasets")

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[name]
    hf_name = info["hf_name"]
    available_splits = info["splits"]

    if split not in available_splits:
        raise ValueError(
            f"Split '{split}' not available for '{name}'. "
            f"Available splits: {available_splits}"
        )

    print(f"Downloading {hf_name} ({split})...")
    ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)

    # Save JSON metadata
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)
    meta_file = out_path / f"{split}_meta.json"
    with open(meta_file, "w") as f:
        json.dump(
            {
                "name": name,
                "hf_name": hf_name,
                "split": split,
                "num_samples": len(ds),
                "task": info["task"],
                "modality": info["modality"],
                "features": list(ds.features.keys()),
            },
            f,
            indent=2,
        )
    print(f"Saved metadata to {meta_file}")
    return ds


def list_datasets():
    """Print all available datasets."""
    print(f"{'Name':<15} {'HF Hub':<40} {'Task':<25} {'Splits'}")
    print("-" * 100)
    for name, info in DATASET_REGISTRY.items():
        splits = ", ".join(info["splits"])
        print(f"{name:<15} {info['hf_name']:<40} {info['task']:<25} {splits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CUBE experiment datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (see --list for options)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="datasets/")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list or args.dataset is None:
        list_datasets()
    else:
        download_dataset(
            args.dataset,
            split=args.split,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
