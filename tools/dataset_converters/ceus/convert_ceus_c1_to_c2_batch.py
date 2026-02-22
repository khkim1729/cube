#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple


FNH_ID = 1
HCC_ID = 2


def infer_class_from_file_name(file_name: str) -> int:
    """
    Infer class id from COCO image file_name.
    Expected to contain '/FNH/' or '/HCC/' somewhere in the path.
    """
    # normalize separators just in case
    p = file_name.replace("\\", "/")
    if "/FNH/" in p:
        return FNH_ID
    if "/HCC/" in p:
        return HCC_ID
    # sometimes might be '.../FNH' at end (rare), still handle:
    if p.endswith("/FNH") or p.endswith("/FNH/"):
        return FNH_ID
    if p.endswith("/HCC") or p.endswith("/HCC/"):
        return HCC_ID

    raise ValueError(
        f"Cannot infer class from file_name (need '/FNH/' or '/HCC/'): {file_name}"
    )


def convert_c1_to_c2(coco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Convert a c1 (single lesion) video COCO to c2 (FNH/HCC) by only changing:
      - annotations[*].category_id
      - categories
    Everything else is preserved.
    Returns converted coco and stats dict.
    """
    if "images" not in coco or "annotations" not in coco:
        raise KeyError("COCO json must contain 'images' and 'annotations'")

    # map image_id -> class_id
    imgid2cid: Dict[int, int] = {}
    for img in coco["images"]:
        if "id" not in img or "file_name" not in img:
            raise KeyError("Each image must have 'id' and 'file_name'")
        cid = infer_class_from_file_name(img["file_name"])
        imgid2cid[int(img["id"])] = cid

    # rewrite categories
    coco_out = coco  # in-place okay, but we’ll treat as output
    coco_out["categories"] = [
        {"id": FNH_ID, "name": "FNH"},
        {"id": HCC_ID, "name": "HCC"},
    ]

    # rewrite annotation category_id based on its image_id
    n_fnh = 0
    n_hcc = 0
    for ann in coco_out["annotations"]:
        if "image_id" not in ann:
            raise KeyError("Each annotation must have 'image_id'")
        image_id = int(ann["image_id"])
        if image_id not in imgid2cid:
            raise KeyError(f"Annotation refers to unknown image_id={image_id}")
        new_cid = imgid2cid[image_id]
        ann["category_id"] = new_cid
        if new_cid == FNH_ID:
            n_fnh += 1
        else:
            n_hcc += 1

    stats = {
        "images": len(coco_out["images"]),
        "annotations": len(coco_out["annotations"]),
        "ann_fnh": n_fnh,
        "ann_hcc": n_hcc,
    }
    return coco_out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        default="data/CEUS/annotations/video/c1",
        help="Input directory containing ceus_video_c1_fold{k}_{split}.json",
    )
    ap.add_argument(
        "--out-dir",
        default="data/CEUS/annotations/video/c2",
        help="Output directory for ceus_video_c2_fold{k}_{split}.json",
    )
    ap.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated folds")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print what would happen",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any annotation/image cannot be inferred. (default True behavior anyway)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip() != ""]
    splits = [x.strip() for x in args.splits.split(",") if x.strip() != ""]

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0

    for k in folds:
        for split in splits:
            in_path = in_dir / f"ceus_video_c1_fold{k}_{split}.json"
            out_path = out_dir / f"ceus_video_c2_fold{k}_{split}.json"

            if not in_path.exists():
                print(f"[SKIP] missing input: {in_path}")
                total_fail += 1
                continue

            try:
                with open(in_path, "r") as f:
                    coco = json.load(f)

                coco2, stats = convert_c1_to_c2(coco)

                print(
                    f"[OK] fold={k} split={split} "
                    f"images={stats['images']} anns={stats['annotations']} "
                    f"ann_fnh={stats['ann_fnh']} ann_hcc={stats['ann_hcc']} "
                    f"-> {out_path}"
                )

                if not args.dry_run:
                    with open(out_path, "w") as f:
                        json.dump(coco2, f)
                total_ok += 1

            except Exception as e:
                print(f"[FAIL] fold={k} split={split}  file={in_path}\n  {type(e).__name__}: {e}")
                total_fail += 1

    print(f"\nDone. ok={total_ok} fail={total_fail}")
    if total_fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()