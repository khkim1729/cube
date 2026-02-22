#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert CEUS COCO-VID c2 annotations (FNH/HCC) -> HCC-only jsons (batch for fold0~4, train/val/test).

Policy:
- Keep only videos that contain at least one HCC annotation.
- Keep ALL images belonging to those kept videos (preserve temporal structure).
- Keep only HCC annotations.
- Remap HCC category_id to 1.
- categories list will contain only:
    { "id": 1, "name": "HCC" }
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, Any, List, Set


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False)


def find_cat_id(categories: List[Dict[str, Any]], target_name: str) -> int:
    for c in categories:
        if str(c.get("name", "")).lower() == target_name.lower():
            return int(c["id"])
    raise ValueError(f"Category '{target_name}' not found.")


def convert_one(in_path: str, out_path: str, dry_run: bool = False) -> Dict[str, Any]:
    d = load_json(in_path)

    for k in ["videos", "images", "annotations", "categories"]:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {in_path}")

    hcc_cat_id = find_cat_id(d["categories"], "HCC")

    # 1️⃣ HCC annotation만 선택
    anns_keep = [
        a for a in d["annotations"]
        if int(a.get("category_id", -999)) == hcc_cat_id
    ]

    img_by_id = {int(im["id"]): im for im in d["images"]}

    # 2️⃣ HCC가 존재하는 video만 유지
    keep_video_ids: Set[int] = set()
    for a in anns_keep:
        img_id = int(a["image_id"])
        im = img_by_id.get(img_id)
        if im is None:
            raise KeyError(f"annotation references missing image_id={img_id}")
        keep_video_ids.add(int(im["video_id"]))

    videos_keep = [v for v in d["videos"] if int(v["id"]) in keep_video_ids]

    # 3️⃣ 해당 video의 모든 image 유지
    images_keep = [im for im in d["images"] if int(im["video_id"]) in keep_video_ids]
    keep_image_ids = set(int(im["id"]) for im in images_keep)

    # 4️⃣ annotation도 image 기준으로 필터
    anns_keep = [
        a for a in anns_keep
        if int(a["image_id"]) in keep_image_ids
    ]

    # 5️⃣ 🔥 category_id를 1로 재매핑
    for a in anns_keep:
        a["category_id"] = 1

    # 6️⃣ categories 재구성
    categories_keep = [
        {
            "id": 1,
            "name": "HCC",
            "supercategory": "lesion"
        }
    ]

    out = dict(d)
    out["videos"] = videos_keep
    out["images"] = images_keep
    out["annotations"] = anns_keep
    out["categories"] = categories_keep

    report = {
        "videos_in": len(d["videos"]),
        "videos_out": len(videos_keep),
        "images_in": len(d["images"]),
        "images_out": len(images_keep),
        "anns_in": len(d["annotations"]),
        "anns_out": len(anns_keep),
        "ann_cat_dist_out": dict(Counter(a["category_id"] for a in anns_keep)),
    }

    if not dry_run:
        save_json(out, out_path)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/CEUS/annotations/video/c2")
    ap.add_argument("--dst_dir", default="data/CEUS/annotations/video/hcc")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    os.makedirs(args.dst_dir, exist_ok=True)

    ok, fail = 0, 0

    for fold in folds:
        for split in splits:
            in_name = f"ceus_video_c2_fold{fold}_{split}.json"
            out_name = f"ceus_video_hcc_fold{fold}_{split}.json"

            in_path = os.path.join(args.src_dir, in_name)
            out_path = os.path.join(args.dst_dir, out_name)

            try:
                rep = convert_one(in_path, out_path, dry_run=args.dry_run)
                print(f"[OK] fold={fold} split={split} "
                      f"videos={rep['videos_out']} "
                      f"images={rep['images_out']} "
                      f"anns={rep['anns_out']} -> {out_path}")
                ok += 1
            except Exception as e:
                print(f"[FAIL] fold={fold} split={split} err={e}")
                fail += 1

    print(f"\nDone. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()