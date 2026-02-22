#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert CEUS COCO-VID c2 annotations (FNH/HCC) -> FNH-only jsons (batch for fold0~4, train/val/test).

Policy:
- Keep only videos that contain at least one FNH annotation.
- Keep ALL images belonging to those kept videos (preserve video temporal structure).
- Keep only FNH annotations (category_id == fnh_cat_id).
- Keep categories list with only FNH.
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
    # robust: case-insensitive match on "name"
    for c in categories:
        if str(c.get("name", "")).lower() == target_name.lower():
            return int(c["id"])
    raise ValueError(f"Category '{target_name}' not found in categories: {[c.get('name') for c in categories]}")


def convert_one(in_path: str, out_path: str, keep_name: str = "FNH", dry_run: bool = False) -> Dict[str, Any]:
    d = load_json(in_path)

    # basic checks
    for k in ["videos", "images", "annotations", "categories"]:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {in_path}")

    keep_cat_id = find_cat_id(d["categories"], keep_name)

    # 1) keep only annotations of keep_cat_id
    anns_keep = [a for a in d["annotations"] if int(a.get("category_id", -999)) == keep_cat_id]

    # 2) keep only videos that have at least one kept annotation (via images -> video_id)
    img_by_id = {int(im["id"]): im for im in d["images"]}

    keep_video_ids: Set[int] = set()
    for a in anns_keep:
        img_id = int(a["image_id"])
        im = img_by_id.get(img_id)
        if im is None:
            raise KeyError(f"annotation references missing image_id={img_id} in {in_path}")
        keep_video_ids.add(int(im["video_id"]))

    videos_keep = [v for v in d["videos"] if int(v["id"]) in keep_video_ids]

    # 3) keep all images belonging to those kept videos (preserve timeline)
    images_keep = [im for im in d["images"] if int(im["video_id"]) in keep_video_ids]
    keep_image_ids = set(int(im["id"]) for im in images_keep)

    # 4) keep only annotations whose image_id is in kept images (should already hold)
    anns_keep = [a for a in anns_keep if int(a["image_id"]) in keep_image_ids]

    # 5) categories: keep only FNH category entry (keep original id)
    categories_keep = [c for c in d["categories"] if int(c["id"]) == keep_cat_id]

    # rebuild json
    out = dict(d)  # shallow copy
    out["videos"] = videos_keep
    out["images"] = images_keep
    out["annotations"] = anns_keep
    out["categories"] = categories_keep

    # report
    cat_counter = Counter(int(a["category_id"]) for a in out["annotations"])
    report = {
        "in_path": in_path,
        "out_path": out_path,
        "keep_category": keep_name,
        "keep_cat_id": keep_cat_id,
        "videos_in": len(d["videos"]),
        "videos_out": len(out["videos"]),
        "images_in": len(d["images"]),
        "images_out": len(out["images"]),
        "anns_in": len(d["annotations"]),
        "anns_out": len(out["annotations"]),
        "ann_cat_dist_out": dict(cat_counter),
    }

    if not dry_run:
        save_json(out, out_path)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/CEUS/annotations/video/c2", help="directory of c2 jsons")
    ap.add_argument("--dst_dir", default="data/CEUS/annotations/video/fnh", help="output directory")
    ap.add_argument("--folds", default="0,1,2,3,4", help="comma-separated folds")
    ap.add_argument("--splits", default="train,val,test", help="comma-separated splits")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default="", help="optional path to write a report json")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    splits = [x.strip() for x in args.splits.split(",") if x.strip() != ""]

    os.makedirs(args.dst_dir, exist_ok=True)

    all_reports = []
    ok = 0
    fail = 0

    for fold in folds:
        for split in splits:
            in_name = f"ceus_video_c2_fold{fold}_{split}.json"
            out_name = f"ceus_video_fnh_fold{fold}_{split}.json"
            in_path = os.path.join(args.src_dir, in_name)
            out_path = os.path.join(args.dst_dir, out_name)

            try:
                rep = convert_one(in_path, out_path, keep_name="FNH", dry_run=args.dry_run)
                all_reports.append(rep)
                print(f"[OK] fold={fold} split={split} "
                      f"videos={rep['videos_out']} images={rep['images_out']} anns={rep['anns_out']} "
                      f"-> {out_path}")
                ok += 1
            except Exception as e:
                print(f"[FAIL] fold={fold} split={split} in={in_path} err={e}")
                fail += 1

    print(f"\nDone. ok={ok} fail={fail}")

    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"ok": ok, "fail": fail, "items": all_reports}, f, ensure_ascii=False, indent=2)
        print(f"[WROTE] report -> {args.report}")


if __name__ == "__main__":
    main()