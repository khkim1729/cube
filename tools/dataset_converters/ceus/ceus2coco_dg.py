# ceus2coco_dg.py
# 기존 ceus json을, 모델에 입력할 수 있는 형태의 json으로 바꿈

import os, json
from ast import literal_eval
from typing import List, Dict, Any

# CONFIG 전역 변수 설정
DATA_ROOT      = "data/CEUS"
OUT_DIR        = os.path.join(DATA_ROOT, "annotations")
OUT_TRAIN      = "ceus_dg_train.json"
OUT_VAL        = "ceus_dg_val.json"

META_ORG_PATH  = os.path.join(DATA_ROOT, "Annotations", "post_padding.json")
META_AUG_PATH  = os.path.join(DATA_ROOT, "Annotations", "post_padding_aug.json")

VAL_FOLD       = 4     # validation fold
TEST_FOLD      = 0     # test fold
IMG_H, IMG_W   = 512, 512
ORG_DIRNAME    = "org"
AUG_DIRNAME    = "aug"

# 카테고리 정의 (클래스 1개)
CATEGORIES = [{"id": 1, "name": "lesion"}]
CAT2ID = {"FNH": 1, "HCC": 1}

# Phase 매핑 (AP=0, PP, LP=1, KP=2)
PHASE2ID = {"AP": 0, "PP": 1, "LP": 1, "KP": 2}
UNKNOWN_PHASE_ID = -1  # 유효하지 않은 phase에 대한 값

# 유틸 함수
def load_meta(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["content"] if isinstance(obj, dict) and "content" in obj else obj

def is_blank(fname: str) -> bool:
    return fname.lower() == "blank.png"

# CEUS -> CocoVID 변환 로직
def to_cocovid(entries: List[Dict[str, Any]], include_folds: List[int]) -> Dict[str, Any]:
    videos, images, anns = [], [], []
    vid_id, img_id, ann_id, global_inst_id = 1, 1, 1, 1

    for item in entries:
        cat = str(item["category_id"])
        if cat not in CAT2ID:
            continue
        fold = int(item["fold"])
        if fold not in include_folds:
            continue

        pid = str(item["pid"])
        phases   = item.get("phase", [])
        fnames   = item.get("filename", [])
        bbox_str = item.get("bbox", [])
        is_aug_l = item.get("is_aug", [False]*len(fnames))
        t1_min   = item.get("t1_min", [None]*len(fnames))
        t1_sec   = item.get("t1_sec", [None]*len(fnames))

        aug_flag_all = all(bool(x) for x in is_aug_l)
        video_name = f"fold_{fold}/{'aug' if aug_flag_all else 'org'}/{cat}/{pid}"
        
        # === video 단위 생성 ===
        videos.append({
            "id": vid_id,
            "name": video_name,
            "pid": pid, "fold": fold, "category": cat,
            "is_aug": aug_flag_all
        })
        
        inst_id = global_inst_id
        global_inst_id += 1

        for t, (phase, fname, bb, aug_flag) in enumerate(zip(phases, fnames, bbox_str, is_aug_l)):
            phase_dir = phase if (phase and phase != "-") else "_"
            rel_path = os.path.join(
                f"fold_{fold}",
                "aug" if aug_flag else "org",
                cat, pid, phase_dir, fname
            ).replace("\\", "/")

            # Phase ID 변환 (AP=0, PP, LP=1, KP=2)
            phase_id = PHASE2ID.get(phase, UNKNOWN_PHASE_ID)

            # 유효성 플래그: blank.png이거나 phase가 유효하지 않으면 invalid
            is_valid = not is_blank(fname) and phase_id != UNKNOWN_PHASE_ID

            images.append({
                "id": img_id,
                "file_name": rel_path,
                "height": IMG_H, "width": IMG_W,
                "frame_id": t, "video_id": vid_id,
                "pid": pid, "phase": phase, "phase_id": phase_id, "fold": fold,
                "is_aug": bool(aug_flag),
                "is_valid": is_valid,
                "t1_min": t1_min[t] if t < len(t1_min) else None,
                "t1_sec": t1_sec[t] if t < len(t1_sec) else None
            })

            if (fname.lower() != "blank.png") and bb and bb != "[]":
                try:
                    x1,y1,x2,y2 = list(map(int, literal_eval(bb)))
                    w,h = max(0, x2-x1), max(0, y2-y1)
                except Exception:
                    x1=y1=w=h=0
                if w > 0 and h > 0 and phase != "-":
                    anns.append({
                        "id": ann_id,
                        "video_id": vid_id,
                        "image_id": img_id,
                        "category_id": CAT2ID[cat],
                        "instance_id": inst_id,
                        "bbox": [x1, y1, w, h],
                        "area": int(w*h),
                        "iscrowd": False,
                        "occluded": False,
                        "generated": False
                    })
                    ann_id += 1
            img_id += 1
        vid_id += 1

    info_block = {
        "description": "CEUS liver dataset (1 Class: lesion)",
        "date_created": "2025-12-03",
        "contributor": "SNU Medical AI Lab CEUS Team",
        "version": "2.0"
    }

    return {
        "info": info_block,
        "licenses": [],
        "videos": videos,
        "images": images,
        "annotations": anns,
        "categories": CATEGORIES
    }

# 메인 실행
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_org = load_meta(META_ORG_PATH)
    meta_aug = load_meta(META_AUG_PATH)

    ALL_FOLDS = {0, 1, 2, 3, 4}
    train_folds = sorted(list(ALL_FOLDS - {VAL_FOLD, TEST_FOLD}))
    val_folds   = [VAL_FOLD]

    print(f"Train folds: {train_folds}, Val fold: {VAL_FOLD}, Test fold: {TEST_FOLD}")

    # ----- Train -----
    train_entries = meta_org + meta_aug
    coco_train = to_cocovid(train_entries, include_folds=train_folds)
    out_train = os.path.join(OUT_DIR, OUT_TRAIN)
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(coco_train, f, ensure_ascii=False)
    print(f"[OK] {OUT_TRAIN} → {out_train}")
    print(f"  videos={len(coco_train['videos'])}, images={len(coco_train['images'])}, anns={len(coco_train['annotations'])}")

    # ----- Val -----
    val_entries = meta_org + meta_aug
    coco_val = to_cocovid(val_entries, include_folds=val_folds)
    out_val = os.path.join(OUT_DIR, OUT_VAL)
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(coco_val, f, ensure_ascii=False)
    print(f"[OK] {OUT_VAL} → {out_val}")
    print(f"  videos={len(coco_val['videos'])}, images={len(coco_val['images'])}, anns={len(coco_val['annotations'])}")

if __name__ == "__main__":
    main()
