# make_blank_images_ceus.py
# -*- coding: utf-8 -*-
"""
CEUS JSON을 읽어 blank.png가 필요한 pid마다
pid/_/blank.png (512x512, 1채널 검은색) 를 생성하고,
어떤 pid에 생성했는지 보고서를 저장한다.
"""

from pathlib import Path
import json
from PIL import Image

# =========================
# 전역변수 (여기만 수정하세요)
# =========================
DATA_ROOT   = Path("/home/introai21/mmtracking/data/CEUS/Data")              # CEUS 데이터 루트
ORG_JSON    = Path("/home/introai21/mmtracking/data/CEUS/Annotations/post_padding.json")  # org json 경로 (없으면 None)
AUG_JSON    = Path("/home/introai21/mmtracking/data/CEUS/Annotations/post_padding_aug.json")  # aug json 경로 (없으면 None)
OUTPUT_DIR  = Path("/home/introai21/mmtracking/data/CEUS")   # 레포트/로그를 저장할 폴더
REPORT_NAME = "created_blank_pids.txt"       # 레포트 파일명
# =========================


def load_records(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 단건 dict일 수도, 리스트일 수도 있음
    if isinstance(data, dict):
        return [data]
    return data


def ensure_blank_png(out_png: Path) -> bool:
    """out_png 경로에 1채널 512x512 검은 이미지(blank.png)를 생성한다.
    이미 존재하면 생성하지 않고 False 반환, 새로 만들면 True."""
    if out_png.exists():
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # 1채널(L) 512x512, 검은색(0)
    img = Image.new(mode="L", size=(512, 512), color=0)
    img.save(out_png)
    return True


def process_split(json_path: Path, variant: str, created_log: list, skipped_log: list):
    """variant: 'org' 또는 'aug'"""
    if not json_path or not json_path.exists():
        return

    records = load_records(json_path)

    for rec in records:
        filenames = rec.get("filename", [])
        if not isinstance(filenames, list):
            continue

        # blank.png가 등장하는 pid만 처리
        if "blank.png" not in filenames:
            continue

        fold = rec.get("fold")
        category = rec.get("category_id")  # 'FNH' or 'HCC'
        pid = rec.get("pid")

        if fold is None or not category or not pid:
            skipped_log.append(f"[WARN] Missing keys: fold/category/pid — raw pid={rec.get('pid')}")
            continue

        # fold_k/variant/category/pid/_/blank.png
        out_dir = DATA_ROOT / f"fold_{fold}" / variant / category / pid / "_"
        out_png = out_dir / "blank.png"

        created = ensure_blank_png(out_png)
        if created:
            created_log.append(str(out_png))
        else:
            skipped_log.append(f"[SKIP exists] {out_png}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / REPORT_NAME

    created_log, skipped_log = [], []

    # org / aug 각각 처리 (존재하는 것만)
    if ORG_JSON:
        process_split(ORG_JSON, "org", created_log, skipped_log)
    if AUG_JSON:
        process_split(AUG_JSON, "aug", created_log, skipped_log)

    # 레포트 작성: 어떤 pid(경로)에 생성했는지
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Created blank.png files (1-channel, 512x512)\n")
        for p in created_log:
            f.write(p + "\n")

        f.write("\n# Skipped (already existed or warnings)\n")
        for p in skipped_log:
            f.write(p + "\n")

    print(f"[Done] created={len(created_log)}, skipped={len(skipped_log)}")
    print(f"[Report] {report_path.resolve()}")


if __name__ == "__main__":
    main()