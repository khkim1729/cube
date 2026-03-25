"""
CUBE 실험 결과 CSV 병합 스크립트.
experiments/results/ 아래 모든 개별 실험 CSV를 combined_results.csv로 합침.

사용법:
    python experiments/merge_results.py [--results_dir experiments/results]
"""

import argparse
import csv
import glob
from pathlib import Path


def merge_csv(results_dir: str = "experiments/results") -> Path:
    out_dir = Path(results_dir)
    csv_files = sorted(glob.glob(str(out_dir / "*.csv")))
    csv_files = [f for f in csv_files if "combined" not in Path(f).name]

    if not csv_files:
        print("No CSV files found.")
        return None

    combined_path = out_dir / "combined_results.csv"
    all_rows = []
    header = None

    for f in csv_files:
        with open(f, newline="") as fp:
            reader = csv.DictReader(fp)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    # Sort by timestamp
    all_rows.sort(key=lambda r: (r.get("baseline",""), r.get("budget",""), int(r.get("checkpoint_idx", 0))))

    with open(combined_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged {len(all_rows)} rows from {len(csv_files)} files → {combined_path}")

    # Print summary table
    print(f"\n{'baseline':<12} {'budget':<16} {'ckpts':<6} "
          f"{'total_bias(last)':<20} {'fusion_bias(last)':<20} {'HL_proxy(last)':<20} {'reward(last)'}")
    print("-" * 110)

    from collections import defaultdict
    runs = defaultdict(list)
    for row in all_rows:
        key = (row["baseline"], row["budget"])
        runs[key].append(row)

    for (bl, bg), rows in sorted(runs.items()):
        last = rows[-1]
        print(
            f"{bl:<12} {bg:<16} {len(rows):<6} "
            f"{float(last['total_bias_norm']):<20.6e} "
            f"{float(last['fusion_bias_proj_mean']):<20.6e} "
            f"{float(last['HL_proxy_mean']):<20.4e} "
            f"{float(last['reward_mean']):.3f}"
        )

    return combined_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results")
    args = parser.parse_args()
    merge_csv(args.results_dir)
