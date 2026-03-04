#!/usr/bin/env python3
"""
CUBE Result Analyzer — 모든 CSV를 읽어 (baseline×budget) 조합별 통계 출력.

사용법:
    python experiments/analyze_results.py [--results_dir experiments/results]
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

COMBO_ORDER = [
    ("reinforce", "none"),
    ("reinforce", "prompt_skip"),
    ("reinforce", "subset_select"),
    ("grpo",      "none"),
    ("grpo",      "prompt_skip"),
    ("grpo",      "subset_select"),
    ("rloo",      "none"),
    ("rloo",      "prompt_skip"),
    ("rloo",      "subset_select"),
    ("stv",       "none"),
    ("stv",       "prompt_skip"),
    ("stv",       "subset_select"),
]

METRICS = [
    "total_bias_norm",
    "fusion_bias_proj_mean",
    "HL_proxy_mean",
]

BASELINE_LABEL = {
    "reinforce": "REINFORCE",
    "grpo":      "GRPO",
    "rloo":      "RLOO",
    "stv":       "STV",
}
BUDGET_LABEL = {
    "none":          "None",
    "prompt_skip":   "PSkip",
    "subset_select": "SubSel",
}


def load_all_runs(results_dir: Path):
    """결과 디렉토리의 모든 CSV를 읽어, (baseline, budget) 별로 마지막 체크포인트 값 수집."""
    runs = defaultdict(list)  # key=(baseline, budget), value=list of {metric: val}

    for csv_file in sorted(results_dir.glob("*.csv")):
        if csv_file.name in ("combined_results.csv",):
            continue
        try:
            with open(csv_file, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            last = rows[-1]  # 마지막 체크포인트
            bl = last.get("baseline", "")
            bg = last.get("budget", "")
            if not bl or not bg:
                continue
            key = (bl, bg)
            record = {}
            for m in METRICS:
                val = last.get(m, None)
                if val is not None:
                    try:
                        record[m] = float(val)
                    except (ValueError, TypeError):
                        pass
            if record:
                runs[key].append(record)
        except Exception as e:
            print(f"  Warning: {csv_file.name}: {e}")
            continue

    return runs


def fmt_val(v, metric):
    """메트릭에 따라 적절한 형식으로 출력."""
    if "HL_proxy" in metric:
        if abs(v) >= 1e9:
            return f"{v:.2e}"
        return f"{v:.2e}"
    if "bias" in metric:
        if abs(v) >= 0.1:
            return f"{v:.3f}"
        if abs(v) >= 1e-3:
            return f"{v:.2e}"
        if abs(v) == 0:
            return "0.000"
        return f"{v:.2e}"
    return f"{v:.3e}"


def stats(vals):
    import statistics
    if not vals:
        return None, None, None
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = statistics.stdev(vals)
    else:
        std = 0.0
    return mean, std, len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_all_runs(results_dir)

    print(f"\n{'='*90}")
    print(f"  CUBE Simulation Results — {results_dir}")
    print(f"  총 CSV 파일: {sum(len(v) for v in runs.values())} runs across {len(runs)} combos")
    print(f"{'='*90}")

    # ─── 메인 결과 표 ───
    header = f"  {'Baseline':<12} {'Budget':<10} {'n_runs':>6}  {'Bias (mean±std)':<22} {'Fusion (mean±std)':<22} {'HL_proxy (mean)':<18}"
    print(f"\n{header}")
    print(f"  {'-'*90}")

    for bl, bg in COMBO_ORDER:
        key = (bl, bg)
        recs = runs.get(key, [])
        n = len(recs)
        if n == 0:
            print(f"  {BASELINE_LABEL[bl]:<12} {BUDGET_LABEL[bg]:<10} {'0':>6}  (no data)")
            continue

        bias_vals   = [r["total_bias_norm"] for r in recs if "total_bias_norm" in r]
        fusion_vals = [r["fusion_bias_proj_mean"] for r in recs if "fusion_bias_proj_mean" in r]
        hl_vals     = [r["HL_proxy_mean"] for r in recs if "HL_proxy_mean" in r]

        b_m, b_s, _ = stats(bias_vals)
        f_m, f_s, _ = stats(fusion_vals)
        h_m, h_s, _ = stats(hl_vals)

        bias_str   = f"{b_m:.3e} ± {b_s:.1e}" if b_m is not None else "—"
        fusion_str = f"{f_m:.2e} ± {f_s:.1e}" if f_m is not None else "—"
        hl_str     = f"{h_m:.2e}" if h_m is not None else "—"

        print(f"  {BASELINE_LABEL[bl]:<12} {BUDGET_LABEL[bg]:<10} {n:>6}  {bias_str:<22} {fusion_str:<22} {hl_str:<18}")

    print(f"\n{'='*90}")

    # ─── Key statistics ───
    print("\n  [Key observations]")
    grpo_hl = [r["HL_proxy_mean"] for (bl,bg), recs in runs.items()
               if bl=="grpo" for r in recs if "HL_proxy_mean" in r]
    nongrpo_hl = [r["HL_proxy_mean"] for (bl,bg), recs in runs.items()
                  if bl!="grpo" for r in recs if "HL_proxy_mean" in r]
    if grpo_hl and nongrpo_hl:
        ratio = min(grpo_hl) / max(nongrpo_hl)
        print(f"  GRPO HL range:     {min(grpo_hl):.2e} – {max(grpo_hl):.2e}")
        print(f"  non-GRPO HL range: {min(nongrpo_hl):.2e} – {max(nongrpo_hl):.2e}")
        print(f"  min(GRPO) / max(non-GRPO) ratio: {ratio:.2e}")

    # Fusion bias: check zero conditions
    fusion_reinforce = [r["fusion_bias_proj_mean"]
                        for (bl,bg), recs in runs.items()
                        if bl=="reinforce" for r in recs if "fusion_bias_proj_mean" in r]
    fusion_none = [r["fusion_bias_proj_mean"]
                   for (bl,bg), recs in runs.items()
                   if bg=="none" for r in recs if "fusion_bias_proj_mean" in r]
    if fusion_reinforce:
        print(f"  REINFORCE fusion bias (all budgets, all runs): max={max(map(abs,fusion_reinforce)):.2e}")
    if fusion_none:
        print(f"  None-budget fusion bias (all baselines, all runs): max={max(map(abs,fusion_none)):.2e}")


if __name__ == "__main__":
    main()
