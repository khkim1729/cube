import argparse, json, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # train: epoch별 loss 리스트
    train_losses = defaultdict(list)
    # val: mAP(step 기반)
    val_steps = []
    val_maps  = []

    with open(args.json, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            # train 라인: epoch와 loss가 존재
            if "epoch" in j and "loss" in j:
                train_losses[int(j["epoch"])].append(float(j["loss"]))
            # val 라인: coco/bbox_mAP 만 있음(여기엔 epoch/ loss 없음)
            elif "coco/bbox_mAP_50" in j and "step" in j:
                val_steps.append(int(j["step"]))
                val_maps.append(float(j["coco/bbox_mAP_50"]))

    # 1) train epoch 평균 loss 플롯
    epochs = sorted(train_losses.keys())
    avg_loss = [float(np.mean(train_losses[e])) for e in epochs]

    plt.figure()
    plt.plot(epochs, avg_loss, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("avg loss")
    plt.title("Epoch-wise Average Training Loss")
    plt.grid(True)
    out1 = os.path.join(args.out_dir, "epoch_avg_loss.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out1}")

    # 2) (옵션) val mAP 플롯 (있을 때만)
    if val_steps and val_maps:
        plt.figure()
        plt.plot(val_steps, val_maps, marker="o")
        plt.xlabel("step (validation)")
        plt.ylabel("coco/bbox_mAP_50")
        plt.title("Validation mAP")
        plt.grid(True)
        out2 = os.path.join(args.out_dir, "val_map.png")
        plt.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {out2}")

if __name__ == "__main__":
    main()