"""
Inference code for VLM CUBE experiment checkpoints.

Usage:
    python experiments/inference_vlm.py \\
        --ckpt_path experiments/results_vlm/checkpoints/best \\
        --dataset mathvista \\
        --n_samples 100 \\
        --gpu_id 1 \\
        --output inference_result.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from experiments.vlm_utils import (
    load_vlm_dataset,
    build_qwen_prompt,
    extract_answer,
    compute_reward,
)


def load_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load a saved VLM checkpoint (model + processor)."""
    ckpt_path = Path(ckpt_path)
    print(f"Loading checkpoint from {ckpt_path}...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    
    return model, processor


def run_inference(
    model,
    processor,
    items,
    max_new_tokens: int = 256,
    temperature: float = 0.0,  # greedy decoding
    device: str = "cuda",
):
    """Run inference on items and compute rewards.
    
    Returns:
        results: list of dicts with keys: question, gold, response, reward, correct
    """
    results = []
    model.eval()
    
    with torch.no_grad():
        for idx, item in enumerate(items):
            messages = build_qwen_prompt(item)
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if item.get("image") is not None:
                inputs = processor(
                    text=[text_prompt],
                    images=[item["image"]],
                    return_tensors="pt",
                ).to(device)
            else:
                inputs = processor(
                    text=[text_prompt],
                    return_tensors="pt",
                ).to(device)
            
            output = model.generate(
                **inputs,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            gen_ids = output[0, inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            reward = compute_reward(response, item["gold"], item["type"])
            
            results.append({
                "idx": idx,
                "question": item["question"][:100],  # truncate for readability
                "gold": item["gold"],
                "response": response,
                "reward": float(reward),
                "correct": bool(reward > 0.5),
            })
            
            if (idx + 1) % max(1, len(items) // 10) == 0:
                print(f"  [{idx+1}/{len(items)}] accuracy={sum(r['correct'] for r in results) / len(results):.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VLM inference on checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint (e.g., experiments/results_vlm/checkpoints/best)")
    parser.add_argument("--dataset", type=str, default="mathvista",
                        choices=["mathvista", "mmstar", "chartqa", "scienceqa", "mmbench"])
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--output", type=str, default="inference_result.json")
    
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    
    # Load checkpoint
    model, processor = load_checkpoint(args.ckpt_path, device=device)
    
    # Load dataset
    print(f"Loading {args.dataset}...")
    items = load_vlm_dataset(args.dataset, split=None, n_pool=args.n_samples)
    
    # Run inference
    print(f"Running inference on {len(items)} samples...")
    results = run_inference(model, processor, items, device=device)
    
    # Compute metrics
    accuracy = sum(r['correct'] for r in results) / len(results)
    avg_reward = sum(r['reward'] for r in results) / len(results)
    
    # Save results
    output_path = Path(args.output)
    summary = {
        "ckpt_path": args.ckpt_path,
        "dataset": args.dataset,
        "n_samples": len(results),
        "accuracy": round(accuracy, 4),
        "avg_reward": round(avg_reward, 4),
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Done! Results saved to {output_path}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
