"""
VLM Utilities for CUBE experiments (Qwen2-VL-7B-Instruct).

Provides:
  - load_qwen_model(lora_rank, device) → (model, processor)
  - load_vlm_dataset(name, split, n_pool) → list of dicts
  - build_qwen_prompt(item) → str
  - extract_answer(text, qtype) → str
  - compute_reward(response, gold, qtype) → float
  - generate_rollouts_vlm(model, processor, items, N, ...) → Rollouts
  - generate_rollouts_vlm_var(model, processor, items, N, M, ...) → Rollouts (variable N_j)
  - compute_log_probs_batch(model, processor, items, responses, device) → (M,) Tensor
  - probe_project_grads(param_grads, R, seed, device) → (R,) Tensor

Memory design:
  - probe_project_grads: generates N(0,I) block-by-block per parameter;
    never materializes full (R, d) matrix → O(max_block_size) extra memory.
  - generate_rollouts_vlm: sequential per-rollout generation (variable-length).
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rollouts dataclass (same interface as cube_sim.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rollouts:
    """Container for a single (X, Y) VLM minibatch."""
    items: list               # B items (question/image/gold dicts)
    responses: list           # M response strings (flat, ordered by prompt)
    rewards: torch.Tensor     # (M,) float
    prompt_ids: torch.Tensor  # (M,) which prompt [0..B)
    N_per_prompt: int
    B: int
    M: int
    N_list: Optional[List[int]] = None  # per-prompt counts; None = uniform N_per_prompt

    def prompt_slice(self, j: int) -> slice:
        """Return flat-array slice for prompt j."""
        if self.N_list is None:
            N = self.N_per_prompt
            return slice(j * N, (j + 1) * N)
        offset = sum(self.N_list[:j])
        return slice(offset, offset + self.N_list[j])

    def prompt_N(self, j: int) -> int:
        """Return rollout count for prompt j."""
        return self.N_per_prompt if self.N_list is None else self.N_list[j]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen_model(lora_rank: int = 16, device: str = "cuda"):
    """Load Qwen2-VL-7B-Instruct with LoRA adapter.

    Returns:
        model     : PeftModel with LoRA, in bf16
        processor : AutoProcessor for Qwen2-VL
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"  Loading {model_id} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "mathvista": {
        "hf_name": "AI4Math/MathVista",
        "split": "testmini",
        "gold_field": "answer",
        "question_field": "question",
        "image_field": "image",
        "type": "math",
    },
    "mmstar": {
        "hf_name": "Lin-Chen/MMStar",
        "split": "val",
        "gold_field": "answer",
        "question_field": "question",
        "image_field": "image",
        "type": "mcq",
    },
    "chartqa": {
        "hf_name": "HuggingFaceM4/ChartQA",
        "split": "test",
        "gold_field": "label",
        "question_field": "query",
        "image_field": "image",
        "type": "chart",
    },
    "mmbench": {
        "hf_name": "lmms-lab/MMBench_EN",
        "split": "test",
        "gold_field": "answer",
        "question_field": "question",
        "image_field": "image",
        "type": "mcq",
    },
    "scienceqa": {
        "hf_name": "HuggingFaceM4/ScienceQAImg_Modif",
        "split": "test",
        "gold_field": None,   # special: answer_idx → choices[idx]
        "question_field": "question",
        "image_field": "image",
        "type": "mcq",
    },
    "mmmu_pro": {
        "hf_name": "MMMU/MMMU_Pro",
        "config_name": "vision",
        "split": "test",
        "gold_field": "answer",
        "question_field": "question",
        "image_field": "image",
        "type": "mcq",
    },
}


def load_vlm_dataset(name: str, split: str = None, n_pool: int = 512) -> list:
    """Load a VLM dataset and return list of standardized item dicts.

    Each item: {"question": str, "image": PIL.Image or None, "gold": str, "type": str}
    """
    import sys as _sys
    import os as _os
    from pathlib import Path as _Path
    # Bypass local cube/datasets/ directory that shadows HuggingFace datasets
    _repo_root = str(_Path(__file__).parent.parent.resolve())
    _removed = [i for i, p in enumerate(_sys.path)
                if _os.path.abspath(p) == _repo_root]
    for i in reversed(_removed):
        _sys.path.pop(i)
    try:
        from datasets import load_dataset
    finally:
        for i, p in zip(_removed, [_repo_root] * len(_removed)):
            _sys.path.insert(i, p)

    cfg = DATASET_CONFIGS[name]
    ds_split = split or cfg["split"]
    print(f"  Loading dataset {cfg['hf_name']} split={ds_split} ...")
    config_name = cfg.get("config_name")
    if config_name is None:
        ds = load_dataset(cfg["hf_name"], split=ds_split, trust_remote_code=True)
    else:
        ds = load_dataset(cfg["hf_name"], config_name, split=ds_split, trust_remote_code=True)

    items = []
    for ex in ds:
        question = ex.get(cfg["question_field"], "")
        image = ex.get(cfg["image_field"], None)
        # MathVista stores a path string; use decoded_image (PIL) if available
        if isinstance(image, str):
            image = ex.get("decoded_image", None)

        # Gold answer
        if name == "scienceqa":
            choices = ex.get("choices", [])
            ans_idx = ex.get("answer", 0)
            gold = choices[ans_idx] if choices and ans_idx < len(choices) else str(ans_idx)
        else:
            gold = str(ex.get(cfg["gold_field"], ""))

        # For MCQ: append choices to question
        if name in ("mmstar", "mmbench") and "choices" in ex:
            choices_str = "\n".join(
                f"({chr(65+i)}) {c}" for i, c in enumerate(ex["choices"])
            )
            question = f"{question}\n{choices_str}"
        elif name == "scienceqa" and "choices" in ex:
            choices_str = "\n".join(
                f"({chr(65+i)}) {c}" for i, c in enumerate(ex["choices"])
            )
            question = f"{question}\n{choices_str}"

        items.append({
            "question": question,
            "image": image,
            "gold": gold,
            "type": cfg["type"],
        })

        if len(items) >= n_pool:
            break

    print(f"  Loaded {len(items)} items from {name}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prompt Building
# ─────────────────────────────────────────────────────────────────────────────

def build_qwen_prompt(item: dict) -> list:
    """Build Qwen2-VL chat messages list for processor.apply_chat_template.

    Returns list of message dicts (chat template format).
    """
    content = []
    if item.get("image") is not None:
        content.append({"type": "image", "image": item["image"]})
    content.append({"type": "text", "text": item["question"]})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# 5. Answer Extraction & Reward
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(text: str, qtype: str) -> str:
    """Extract final answer from model response text.

    Tries multiple patterns:
      1. \\boxed{...}
      2. "answer: X" / "Answer: X"
      3. "The answer is X"
      4. Last number (for math)
      5. Last uppercase letter A-E (for MCQ)
      6. Full text stripped
    """
    text = text.strip()

    # Boxed answer (LaTeX)
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # Explicit "answer: ..." pattern
    m = re.search(r"(?i)answer\s*[:=]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip().rstrip(".")

    # "The answer is ..."
    m = re.search(r"(?i)the answer is\s+(.+?)(?:\.|,|\n|$)", text)
    if m:
        return m.group(1).strip()

    if qtype in ("math",):
        # Last number in text
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
        if nums:
            return nums[-1]

    if qtype in ("mcq",):
        # Last standalone letter A-E
        letters = re.findall(r"\b([A-E])\b", text)
        if letters:
            return letters[-1]

    return text[:50]  # fallback: first 50 chars


def compute_reward(response: str, gold: str, qtype: str) -> float:
    """Compute binary verifiable reward: 1.0 if correct, 0.0 otherwise."""
    pred = extract_answer(response, qtype).strip().lower()
    gold_norm = gold.strip().lower()

    # Exact match
    if pred == gold_norm:
        return 1.0

    # Number comparison for math
    if qtype == "math":
        try:
            pred_f = float(pred.replace(",", ""))
            gold_f = float(gold_norm.replace(",", ""))
            if abs(pred_f - gold_f) < 1e-4 * max(1.0, abs(gold_f)):
                return 1.0
        except ValueError:
            pass

    # MCQ: accept if gold letter appears in pred
    if qtype == "mcq" and len(gold_norm) == 1 and gold_norm.isalpha():
        if gold_norm in pred[:5]:
            return 1.0

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rollout Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_rollouts_vlm(
    model,
    processor,
    items: list,          # B items
    N: int,               # rollouts per prompt
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: str = "cuda",
) -> Rollouts:
    """Generate N rollouts per prompt sequentially.

    Returns Rollouts dataclass with M = B*N responses and rewards.
    """
    B = len(items)
    M = B * N
    all_responses = []
    rewards_list = []

    model.eval()
    with torch.no_grad():
        for j, item in enumerate(items):
            messages = build_qwen_prompt(item)
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Prepare inputs
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

            for _ in range(N):
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
                all_responses.append(response)
                rewards_list.append(reward)

    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    prompt_ids = torch.arange(B, device=device).repeat_interleave(N)

    return Rollouts(
        items=items,
        responses=all_responses,
        rewards=rewards,
        prompt_ids=prompt_ids,
        N_per_prompt=N,
        B=B,
        M=M,
    )


def generate_rollouts_vlm_var(
    model,
    processor,
    items: list,          # B items
    N: int,               # base rollout count (uniform reference)
    M: int,               # total rollout budget (sum of all N_j)
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: str = "cuda",
    n_probe: int = 4,     # probe rollouts per prompt for allocation
) -> Rollouts:
    """Generate variable N_j rollouts per prompt, allocated by reward variance.

    Implements Adaptive Rollout Allocation for VLMs:
      1. Sample n_probe rollouts per prompt to estimate per-prompt accuracy.
      2. Compute weight_j = acc_j * (1 - acc_j) * 4  (Bernoulli variance proxy,
         maximized when acc_j = 0.5, i.e., prompt is near the learning boundary).
      3. Allocate N_j ∝ weight_j with sum(N_j) = M, at least 1 per prompt.
      4. Sample the actual N_j rollouts per prompt.
    """
    B = len(items)

    def _generate_n(item, n_rollouts):
        """Generate n_rollouts for a single item, return (responses, rewards)."""
        messages = build_qwen_prompt(item)
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if item.get("image") is not None:
            inputs = processor(
                text=[text_prompt], images=[item["image"]], return_tensors="pt",
            ).to(device)
        else:
            inputs = processor(text=[text_prompt], return_tensors="pt").to(device)
        resps, rews = [], []
        for _ in range(n_rollouts):
            output = model.generate(
                **inputs,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            gen_ids = output[0, inputs["input_ids"].shape[1]:]
            resp = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            resps.append(resp)
            rews.append(compute_reward(resp, item["gold"], item["type"]))
        return resps, rews

    # Step 1: probe rollouts to estimate per-prompt accuracy
    model.eval()
    probe_accs = []
    with torch.no_grad():
        for item in items:
            _, rews = _generate_n(item, n_probe)
            probe_accs.append(sum(rews) / len(rews))

    # Step 2: compute allocation weights (Bernoulli variance proxy)
    weights = [acc * (1.0 - acc) * 4.0 for acc in probe_accs]
    total_w = sum(weights)
    if total_w < 1e-8:
        weights = [1.0] * B
        total_w = float(B)
    weights = [w / total_w for w in weights]

    # Step 3: compute N_j with sum(N_j) == M, min 1 per prompt
    remaining = M - B
    extra = [int(round(w * remaining)) for w in weights]
    diff = remaining - sum(extra)
    if diff != 0:
        sorted_idx = sorted(range(B), key=lambda j: weights[j], reverse=True)
        for i in range(abs(diff)):
            j = sorted_idx[i % B]
            extra[j] = max(0, extra[j] + (1 if diff > 0 else -1))
    N_list = [1 + e for e in extra]  # guaranteed: sum == M, each >= 1

    # Step 4: sample actual N_j rollouts per prompt
    all_responses = []
    rewards_list = []
    model.eval()
    with torch.no_grad():
        for j, item in enumerate(items):
            resps, rews = _generate_n(item, N_list[j])
            all_responses.extend(resps)
            rewards_list.extend(rews)

    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    prompt_ids_list = []
    for j in range(B):
        prompt_ids_list.extend([j] * N_list[j])
    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=device)

    return Rollouts(
        items=items,
        responses=all_responses,
        rewards=rewards,
        prompt_ids=prompt_ids,
        N_per_prompt=N,
        B=B,
        M=sum(N_list),
        N_list=N_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Log-Probability Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_probs_batch(
    model,
    processor,
    items: list,          # B items (question+image)
    responses: list,      # M response strings (flat, ordered by prompt)
    N: int,               # uniform rollouts per prompt (ignored when N_list given)
    device: str = "cuda",
    N_list: Optional[List[int]] = None,  # per-prompt counts for variable N_j
) -> torch.Tensor:
    """Compute log π(y_m | x_m) for all M rollouts, with grad.

    Returns (M,) tensor of scalar log-probs (each with grad for backward).
    Supports variable N_j per prompt via N_list parameter.
    """
    B = len(items)
    log_pi_list = []

    for j, item in enumerate(items):
        N_j = N_list[j] if N_list is not None else N
        offset = sum(N_list[:j]) if N_list is not None else j * N

        messages = build_qwen_prompt(item)
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Call processor once per prompt to get all inputs (pixel_values, image_grid_thw, etc.)
        proc_out = processor(
            text=[text_prompt],
            images=[item["image"]] if item.get("image") else None,
            return_tensors="pt",
        ).to(device)
        prompt_ids = proc_out.input_ids  # (1, L_prompt)
        L_p = prompt_ids.shape[1]

        for i in range(N_j):
            resp = responses[offset + i]
            resp_ids = processor.tokenizer(
                resp, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)  # (1, L_resp)

            # Concatenate prompt + response tokens
            full_ids = torch.cat([prompt_ids, resp_ids], dim=1)  # (1, L_full)

            # Build model kwargs: extend attention_mask and mm_token_type_ids for resp tokens
            model_kwargs = {"input_ids": full_ids}
            if "attention_mask" in proc_out:
                extra_mask = torch.ones(
                    (1, resp_ids.shape[1]), device=device,
                    dtype=proc_out.attention_mask.dtype
                )
                model_kwargs["attention_mask"] = torch.cat(
                    [proc_out.attention_mask, extra_mask], dim=1
                )
            if "pixel_values" in proc_out:
                model_kwargs["pixel_values"] = proc_out.pixel_values
            if "image_grid_thw" in proc_out:
                model_kwargs["image_grid_thw"] = proc_out.image_grid_thw
            if "mm_token_type_ids" in proc_out:
                extra_type = torch.zeros(
                    (1, resp_ids.shape[1]), device=device,
                    dtype=proc_out.mm_token_type_ids.dtype
                )
                model_kwargs["mm_token_type_ids"] = torch.cat(
                    [proc_out.mm_token_type_ids, extra_type], dim=1
                )

            # Forward pass (with grad)
            out = model(**model_kwargs)

            logits = out.logits[0]  # (L_full, vocab)
            # Shift: logits[t] predicts token[t+1]
            shift_logits = logits[L_p - 1:-1]       # (L_resp, vocab)
            shift_labels = resp_ids[0]               # (L_resp,)

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            lp = log_probs[torch.arange(len(shift_labels), device=device), shift_labels].sum()
            log_pi_list.append(lp)

    return torch.stack(log_pi_list)  # (M,) each with grad


# ─────────────────────────────────────────────────────────────────────────────
# 8. Memory-Efficient Probe Projection
# ─────────────────────────────────────────────────────────────────────────────

def probe_project_grads(
    param_grads: tuple,   # tuple of per-parameter gradient tensors
    R: int,
    seed: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Project parameter gradients onto R probe vectors.

    Memory-efficient: generates N(0,I) probe entries block-by-block per parameter.
    Never materializes the full (R, d) probe matrix.

    Returns (R,) tensor of probe projections.
    """
    projections = torch.zeros(R, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for g in param_grads:
        if g is None:
            continue
        g_flat = g.detach().flatten()  # (n_p,)
        n_p = g_flat.shape[0]
        # Generate probe rows for this parameter block: (R, n_p)
        # But do it in sub-blocks if n_p is large to limit peak memory.
        BLOCK = 1 << 20  # 1M elements per sub-block
        offset = 0
        while offset < n_p:
            end = min(offset + BLOCK, n_p)
            block_size = end - offset
            # (R, block_size) probe block
            v_block = torch.randn(R, block_size, generator=rng, device=device)
            projections += v_block @ g_flat[offset:end]  # (R,)
            offset = end

    return projections


# ─────────────────────────────────────────────────────────────────────────────
# 9. VLM Weighted Probe Projections (replaces compute_multi_weight_projs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vlm_weight_projs(
    model,
    processor,
    rollouts: Rollouts,
    weight_vecs: list,    # list of n_w tensors, each (M,)
    R: int,
    probe_seed: int,
    device: str = "cuda",
) -> list:
    """Compute R probe projections via n_w weighted backward passes for a VLM.

    Replaces compute_multi_weight_projs from cube_sim.py for variable-length VLM sequences.

    Strategy:
      1. Compute log π(y_m | x_m) for each m → (M,) list of scalars with grad
      2. For each weight vector w_i: loss = Σ_m w_i[m] * log_pi[m]
         backward → per-param grads → probe_project_grads → (R,)

    Returns list of n_w tensors, each (R,).
    """
    M = rollouts.M
    N = rollouts.N_per_prompt
    B = rollouts.B
    n_w = len(weight_vecs)

    # Get LoRA parameters (trainable only)
    lora_params = [p for p in model.parameters() if p.requires_grad]

    # Compute log_pi for all M rollouts (each scalar with grad in computation graph)
    log_pi_list = compute_log_probs_batch(
        model, processor,
        rollouts.items,
        rollouts.responses,
        N=rollouts.N_per_prompt,
        device=device,
        N_list=rollouts.N_list,
    )  # (M,) tensor with grad

    results = []
    for i, w in enumerate(weight_vecs):
        # loss = Σ_m w[m] * log_pi[m]
        loss = (w.detach() * log_pi_list).sum()
        retain = (i < n_w - 1)
        grads = torch.autograd.grad(
            loss, lora_params,
            retain_graph=retain,
            create_graph=False,
            allow_unused=True,
        )
        proj = probe_project_grads(grads, R, seed=probe_seed, device=device)
        results.append(proj)

    return results
