"""
VLM (Vision-Language Model) wrapper for CUBE experiments.

Wraps HuggingFace-compatible VLMs to provide:
  1. Batch rollout generation
  2. Log-probability computation
  3. Flat gradient vector extraction (for probe projection)

Supported model families (tested):
  - Qwen2-VL (primary target: Qwen2-VL-7B-Instruct)
  - LLaVA-1.5 / LLaVA-NeXT
  - InternVL2
  - Phi-3-Vision
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class VLMWrapper(nn.Module):
    """Thin wrapper around a HuggingFace VLM (Qwen2-VL primary).

    Args:
        model     : HuggingFace AutoModelForCausalLM (multimodal), e.g. PeftModel
        processor : AutoProcessor for Qwen2-VL (handles text + images)
        device    : torch device string
    """

    def __init__(self, model, processor, device: str = "cuda"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompts: List[dict],    # list of {'question': str, 'image': PIL.Image or None}
        num_rollouts: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Generate num_rollouts responses per prompt.

        Returns:
            responses : (B, num_rollouts) list of generated strings
            log_probs : (B, num_rollouts) approximate sequence log-probs
        """
        all_responses, all_log_probs = [], []

        for prompt in prompts:
            responses, lps = [], []
            inputs = self._prepare_inputs(prompt)

            for _ in range(num_rollouts):
                output = self.model.generate(
                    **inputs,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
                gen_ids = output.sequences[0, inputs["input_ids"].shape[1]:]
                text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
                lp = self._sequence_log_prob(output.scores, gen_ids)
                responses.append(text)
                lps.append(lp)

            all_responses.append(responses)
            all_log_probs.append(lps)

        return all_responses, all_log_probs

    def compute_log_probs(
        self,
        prompts: List[dict],
        responses: List[List[str]],
    ) -> List[List[float]]:
        """Re-compute log-probs for existing responses (for importance weighting)."""
        all_lps = []
        for prompt, resp_list in zip(prompts, responses):
            inputs = self._prepare_inputs(prompt)
            lps = []
            for resp in resp_list:
                lp = self._compute_single_log_prob(inputs, resp)
                lps.append(lp)
            all_lps.append(lps)
        return all_lps

    def flat_grad(self, loss: torch.Tensor) -> torch.Tensor:
        """Return flattened parameter gradient for trainable parameters.

        Args:
            loss: scalar loss tensor (requires_grad=True)

        Returns:
            grad: (d,) flat gradient vector (LoRA params only if PEFT model)
        """
        grads = torch.autograd.grad(
            loss,
            [p for p in self.model.parameters() if p.requires_grad],
            create_graph=False,
            allow_unused=True,
        )
        parts = []
        for g in grads:
            if g is not None:
                parts.append(g.detach().flatten())
        return torch.cat(parts) if parts else torch.zeros(1, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_inputs(self, prompt: dict):
        """Prepare model inputs for a single prompt dict.

        Handles Qwen2-VL chat template with optional image.
        prompt: {"question": str, "image": PIL.Image or None, ...}
        """
        from experiments.vlm_utils import build_qwen_prompt

        messages = build_qwen_prompt(prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = prompt.get("image", None)
        if image is not None:
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text], return_tensors="pt"
            ).to(self.device)
        return inputs

    @staticmethod
    def _sequence_log_prob(scores, gen_ids) -> float:
        """Sum log-probs across generated tokens from generate() scores."""
        lp = 0.0
        for t, score in enumerate(scores):
            probs = torch.softmax(score[0], dim=-1)
            lp += torch.log(probs[gen_ids[t]] + 1e-12).item()
        return lp

    def _compute_single_log_prob(self, inputs, response: str) -> float:
        """Re-compute log-prob for a given response string."""
        resp_ids = self.processor.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        input_ids = torch.cat([inputs["input_ids"], resp_ids], dim=1)
        L_p = inputs["input_ids"].shape[1]

        pixel_values = inputs.get("pixel_values", None)
        with torch.no_grad():
            kwargs = {"input_ids": input_ids}
            if pixel_values is not None:
                kwargs["pixel_values"] = pixel_values
            out = self.model(**kwargs)
            logits = out.logits[0, L_p - 1:-1]  # (L_resp, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_lps = log_probs[range(resp_ids.shape[1]), resp_ids[0]]
        return token_lps.sum().item()
