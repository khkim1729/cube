"""
VLM (Vision-Language Model) wrapper for CUBE experiments.

Wraps HuggingFace-compatible VLMs to provide:
  1. Batch rollout generation
  2. Log-probability computation
  3. Flat gradient vector extraction (for probe projection)

Supported model families (tested):
  - LLaVA-1.5 / LLaVA-NeXT
  - Qwen-VL / Qwen2-VL
  - InternVL2
  - Phi-3-Vision
  - GPT-4V (API-only, reward extraction only)
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class VLMWrapper(nn.Module):
    """Thin wrapper around a HuggingFace VLM.

    Args:
        model     : HuggingFace AutoModelForCausalLM (multimodal)
        tokenizer : corresponding tokenizer/processor
        device    : torch device
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompts: List[dict],    # list of {'text': str, 'image': PIL.Image or None}
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
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                gen_ids = output.sequences[0, inputs["input_ids"].shape[1]:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
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
        """Return flattened parameter gradient as a 1D tensor.

        Args:
            loss: scalar loss tensor (requires_grad=True)

        Returns:
            grad: (d,) flat gradient vector
        """
        self.model.zero_grad()
        loss.backward()
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=self.device))
        return torch.cat(grads)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, prompt: dict):
        text = prompt.get("text", "")
        image = prompt.get("image", None)
        if image is not None:
            inputs = self.tokenizer(
                text=text, images=image, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return inputs

    @staticmethod
    def _sequence_log_prob(scores, gen_ids) -> float:
        """Sum log-probs across generated tokens."""
        lp = 0.0
        for t, score in enumerate(scores):
            probs = torch.softmax(score[0], dim=-1)
            lp += torch.log(probs[gen_ids[t]] + 1e-12).item()
        return lp

    def _compute_single_log_prob(self, inputs, response: str) -> float:
        resp_ids = self.tokenizer(response, return_tensors="pt").input_ids.to(self.device)
        input_ids = torch.cat([inputs["input_ids"], resp_ids], dim=1)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
            logits = out.logits[0, inputs["input_ids"].shape[1] - 1: -1]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_lps = log_probs[range(resp_ids.shape[1]), resp_ids[0]]
        return token_lps.sum().item()
