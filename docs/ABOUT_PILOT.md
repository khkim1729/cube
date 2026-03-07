# About the Pilot Implementation Strategy

## Question

> `run_experiment.py` still has most of its core logic as placeholders,
> and the correctness fixes that are properly reflected in the pilot
> are not yet reflected in the `cube/` module files.
> For example, the functions in `cube/metrics/bias.py` compute things
> differently — and incorrectly — compared to the pilot.
>
> Is the plan to fully complete and validate the pilot first,
> and then implement `run_experiment.py` and the `cube/` modules?

## Answer

Yes, that is the intended strategy.

The original plan was to implement a complete, paper-correct pilot first
(`experiments/cube_sim.py` + `experiments/run_pilot.py`), validate it
thoroughly against the theoretical predictions from Theorem 1 and Theorem 2,
and only then propagate the correct logic into `run_experiment.py` and the
`cube/` package modules.

This "pilot-first" approach was chosen deliberately:

1. **Correctness before modularity.** The `cube/` module structure
   (`bias.py`, `variance.py`, estimators, budgets) was written as an early
   architectural sketch before the measurement protocol was finalized.
   Some functions in `cube/metrics/bias.py` take a full `(M, M)` matrix
   `A_B` and a full `(d, M)` score matrix `Psi` as inputs — a design that
   is correct in principle but does not match the matrix-free implementation
   used in the pilot (which avoids storing those large matrices).

2. **The pilot is the ground truth.** All formula derivations, bias/variance
   decompositions, and numerical results reported in the paper are validated
   against the pilot implementation. The pilot uses:
   - Matrix-free `compute_baseline_r` (no `M×M` `A_B`)
   - Analytical `compute_HL_sq` (no `M×M` `L`)
   - On-the-fly probe projection via `project_flat_grad` (no `(R,d)` matrix)
   - Bessel-corrected bias-subtracted across-variance estimate

3. **Update complete.** The `cube/` package modules and `run_experiment.py`
   have now been updated to match the pilot approach:
   - `cube/metrics/bias.py`: `compute_bias` and `decompose_bias` now take
     pre-projected `(S, K, R)` tensors directly. `compute_bias_components`
     (which used explicit `Psi` and `A_B` matrices) has been removed.
   - `cube/metrics/variance.py`: `compute_HL_proxy` now uses the analytical
     closed form (no `(M,M)` `L` matrix). `compute_sourcewise_trace`
     (which used `Psi` and `L` matrices) has been removed.
   - `experiments/run_experiment.py`: updated to use the new metric APIs.
     Placeholder gradient samples remain `(S, K, R)` tensors; the real
     VLM training loop is implemented in `run_vlm.py`.

## Code Fix (probe vectors)

The probe vector issue in `run_experiment.py` has been addressed.
Previously `run_experiment.py` called `make_probe_vectors(d, R, seed)` which
materialized a full `(R, d)` matrix in memory. This is now updated:

- **Removed**: `from cube.utils import make_probe_vectors`
- **Added**: `from cube.utils import project_flat_grad`
- The placeholder sweep now documents that real gradient projections should use
  `project_flat_grad(flat_g, R, probe_seed, device)`, which generates each
  `v_r ~ N(0, I_d)` on-the-fly and keeps peak extra memory at `O(d)`.

The pilot (`cube_sim.py` + `run_pilot.py`) has used this memory-efficient
approach from the beginning.

## Current Status (as of 2026-03)

| Component | Status |
|-----------|--------|
| `experiments/cube_sim.py` | Complete, paper-correct |
| `experiments/run_pilot.py` | Complete, 264+ runs validated |
| `experiments/run_vlm.py` | Complete, mirrors `run_pilot.py` for Qwen2-VL-7B + LoRA |
| `cube/utils/probe.py` | Updated: `project_flat_grad` (on-the-fly, O(d) memory) |
| `experiments/run_experiment.py` | Updated metric APIs; VLM training loop in `run_vlm.py` |
| `cube/metrics/bias.py` | Updated: matrix-free approach, `(S,K,R)` probe projections |
| `cube/metrics/variance.py` | Updated: analytical `HL_proxy`, no `(M,M)` L matrix |
