# About the Pilot Implementation Strategy

## Question

> `run_experiment.py` still has most of its core logic as placeholders,
> and the correctness fixes that are properly reflected in the pilot
> are not yet reflected in the `cube/` module files
> (e.g., the functions in `cube/metrics/bias.py` compute things
> differently — and incorrectly — compared to the pilot).
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

3. **Pending update.** The `cube/` package modules (`bias.py`, `variance.py`,
   `run_experiment.py`) will be updated to match the pilot once the pilot
   experiments are complete and the paper results are locked.
   Specifically:
   - `cube/metrics/bias.py`: `compute_bias_components` uses explicit `Psi`
     and `A_B` matrices; will be replaced with the matrix-free approach.
   - `cube/metrics/variance.py`: `compute_HL_proxy` uses the full `(M,M)` `L`;
     will be replaced with the analytical closed form.
   - `experiments/run_experiment.py`: placeholder logic will be replaced with
     the same training loop and measurement protocol as `run_pilot.py`.

## Current Status (as of 2026-03)

| Component | Status |
|-----------|--------|
| `experiments/cube_sim.py` | Complete, paper-correct |
| `experiments/run_pilot.py` | Complete, 174+ runs validated |
| `cube/metrics/bias.py` | Early sketch — pending update |
| `cube/metrics/variance.py` | Variance decomposition correct; `HL_proxy` uses matrix form |
| `experiments/run_experiment.py` | Placeholder — pending update |
| `experiments/run_vlm.py` | New, mirrors `run_pilot.py` for VLMs |
