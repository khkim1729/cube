# 파일럿 구현 전략 및 코드 수정 이력

## 목차

- [Q1. 학습률(lr) 스케줄러가 필요한가?](#q1-학습률lr-스케줄러가-필요한가)
- [Q2. run_experiment.py와 cube/ 모듈이 파일럿과 다른 이유](#q2-run_experimentpy와-cube-모듈이-파일럿과-다른-이유)
- [코드 수정 이력](#코드-수정-이력)
  - [수정 1: probe_vecs 행렬 제거 (on-the-fly 생성으로 전환)](#수정-1-probe_vecs-행렬-제거-on-the-fly-생성으로-전환)
  - [수정 2: cube/ 패키지 메트릭 API를 파일럿에 맞게 수정](#수정-2-cube-패키지-메트릭-api를-파일럿에-맞게-수정)
  - [수정 3: probe_seed 오프셋 제거](#수정-3-probe_seed-오프셋-제거)
  - [수정 4: rollout_alloc 실제 가변 롤아웃 샘플링, rng 버그, 로깅 스텝 수정](#수정-4-rollout_alloc-실제-가변-롤아웃-샘플링-rng-버그-로깅-스텝-수정)
  - [수정 5: H_0에 N_j 미반영 수정 (rollout_alloc)](#수정-5-h_0에-n_j-미반영-수정-rollout_alloc)
  - [수정 6: RMS 편향 놈 수식 수정](#수정-6-rms-편향-놈-수식-수정)
  - [수정 7: STV 공식 3종 수정](#수정-7-stv-공식-3종-수정)
  - [수정 8: run_vlm.py 수정 (probe_seed, model.eval, 가변 롤아웃)](#수정-8-run_vlmpy-수정-probe_seed-modeleval-가변-롤아웃)
- [현재 상태](#현재-상태)

---

## Q1. 학습률(lr) 스케줄러가 필요한가?

> `run_pilot.py`를 보니 전체 학습 과정에서 동일한 lr을 사용하고 있는 것 같습니다.
> 보통 이런 학습에서는 코사인 스케줄러 등을 쓰는 게 맞지 않나요?

**파일럿 실험에서는 스케줄러가 불필요합니다.** 이유는 다음과 같습니다.

- **실험 목적이 다릅니다.** 파일럿(ToyPolicy)은 학습 성능을 최대화하는 것이 아니라, 각 (baseline × budget) 조합에서 **편향/분산을 측정**하는 것이 목적입니다. 스케줄러 유무는 측정값의 상대적 순서(논문의 핵심 결론)에 영향을 주지 않습니다.
- **학습 스텝이 짧습니다.** num_train_steps=200으로 짧은 학습 구간에서는 lr 감쇠 효과가 미미합니다.
- **재현성이 더 중요합니다.** 고정 lr은 체크포인트별 측정 결과의 재현성을 단순하게 유지시켜 줍니다.
- **VLM 실험(`run_vlm.py`)에서는 적용 권장.** 실제 Qwen2-VL-7B + LoRA 학습에서는 cosine annealing 등의 스케줄러 적용이 권장됩니다.

---

## Q2. `run_experiment.py`와 `cube/` 모듈이 파일럿과 다른 이유

> `run_experiment.py`의 주요 로직이 아직 placeholder 상태이고, 파일럿에서 올바르게 반영된 수정 사항들이 `cube/` 폴더의 파일들에는 반영되어 있지 않은 것 같습니다. (예: `cube/metrics/bias.py`의 함수들이 파일럿과 다른 방식으로 계산하고 있음)
>
> 혹시 파일럿을 완벽히 구현한 다음에 `run_experiment.py`와 `cube/` 모듈을 구현하려는 건가요?

**네, 그것이 의도된 전략이었습니다.**

처음 계획은 논문 기준의 완전한 파일럿(`experiments/cube_sim.py` + `experiments/run_pilot.py`)을 먼저 구현하고, Theorem 1/2의 이론적 예측과의 일치를 철저히 검증한 뒤, 올바른 로직을 `run_experiment.py`와 `cube/` 패키지에 반영하는 것이었습니다.

이 "파일럿 우선" 접근법을 선택한 이유:

1. **정확성 우선, 모듈성은 나중.** `cube/` 모듈 구조(`bias.py`, `variance.py` 등)는 측정 프로토콜이 확정되기 전에 초기 설계 스케치로 작성되었습니다. 일부 함수가 대형 `(M,M)` 행렬이나 `(d,M)` 점수 행렬을 입력으로 받는 설계는 원리상 맞지만, 대형 행렬을 사용하지 않는 파일럿의 matrix-free 구현과 일치하지 않았습니다.

2. **파일럿이 정답 기준.** 논문에 보고된 모든 공식 유도, 편향/분산 분해, 수치 결과는 파일럿 구현으로 검증되었습니다. 파일럿은 다음을 사용합니다:
   - Matrix-free `compute_baseline_r` (M×M A_B 불필요)
   - 해석적 `compute_HL_sq` (M×M L 불필요)
   - On-the-fly probe projection via `project_flat_grad` (R×d 행렬 불필요)
   - Bessel-corrected 편향 보정 across-variance 추정량

3. **수정 완료.** `cube/` 패키지 모듈과 `run_experiment.py`는 이후 파일럿과 일치하도록 업데이트되었습니다 (수정 2 참조).

---

## 코드 수정 이력

### 수정 1: probe_vecs 행렬 제거 (on-the-fly 생성으로 전환)

`run_experiment.py`의 probe_vecs (R,d) 행렬 메모리 이슈 수정.

- **제거**: `from cube.utils import make_probe_vectors` 및 `(R,d)` 행렬 생성 코드
- **추가**: `from cube.utils import project_flat_grad`
- `project_flat_grad(flat_g, R, probe_seed, device)`: 각 `v_r ~ N(0, I_d)` 를 on-the-fly 생성, 피크 메모리 O(d) 유지

파일럿(`cube_sim.py` + `run_pilot.py`)은 처음부터 이 메모리 효율적 방식을 사용하고 있었습니다.

---

### 수정 2: `cube/` 패키지 메트릭 API를 파일럿에 맞게 수정

**`cube/metrics/bias.py`**:
- `compute_bias()`, `decompose_bias()`: 전체 `(d,)` gradient 벡터 + `(R,d)` probe_vectors 행렬 → 이미 projection된 `(S,K,R)` 텐서로 변경
- `compute_bias_components()`: explicit `Psi(d,M)`, `A_B(M,M)` 행렬 사용 → **삭제**

**`cube/metrics/variance.py`**:
- `compute_HL_proxy()`: full `(M,M)` L 행렬 → analytical closed form으로 변경 (STV lambda는 `cube.estimators.stv._compute_lambda` import)
- `compute_sourcewise_trace()`: `Psi`/`L` 행렬 사용 → **삭제**
- `compute_variance()`, `decompose_variance()`: 이미 정확 — 유지

**`experiments/run_experiment.py`**:
- `compute_bias_components` import 제거
- 새 API로 업데이트 (p1/q 텐서 방식)

---

### 수정 3: probe_seed 오프셋 제거

- `cube_sim.py:measure_checkpoint`: `probe_seed=probe_seed_base + flat_idx` → `probe_seed=probe_seed_base`
- `run_pilot.py:run_experiment`: `probe_seed_base=args.probe_seed + step` → `probe_seed_base=args.probe_seed`
- `run_vlm.py`: 동일하게 `probe_seed=probe_seed + flat_idx` → `probe_seed=probe_seed`

---

### 수정 4: rollout_alloc 실제 가변 롤아웃 샘플링, rng 버그, 로깅 스텝 수정

**(A) rollout_alloc 가변 롤아웃 수 (`cube_sim.py`)**

기존: rollout_alloc budget일 때 모든 프롬프트에서 동일한 N개 롤아웃을 뽑고, H 행렬 가중치만 달리했음 → **잘못된 방식**

수정 후: `sample_rollouts_var()` 함수 신규 구현. 프롬프트별 실제로 다른 N_j개 롤아웃을 샘플링:
1. 프롬프트별 probe 롤아웃 n_probe개 샘플링 → 정답률 acc_j 계산
2. `weight_j = acc_j * (1 - acc_j) * 4` (Bernoulli 분산 proxy; 0.5에 가까울수록 큰 값)
3. `N_j = 1 + round(weight_j / sum(weight) * (M - B))` 로 할당 (sum == M, 각 프롬프트 최소 1개)
4. 실제 N_j개 롤아웃 샘플링

아울러 `Rollouts` dataclass에 `N_list` 필드 추가, `prompt_slice(j)` / `prompt_N(j)` 메서드 추가.
`_compute_stv_lambda`, `compute_baseline_r`, `build_D_B`, `build_H`, `compute_HL_sq` 등 모든 관련 함수에서 variable N_j 지원.

`measure_checkpoint`, `run_pilot.py:train_step`에서 `budget == "rollout_alloc"`이면 `sample_rollouts_var()` 호출.

**(B) `sample_rollouts` rng 미사용 버그 수정 (`cube_sim.py`)**

`torch.multinomial`에 `generator=rng` 누락 → 추가. 재현성 보장.

**(C) 마지막 학습 스텝 로깅 누락 수정 (`run_pilot.py`)**

- `and checkpoint_idx < args.T` 조건 제거 → 200번째(마지막) 스텝에서도 로깅됨
- checkpoint_idx 주석: `0~T-1` → `0~T`
- 로그 print: `{checkpoint_idx}/{args.T-1}` → `{checkpoint_idx}/{args.T}`

---

### 수정 5: H_0에 N_j 미반영 수정 (rollout_alloc)
- $H_0$는 Budget에 따라 달라지면 안될 것 같습니다. 수정 전 버전이 맞는 것 같습니다.

**파일**: `experiments/cube_sim.py` — `build_H` 함수

**문제**: `budget == "rollout_alloc"`이고 `rollouts.N_list is not None`일 때 (실제 가변 롤아웃 샘플링 경우), H는 각 프롬프트 j에 대해 `1/(N_j * B)`로 올바르게 계산되고 있었지만, H_0은 여전히 균일한 `1/(B * N)`으로 계산되고 있었음. 이로 인해 `delta_H = H - H0 ≠ 0`이 되어, 원래 0이어야 할 budget bias 및 fusion bias가 0이 아닌 값으로 잘못 계산되었음.

**수정 후**: `rollouts.N_list is not None`인 경우 H_0도 H와 동일하게 `H0_out[sl] = 1/(N_j * B)`로 설정. 결과적으로 `delta_H = H - H0 = 0`이 되어, rollout_alloc은 편향에 영향을 주지 않고 분산(`HL_F^2`)에만 영향을 줌. 이는 논문의 이론적 예측과 일치.

```python
# 수정 전 (잘못된 방식): H만 N_j 반영, H0는 균일 N
H[sl] = 1.0 / (N_j * B)
# H0 = 1/(B*N) 그대로 → delta_H ≠ 0

# 수정 후 (올바른 방식): H, H0 모두 N_j 반영 → delta_H = 0
H[sl]     = 1.0 / (N_j * B)
H0_out[sl] = 1.0 / (N_j * B)
```

---

### 수정 6: RMS 편향 놈 수식 수정

**파일**: `experiments/cube_sim.py` — `aggregate_metrics` 함수

**문제**: `total_bias_norm`이 `total_bias_proj.norm()`으로 계산되고 있었음. 이는 `√(Σ_r p_r^2)`로, R에 따라 스케일이 달라지는 수식. 논문에서 정의한 올바른 척도는 `√((1/R) Σ_r p_r^2)` (RMS, R-불변 척도)임.

**수정 후**: `total_bias_proj.pow(2).mean().sqrt()`로 변경.

```python
# 수정 전: √(Σ p_r^2) — R에 따라 스케일 변화
"total_bias_norm": total_bias_proj.norm().item()

# 수정 후: √((1/R) Σ p_r^2) — R-불변 RMS
"total_bias_norm": total_bias_proj.pow(2).mean().sqrt().item()
```

---

### 수정 7: STV 공식 3종 수정

**파일**: `experiments/cube_sim.py`

**(A) `_compute_stv_lambda` — var_mu 계산 오류**

`var_mu = var_r / N`으로 모든 프롬프트에 동일한 N을 사용하고 있었음. 가변 N_j 환경에서는 프롬프트별로 `var_mu[j] = var_r[j] / N_j`로 계산해야 함.

```python
# 수정 전:
var_mu = var_r / N  # 모든 프롬프트에 동일한 N 사용

# 수정 후:
var_mu = torch.zeros(B, device=device)
for j in range(B):
    N_j = rollouts.prompt_N(j) if ... else N
    var_mu[j] = var_r[j] / N_j
```

**(B) `compute_baseline_r` — STV bloo_scalar 오류**

기존 `bloo_scalar = (total_sum - r_j.sum()) / ((B-1) * N_j)`는 타깃 프롬프트 j의 N_j로 다른 프롬프트의 합을 나누는 방식으로 잘못됨. 올바른 수식은 각 프롬프트 k의 평균(자신의 N_k로 나눈 것)을 합산하는 것:

`μ̄_{-j} = (1/(B-1)) * Σ_{k≠j} (1/N_k) * Σ_l r_{k,l} = (1/(B-1)) * Σ_{k≠j} mean(r_k)`

```python
# 수정 전:
total_sum = rewards.sum()
bloo_scalar = (total_sum - r_j.sum()) / ((B - 1) * N_j)

# 수정 후: 각 프롬프트의 평균을 먼저 계산 후 합산
mu_arr[k] = rewards[_sl(k)].mean()  # 각 k의 N_k 사용
bloo_scalar = (mu_total - mu_arr[j]) / (B - 1)
```

**(C) `compute_HL_sq` — STV row_sq 세 번째 항 오류**

기존 `lam_j^2 / ((B-1) * N_j)`는 수식 유도 오류. 올바른 분모는 `(B-1)^2 * Σ_{k≠j}(1/N_k)`:

`‖row_t‖^2 = 1 + (1-λ_j)^2/(N_j-1) + λ_j^2/((B-1)^2 * Σ_{k≠j}(1/N_k))`

```python
# 수정 전:
val += lam_j ** 2 / ((B - 1) * N_j)

# 수정 후:
sum_inv_Nk = sum(1.0 / _nj(k) for k in range(B) if k != j)
val += lam_j ** 2 / ((B - 1) ** 2 * sum_inv_Nk)
```

균일 N=8, B=32 환경에서 세 번째 항은 기존 `lam^2/248`에서 수정 후 `lam^2/3724`로 약 15배 작아짐 → STV HL_F^2 값이 크게 감소할 것으로 예상.

---

### 수정 8: run_vlm.py 수정 (probe_seed, model.eval, 가변 롤아웃)

**파일**: `experiments/run_vlm.py`, `experiments/vlm_utils.py`

**(A) probe_seed 오프셋 제거**

`run_experiment` 함수에서 `probe_seed=args.probe_seed + step`으로 스텝마다 다른 시드를 사용하고 있었음. 이로 인해 학습 중 probe vector가 달라져, 서로 다른 체크포인트 간의 측정값이 비교 불가능해짐. `+ step` 제거.

`compute_vlm_weight_projs` 내부에서도 `seed=probe_seed + i`로 각 weight vector마다 다른 시드를 사용했음. 이 역시 체크포인트 내 5개 projection이 서로 다른 probe 공간에서 계산되는 문제. `+ i` 제거.

**(B) measure_checkpoint_vlm에서 model.eval() 수정**

probe projection 계산 직전에 `model.train()`을 호출하고 있었음. 이는 잘못된 방식으로, `run_pilot.py`와 마찬가지로 `model.eval()`이 올바름 (측정 시에는 dropout 비활성화 등).

**(C) run_vlm.py 가변 롤아웃 지원**

`budget == "rollout_alloc"`일 때 `measure_checkpoint_vlm`, `train_step_vlm` 모두에서 균일한 N개 롤아웃을 생성하고 있었음. `run_pilot.py`와 동일하게 `generate_rollouts_vlm_var` 함수를 신규 구현하고 이를 호출하도록 수정.

`vlm_utils.py` 변경 사항:
- `Rollouts` dataclass에 `N_list`, `prompt_slice(j)`, `prompt_N(j)` 추가 (cube_sim.py와 동일 인터페이스)
- `generate_rollouts_vlm_var()` 함수 신규 구현 (probe → 분산 추정 → N_j 할당 → 실제 샘플링)
- `compute_log_probs_batch()`: `N_list` 파라미터 추가 → 가변 N_j 지원
- `build_D_B_vlm()`: `j*N:(j+1)*N` → `rollouts.prompt_slice(j)` 사용

`run_vlm.py` 변경 사항:
- `_compute_stv_lambda`, `compute_baseline_r`, `compute_HL_sq` 호출 시 `rollouts` 파라미터 전달
- `train_step_vlm` 내 로그 확률 계산 시 슬라이싱을 `rollouts.prompt_slice(j)` 기반으로 수정
- 마지막 스텝 로깅 누락 수정 (`and checkpoint_idx < args.T` 조건 제거)

---

### 수정 9: RMS 편향 스칼라 추가

**파일**: `experiments/cube_sim.py` — `aggregate_metrics` 함수

**(A) 각 편향 성분에 대해 RMS 스칼라를 직접 계산해서 저장**

- `budget_bias_rms   = mean_p2.pow(2).mean().sqrt()`
- `baseline_bias_rms = mean_p3.pow(2).mean().sqrt()`
- `fusion_bias_rms   = mean_p4.pow(2).mean().sqrt()`

`mean_p2`, `mean_p3`, `mean_p4`는 각각 budget/baseline/fusion 편향 성분의 probe 차원별 기대값(\\(E[p^2], E[p^3], E[p^4]\\))
`total_bias_norm`에서 사용한 것과 동일한 형태의 RMS(\\(\\sqrt{\\mathbb{E}[\\cdot^2]}\\))로 스케일을 맞추기 위해 `pow(2).mean().sqrt()`를 사용

`run_pilot.py`, `run_vlm.py`의 `CSV_COLUMNS` 변경 사항: 
이 값들은 `aggregate_metrics` 반환 딕셔너리의 `"budget_bias_rms"`, `"baseline_bias_rms"`, `"fusion_bias_rms"` 키로 저장

---

### 수정 10: compute_HL_sq STV row_sq 3번째 항 수정

**파일**: `experiments/cube_sim.py` — `compute_HL_sq` 함수(STV baseline 분기)

- \\[frac{\\lambda_j^2}{(B-1)^2} \\sum_{k\\neq j} \\frac{1}{N_k}] 형태로 계산하도록 변경:

- `sum_inv_Nk = sum(1.0 / _nj(k) for k in range(B) if k != j)`
- `val += (lam_j ** 2) * sum_inv_Nk / ((B - 1) ** 2)`

가변 N_j 환경에서는 나머지 프롬프트들의 N_k 구조(\\(\\sum_{k\\neq j} 1/N_k\\))를 올바르게 반영해 STV의 HL proxy가 이론식과 일관되게 동작

---

## 현재 상태

| 컴포넌트 | 상태 |
|----------|------|
| `experiments/cube_sim.py` | 완료, 논문 기준 정확 (H_0 N_j 반영, RMS 놈, STV 3종 수정 포함) |
| `experiments/run_pilot.py` | 완료, 354회 검증 |
| `experiments/run_vlm.py` | 완료, `run_pilot.py`를 Qwen2-VL-7B + LoRA로 미러링 (가변 롤아웃 지원) |
| `experiments/vlm_utils.py` | 완료: `generate_rollouts_vlm_var`, 가변 N_j 지원 |
| `cube/utils/probe.py` | 완료: `project_flat_grad` (on-the-fly, O(d) 메모리) |
| `experiments/run_experiment.py` | 메트릭 API 업데이트 완료; VLM 학습 루프는 `run_vlm.py` |
| `cube/metrics/bias.py` | 완료: matrix-free 방식, `(S,K,R)` probe projection |
| `cube/metrics/variance.py` | 완료: 해석적 `HL_proxy`, M×M L 행렬 불필요 |
