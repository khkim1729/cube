# 파일럿 구현 전략 및 코드 수정 이력

## 목차

- [Q1. 학습률(lr) 스케줄러가 필요한가?](#q1-학습률lr-스케줄러가-필요한가)
- [Q2. run_experiment.py와 cube/ 모듈이 파일럿과 다른 이유](#q2-run_experimentpy와-cube-모듈이-파일럿과-다른-이유)
- [코드 수정 이력](#코드-수정-이력)
  - [수정 1: probe_vecs 행렬 제거 (on-the-fly 생성으로 전환)](#수정-1-probe_vecs-행렬-제거-on-the-fly-생성으로-전환)
  - [수정 2: cube/ 패키지 메트릭 API를 파일럿에 맞게 수정](#수정-2-cube-패키지-메트릭-api를-파일럿에-맞게-수정)
  - [수정 3: probe_seed 오프셋 제거](#수정-3-probe_seed-오프셋-제거)
  - [수정 4: rollout_alloc 실제 가변 롤아웃 샘플링, rng 버그, 로깅 스텝 수정](#수정-4-rollout_alloc-실제-가변-롤아웃-샘플링-rng-버그-로깅-스텝-수정)
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

## 현재 상태

| 컴포넌트 | 상태 |
|----------|------|
| `experiments/cube_sim.py` | 완료, 논문 기준 정확 (가변 롤아웃, rng 수정 포함) |
| `experiments/run_pilot.py` | 완료, 264+ 회 검증 (마지막 스텝 로깅 수정 포함) |
| `experiments/run_vlm.py` | 완료, `run_pilot.py`를 Qwen2-VL-7B + LoRA로 미러링 |
| `cube/utils/probe.py` | 완료: `project_flat_grad` (on-the-fly, O(d) 메모리) |
| `experiments/run_experiment.py` | 메트릭 API 업데이트 완료; VLM 학습 루프는 `run_vlm.py` |
| `cube/metrics/bias.py` | 완료: matrix-free 방식, `(S,K,R)` probe projection |
| `cube/metrics/variance.py` | 완료: 해석적 `HL_proxy`, M×M L 행렬 불필요 |
