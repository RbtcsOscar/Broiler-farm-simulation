# 2D 육계 군집–로봇 상호작용 시뮬레이터 (v0.2.3)

가이드 `2D_육계_군집_로봇_상호작용_시뮬레이터_구현_가이드_v0_1.md` 를 그대로 구현한 것.
18 m × 4.5 m 축사에서 1,500마리 육계의 평상시 군집행동과 로봇 접근에 대한
직접 회피 / 연쇄반응 / WALK·BURST 를 재현한다.

물리는 전부 SI 단위(m, s)이고, 화면 표시는 `config.PX_PER_M` 로만 스케일한다.
닭은 원(몸통) + 앞쪽 **빨간 삼각형(head 방향)** 으로 그린다.

## 설치 & 실행

```bash
pip install -r requirements.txt      # numpy, pygame

python main.py                       # GUI 실행 (기본 N=1500)
python main.py -n 800                # 닭 수 변경
python main.py --scale 120           # 화면 확대 (px/m)
python main.py --habituation         # habituation ON 으로 시작
python main.py --no-robot            # BASE 군집만 (Phase 3 확인용)
python main.py --headless -t 120     # 화면 없이 120초 → logs/ 에 저장
```

성능: N=1500 에서 약 25~30 physics step/s (20 Hz 이상, 실시간 재생 가능).
느리면 `config.SIM_STEPS_PER_FRAME` 를 낮추거나 `-n` 으로 개체 수를 줄인다.

## 조작키 (GUI)

| 키 | 동작 |
|---|---|
| `space` | 일시정지 |
| `h` | habituation ON/OFF 토글 |
| `r` | 재초기화(새 배치) |
| `+` / `-` | 배속 (프레임당 physics step 수) |
| `g` | head 삼각형 표시 토글 |
| `esc` | 종료 |

색: **BASE = beige**, **ESCAPE-WALK = orange**, **ESCAPE-BURST = red**, head ▶ 항상 빨강.

## 코드 구조 (가이드 §27)

```
chicken_sim/
├── main.py          # 진입점 / CLI
├── config.py        # 모든 파라미터 (SI 단위, 표시 스케일은 PX_PER_M 만)
├── simulation.py    # World/Chicken/Robot/Renderer/Logger 조립 + 메인 루프
├── world.py         # geometry: 벽·급이통(collision) / 급이·급수 라인(visual only)
├── spatial_grid.py  # uniform grid, 3x3 이웃 검색 (§24)
├── chickens.py      # ChickenManager (NumPy 배열, §26 업데이트 순서 전체)
├── stimulus.py      # D, C, S, P_escape, U 순수 함수 (§10~§15)
├── robot.py         # scripted 등속 직선 궤적 (§30)
├── renderer.py      # pygame 렌더링, m→px 스케일, 빨간 head 삼각형
└── logger.py        # summary.csv + snapshots.npz (§32)
```

## 핵심 설계 원칙 반영 (§34)

- **행동모델 ↔ 물리 collision 분리**: 행동은 desired heading 만 결정, collision 은 위치 보정.
- **급이·급수 라인은 장애물이 아님**: `visual=True, collision=False`. 닭이 자유 통과.
- **벽·급이통은 hard obstacle**: 절대 관통 불가 (clamp + push-out).
- **닭은 head 방향으로만 이동**: `v = v·[cosθ, sinθ]`, 옆걸음 없음.
- **개체별 민감도 `s_i ~ Beta`**: 초기 1회 생성 후 고정.
- **직접 자극 `D` vs 사회적 자극 `C` 분리**, habituation 은 `D` 에만 적용.

## 한 step 업데이트 순서 (chickens.ChickenManager.step, §26)

grid 재구성 → neighbor(nn, C_i, e_C) → D → habituation/h → S →
BASE→ESCAPE 전환 → WALK/BURST 결정 → desired heading →
heading update → speed update → 위치 → 벽/급이통/닭-닭 충돌 →
timer/ESCAPE 종료 → prev_stimulus 저장.

## 주요 튜닝 파라미터 (config.py)

| 파라미터 | 의미 | 효과 |
|---|---|---|
| `BETA` | 사회적 전파 강도 β | 클수록 연쇄반응(파도)이 넓고 오래 감 |
| `LAMBDA_ESCAPE` | 전환율 λ_E | 클수록 쉽게 ESCAPE 로 전환 |
| `R_STIM` | 로봇 자극 반경 | 반응 시작 거리 |
| `D_CALM` | 진정 임계 | 로봇이 이 거리 밖으로 가면 진정 → 군집 회복 |
| `THETA_B` | BURST 임계(U) | 낮출수록 burst 자주 발생 |
| `V_WALK`, `V_BURST` | 회피/버스트 속도 | v_B > v_W |
| `S_BETA_ALPHA/BETA` | 민감도 분포 | 반응 개체 비율/이질성 |
| `PX_PER_M`, `CHICKEN_DRAW_SCALE` | 표시 스케일 | 화면 크기·닭 글리프 크기(물리 무관) |

> 연쇄반응 세기(`BETA`)·`LAMBDA_ESCAPE`·`D_CALM` 은 실제 관측 데이터에 맞춰 튜닝하는
> 대상이다(§34 원칙 8). 기본값은 "로봇 주변에서 국소 반응 → 통과 후 군집 회복" 이
> 눈에 보이도록 잡아두었다.

## 로그 (§32)

`--headless` 또는 GUI 종료 시 `logs/` 에 저장:
- `summary_*.csv` : t, escape_ratio, mean_speed, robot_x/y, robot_min_dist, mean_nn_dist, mean_S
- `snapshots_*.npz` : 1초 간격 `x, y, theta, state, S` 전체 배열 (재생/분석용)

## v0.2 / v0.2.1 변경사항

행동상태(BASE/ESCAPE)와 movement mode(WALK/BURST) 골격은 그대로 두고,
"계속 밀림 / 저민감도 과반응 / 원거리 BURST / 벽 끼임 / 무자극 head 회전"
다섯 문제를 구조적으로 고쳤다.

- **거리기반 ESCAPE bout (v0.2 §3.1)**: 시간이 아니라 이동거리 예산
  `L ~ N(L_ESCAPE_MEAN, STD)` 만큼만 이동하고 멈춘다. 더 이상 로봇이
  멀어질 때까지 계속 밀려가지 않는다. (safety timer 는 runaway 방지용으로만 잔존)
- **rest + retrigger (§3.2)**: bout 후 `reaction_rest_timer` 동안 재반응 금지.
  로봇이 같은 자리에 있으면 계속 멈춰 있고, `S > S_stop + RETRIGGER_DELTA`
  (또는 `D ≥ D_EMERGENCY`)로 자극이 다시 커질 때만 다음 짧은 bout 발생.
- **active escape threshold (§3.3)**: `P = 1 − exp(−λ·max(0, S − THETA_ESCAPE)·dt)`.
  `S ≤ THETA_ESCAPE` 개체는 아무리 오래 노출돼도 active escape 안 함 → 저민감도 개체 보존.
- **근접 BURST (§3.4)**: `clearance < BURST_CLEARANCE` **그리고** `dD⁺/dt > BURST_DDOT_TH`
  일 때만. 사회적 자극만으로는 BURST 안 켜짐.
- **crowd pressure / passive move (§3.5, §6, §12)**: `R_PRESSURE` 내 이웃의
  **net 밀림**을 sensitivity 와 독립적으로 계산. 둔감한 닭도 뒤에서 밀리면
  `V_CROWD_MAX` 이하로 천천히 수동 이동. (C_i 와 절대 합치지 않음)
- **wall-aware flow (§11)**: 벽에서 heading 반사 대신, 벽 밖으로 향하는 normal
  성분만 제거하고 tangent 로 흐르게. 정면 압축(속도≈0)이면 tangent+안쪽 bias 로 탈출.
  벽 근처 속도는 `WALL_SPEED_FACTOR` 로 감쇠하되 crowd 있으면 완전 정지 안 함.
- **overlap solver 안정화 (§14)**: iteration 당 밀림 `MAX_OVERLAP_PUSH` 로 cap,
  `CONSTRAINT_ITERS` 회 overlap→feeder→wall 반복.
- **(v0.2.1 §20) 무자극 BASE heading freeze**: 정지한 BASE 닭은 `v=0, ω=0, θ 고정`.
  `turn_active = ESCAPE | BASE bout | crowd flow` 인 개체만 회전한다.
  → renderer 의 head 삼각형(θ 기반)이 무자극 상태에서 빙글빙글 도는 현상 제거.
- **(v0.2.1 §20.3) BASE 이동 빈도 = 초당 rate**: `P = 1 − exp(−LAMBDA_BASE_MOVE·dt)`
  로 timestep 독립. 무자극이면 대부분 정지, 일부만 간헐 이동.

### ablation / rollback 플래그 (config.py, 부록 B)

문제 원인 분리를 위해 각 기능을 독립 on/off:
`USE_DISTANCE_BOUT`, `USE_REACTION_REST`, `USE_ESCAPE_THRESHOLD`,
`USE_CLOSE_BURST_GATE`, `USE_CROWD_PRESSURE`, `USE_WALL_FLOW` (모두 기본 True).
예: `USE_WALL_FLOW=False, USE_CROWD_PRESSURE=True` 로 두면 벽 끼임이 pressure
때문인지 wall handling 때문인지 분리할 수 있다.

### ⚠ 캘리브레이션 주의 (실측 영상 맞추기 전 알아둘 것)

- **BURST 는 현재 기본값에서 사실상 발생하지 않는다.** 로봇이 등속
  `ROBOT_VEL=0.35 m/s`, `R_STIM=1.25 m` 이면 `dD/dt` 의 물리적 상한이
  `≈ v_robot / R_STIM ≈ 0.28`(측정 최대 ~0.33)인데, 스펙 기본값
  `BURST_DDOT_TH=0.50` 은 이보다 크다. BURST 를 보려면 `BURST_DDOT_TH` 를
  0.15~0.20 으로 낮추거나 로봇 속도를 높여라. (스펙 숫자를 임의로 바꾸지 않고
  그대로 두었으니, 영상 캘리브레이션 시 이 항목을 조정할 것 — 튜닝 우선순위 §18-8)
- **순간 escape 비율은 낮은 게 정상이다.** 거리기반 bout 로 한 번 반응이
  수 cm~수십 cm 짧은 hop 이라, 동시 ESCAPE 는 1% 안팎이지만 로봇이 지나가는
  동안 전체의 ~15% 개체가 한 번씩 반응한다(로그 `bout_end_count` 누적으로 확인).
- **성능**: `CONSTRAINT_ITERS=2` 기준 N=1500 에서 약 16 physics step/s
  (실시간의 0.8배). 실시간이 필요하면 `CONSTRAINT_ITERS=1` 로 낮추거나
  `-n` 으로 개체 수를 줄여라.
- v0.2 재튜닝값(스펙 §4.1): `R_STIM 1.60→1.25`, `BETA 0.60→0.40`,
  `S_BETA_ALPHA 2.0→1.3`. 전부 검증 전 초기값이므로 영상에 맞춰 §18 우선순위대로 조정.

### 추가된 로그 컬럼 (§17.1)

`summary_*.csv` 에 `burst_count, crowd_active_ratio, near_wall_ratio,
reaction_rest_ratio, bout_end_count` 추가.

## v0.2.2 변경사항 (social flow 복구 + 실측 반응거리 + 무반응 개체)

- **clearance 기반 직접 자극 (§21.7)**: `D = clip((ONSET−clearance)/(ONSET−FULL),0,1)`,
  `clearance = d_center−(r_robot+r_c)`. 외곽 30 cm 밖 0, 5 cm 안 1. 중심거리/R_STIM 대신
  로봇-닭 외곽거리라 실측(15~30 cm)과 직접 비교된다. 로봇 속도도 실측 시나리오 0.9 m/s.
- **social/crowd flow 복구 (§21.2~21.5)**: crowd activation 을 net push 가 아니라 scalar
  `crowd_pressure` 로 판단. crowd 방향 = `0.35·repulsion + 0.65·local_flow`(이웃 실제
  velocity 평균) 로 바꿔, 대칭 압축에서도 방향을 얻는다. BASE crowd-only 는 escape
  vector 와 분리(`crowd_dir_valid`).
- **무반응(low responder) 개체 (§21.9~21.10)**: 평균 0.5%(run 마다 0.4~0.6%, N=1500→약
  6~9마리)를 `direct_response_gain=0` 으로 둔다. **로봇 직접 자극에만 무반응**이고 BASE
  자발 이동·social·crowd 밀림은 정상. collision 판정 거리는 그대로 계산.
- **D_eff 일원화**: habituation·S·emergency·escape 직접항·burst 미분 모두 `D_eff = D_raw
  × gain` 사용. `first_response_clearance`(개체별 최초 ESCAPE 시 clearance) 기록.

### ⚠ 내가 스펙 초기값에서 벗어난 2곳 (측정 기반, 반드시 알아둘 것)

가이드 §21 의 숫자는 명시적으로 "검증 전 초기 튜닝값"이라, 이 밀도/속도에서 §21.13
Test 를 통과하도록 두 값을 측정 기반으로 조정했다. 실측 영상 캘리브레이션 시 이 두 개를
최우선으로 다시 맞추면 된다.

1. **`PRESSURE_THRESHOLD` 0.08 → 1.10.**
   이 밀도(육계 rest 간격 ~0.15 m, R_PRESSURE=0.22)에서 `crowd_pressure` 의 rest 분포는
   median≈0.47 / p95≈1.09 다. 스펙의 0.08 은 rest 의 91% 를 crowd_active 로 잡아 무자극
   군집이 통째로 퍼진다(Test A 실패). passive flow 는 '정상 이상으로 눌린' 경우만 발동해야
   하므로 rest 분포 위(≈p95)로 올렸다. + 추가로 passive 이동은 이웃이 실제로 흐를 때
   (`local_flow` present)만 트리거해 정적 self-repulsion 에 의한 팽창을 막았다.
2. **`LAMBDA_ESCAPE` 2.5 → 50.**
   반응확률은 hazard 구조라 노출시간에 좌우된다(§21.8). 로봇 0.9 m/s 가 clearance 5~30 cm
   밴드를 ~0.28 s 에 지나므로, 스펙의 2.5 로는 거의 접촉 직전(median 2.8 cm)에야 반응한다.
   λ=50 에서 첫 반응 clearance median≈17 cm(IQR 11~22)로 목표 15~30 cm 에 든다.
   (로봇 속도를 바꾸면 이 값도 같이 바꿔야 한다 — 둘은 커플링된 캘리브레이션.)

### v0.2.2 신규 ablation 플래그

`USE_CLEARANCE_DIRECT_STIMULUS`(off 면 legacy R_STIM 자극),
`USE_LOCAL_VELOCITY_FLOW`(crowd 방향에 이웃 velocity 결합),
`USE_LOW_RESPONDER`(무반응 개체). 모두 기본 True.

### §21.13 테스트 매트릭스 통과 현황 (N=1500)

- **A** robot OFF → 무자극 군집 정지: meanV≈0.003, crowd_move≈1% ✓
- **C** robot 0.9 m/s → 첫 반응 clearance median≈17 cm ✓
- **D** social ON → 로봇 주변 국소 반응 밴드 + crowd flow, 통과 후 회복(peak≈1%, decay) ✓
- **E** low responder → run 별 6~9마리(0.4~0.6%) ✓
- **BURST** → 근접 개체에서 규칙적으로 발화(0.9 m/s 에서 dD/dt 상한 ~3.6 ≫ 0.5) ✓
- **F** wall crowd → 벽 접선 flow(관통 없음) ✓

로그에 `crowd_move_ratio, D_raw_max, D_eff_max, v_crowd_mean, low_responder_count` 추가.

## v0.2.3 변경사항 (social contagion / passive flow 복구)

v0.2.2 에서 social wave 가 여러 gate 에 연속으로 막혀 거의 안 보이던 문제를 구조적으로
고쳤다. **social contagion(상태변화 연쇄) 과 passive crowd flow(수동 이동) 를 분리**한다.

- **최초 반응 ≠ 재반응 (§3)**: `needs_retrigger` 플래그. 한 번도 안 움직인 개체는 retrigger
  문턱 없이 첫 반응 가능하고, bout 를 끝낸 개체만 `stim_up`(자극 증가) 을 요구한다. v0.2.2 는
  최초 반응에도 retrigger 문턱이 걸려 social wave 가 시작 전에 막히던 구조였다.
- **direct/social drive 분리 (§4)**: `escape_drive = max(0, s·h·D − THETA) + BETA·C_i`.
  THETA_ESCAPE 를 **직접자극에만** 적용하고 social(`BETA·C_i`) 은 sensitivity/threshold 로
  다시 누르지 않는다 → 주변에 ESCAPE 가 생기면 연쇄반응이 threshold 에 막히지 않고 시작.
  `P = 1 − exp(−λ·drive·dt)`.
- **local flow magnitude 보존 (§6)**: `local_flow_speed`(이웃 평균 속도 크기)를 방향과 별도 저장.
- **passive crowd gate 완화 (§7)**: `crowd_move = crowd_active & dir_valid &
  (flow_present | push_present)` — 이웃이 실제로 흐르거나 비대칭 압박 중 하나만 만족해도 이동.
- **passive 최소 가시 속도 (§8)**: `v_crowd = min(V_CROWD_MAX, V_CROWD_MIN + gain·excess)`.

### 결과 — λ 를 spec nominal 2.5 로 되돌림 (중요)

escape_drive 로 social 이 direct 에 **더해지면서**, v0.2.2 에서 반응거리 때문에 억지로
올렸던 `LAMBDA_ESCAPE=50` 이 더는 필요없어졌다. **λ=2.5(spec nominal), BETA=0.60 그대로** 로
아래가 동시에 성립한다:
- 로봇 근처 첫 반응 clearance median≈16 cm (직접+사회) — 목표 15~30 cm.
- social wave 국소 전파 후 감쇠(Test B: 강제 ESCAPE 1 → peak 3 → 소멸). 로봇 통과 시
  peak esc≈1.5% → 통과 후 회복. **λ≥5 면 flock 전체로 번지는 과전파**라 2.5 가 안정 지점.
- passive crowd flow 로봇 주변 `v_crowd`≈0.07~0.10 m/s (목표 0.04~0.12).

### §12 테스트 매트릭스 (N=1500)

A(robot OFF→정지 meanV≈0.002) / B(강제 ESCAPE 1→wave 후 감쇠) / C(BASE 이웃 0.07~0.10 m/s
수동 flow) / D(대칭 무자극→정지) / F·G(bout 후 동일자극 정지 / 자극증가 시 새 bout) /
벽 tangent flow / burst 근접 발화 / 무반응 7마리 — 전부 통과.

로그 추가: `social_drive_max, local_flow_speed_max, push_present_ratio, flow_present_ratio,
v_crowd_max`.

### ⚠ 남은 측정 기반 조정 1곳

`PRESSURE_THRESHOLD`(1.10) 와 `CROWD_PUSH_TRIGGER`(0.60) 는 spec nominal(0.08 / 0.05) 대신
**측정한 rest 분포 위**로 잡았다. 이 밀도(간격 ~0.15 m, R_PRESSURE 0.22)에서 rest 의
`crowd_pressure` median≈0.47, `crowd_push` median≈0.30 이라, spec 값이면 rest 의 ~90% 가
crowd_active/push_present 로 잡혀 무자극 군집이 통째로 퍼진다(Test A/D 실패). passive flow 는
'정상 이상으로 눌린' 경우만 발동해야 하므로 rest 분포 위로 올렸다. (`CROWD_FLOW_SPEED_MIN`
=0.015 는 spec 유지 — flow_present 는 이웃이 실제 움직일 때만 참이라 rest 에서 안전.)
실측 캘리브레이션 시 R_PRESSURE 를 실제 personal space(≈0.13?)로 줄이면 spec 의 작은
threshold 도 그대로 쓸 수 있다.

> 튜닝 노브: social wave 확대 `BETA` 0.60→0.70(과하면 전체 전파), passive 세기
> `V_CROWD_MIN`/`MAX`, 반응거리 로봇속도+clearance+λ. (BETA·V_CROWD_MAX 동시 상향은
> '계속 밀림' 재발하니 따로.)

## 알려진 한계 / 다음 작업

- v0.2.2 는 로봇 0.9 m/s + clearance 자극이라 BURST 가 정상 발화한다. 단, 로봇을 0.35 m/s
  등 느리게 바꾸면 `dD/dt` 상한이 낮아져 BURST 가 잘 안 뜰 수 있다(그땐 `BURST_DDOT_TH`
  하향 필요). 속도-반응거리-λ 는 함께 캘리브레이션할 것.
- feeder 접근 행동(스펙 §20.5)은 선택 기능이라 아직 미구현. 필요하면 급이통 둘레
  `FEED_TARGET_RADIUS=0.23` 링 위 점을 target 으로 하는 BASE 이동으로 붙이면 된다
  (급이통 중심은 hard obstacle 이라 금지).
- 로봇은 scripted 등속 직선만. planner 결합은 닭 모델 검증 이후.
- 시야각 factor `V_i` 는 초기 OFF (`D_i = D_i^0`).

## 이 시뮬레이터를 로봇 planner 예측 모델로 쓰려면

`snapshots_*.npz` 의 상태열을 학습/검증 데이터로 쓰거나,
`ChickenManager.step()` 을 planner 루프 안에서 로봇 위치를 바꿔가며 rollout 하면
"이 경로로 가면 닭이 어떻게 반응하는가" 를 예측할 수 있다.
