"""
config.py
=========
v0.4.4 threat-loss termination + dynamic escape direction + contact-observed social source.

목적
- social interaction은 두 요소만 유지한다.
  1) calm-BASE spacing attraction / short-range repulsion
  2) previous-frame movement-following excitation
- C_i ESCAPE fraction, 3 m social path, density path discount,
  provenance/upstream graph, recruited-cohort memory는 사용하지 않는다.
- behavioral locomotion과 mechanical displacement 분리는 유지한다.
- ESCAPE direction은 매 frame robot/social/separation에서 다시 계산한다.
- final voluntary vector는 direct threat 중 robot-inward 성분만 제거한다.
- actual contact push는 social perception source로 허용하되 passive crowd relaxation은 제외한다.
- threat-loss termination으로 안전해진 ESCAPE가 조기 종료될 수 있다.
- wall relief는 이번 ablation에서 OFF한다.

주의
- R_SOCIAL=0.75 m는 Febrer et al. (2006)의 "nearest bird가 약 75 cm보다
  멀 때 후보 위치를 높은 확률로 거부한 attraction model이 자료에 가장 잘 맞았다"는
  결과를 동적 spacing rule의 참고 거리로 옮긴 개발값이다.
  원 논문의 force constant 또는 선호거리와 동일한 값이 아니다.
- 행동 관련 숫자는 개발/튜닝값이며 생물학적 상수로 간주하지 않는다.
"""


import numpy as np

# ----------------------------------------------------------------------------
# World
# ----------------------------------------------------------------------------
WORLD_LENGTH = 18.0
WORLD_WIDTH  = 4.5

FEEDER_LINE_Y  = 1.5
DRINKER_LINE_Y = 3.0
LINE_START_X   = 1.2
LINE_LENGTH    = 15.6

FEEDER_COUNT  = 10
FEEDER_RADIUS = 0.15

# ----------------------------------------------------------------------------
# Chicken geometry / time
# ----------------------------------------------------------------------------
CHICKEN_COUNT  = 1500
CHICKEN_RADIUS = 0.055
DT = 0.05

# ----------------------------------------------------------------------------
# Neighborhoods
# ----------------------------------------------------------------------------
R_MIN    = 0.14   # 자발적으로 걷는 BASE 개체의 근접 separation 조향
R_SOCIAL = 0.75   # local flock cohesion radius; development scale inspired by Febrer 2006, not a force constant
R_C      = 0.40   # compatibility alias; simple-social excitation uses R_FLOW
R_FLOW   = 0.45   # 실제 movement-flow 방향을 감지하는 이웃 반경

# Local density는 social radius와 같은 공간척도에서 진단한다.
# 별도 독립 숫자를 추가하지 않기 위해 R_C를 재사용한다.
R_DENSITY = R_C

# Simple robot-centered social-response envelope. No graph/path/provenance is used.
# 3.0 m is a development/tuning value based on observed finite reaction extent.
SOCIAL_PROP_RANGE = 3.0

# 밀집 crowd relaxation. behavioral heading/speed mode가 아니라 passive mechanical displacement.
# 아래 값들은 개발/튜닝값이며 생물학적 상수가 아니다.
R_PRESSURE = 0.17
PRESSURE_THRESHOLD = 0.20
V_CROWD_MAX = 0.07

# 과밀 jam 시작점. 이 이상에서는 voluntary locomotion이 점차 감소한다.
# social excitation 자체를 약화시키는 값이 아니라 "몸이 빠져나갈 수 있는 mobility"만 낮춘다.
JAM_PRESSURE = 1.50

# 기본 wall rule은 tangent projection이다.
# 아래 relief는 실제 주행 영상에서 확인한 wall-side redistribution을 위한 확장모델.
# ablation 비교가 가능하도록 연구 옵션으로 분리한다.
ENABLE_WALL_RELIEF = False
WALL_RELIEF_DEPTH = 1.20

# ----------------------------------------------------------------------------
# Locomotion
# ----------------------------------------------------------------------------
V_BASE  = 0.06
V_WALK  = 0.35
V_BURST = 0.90

TAU_BASE  = 0.50
TAU_WALK  = 0.40
TAU_BURST = 0.12
TAU_STOP   = 0.12

K_THETA   = 6.0
OMEGA_MAX = 8.0

# ----------------------------------------------------------------------------
# Individual responsiveness
# robot_sensitivity / social_sensitivity 를 독립적으로 1회 샘플링.
# ----------------------------------------------------------------------------
ROBOT_SENS_ALPHA = 2.0
ROBOT_SENS_BETA  = 2.5
SOCIAL_SENS_ALPHA = 5.0
SOCIAL_SENS_BETA  = 2.0

# ----------------------------------------------------------------------------
# Robot-threat / social-flow response
# ----------------------------------------------------------------------------
# surface clearance가 이 값 이상이면 direct distance cue = 0.
# 0.8~1.0 m/s 주행에서 중앙 반응거리가 대략 15~30 cm권에 오도록 시작한 개발값.
ROBOT_CUE_CLEARANCE = 0.45

# ----------------------------------------------------------------------------
# Renderer compatibility
# ----------------------------------------------------------------------------
# 기존 renderer.py가 이 이름으로 stimulus ring 반경을 참조한다.
# v0.3에서는 의미가 동일하게 'direct robot cue가 0이 되는 surface clearance'이다.
DIRECT_ONSET_CLEARANCE = ROBOT_CUE_CLEARANCE

# closing-speed cue는 선형 정규화한다.
# 0.2 m/s -> 0.25, 0.4 m/s -> 0.5, 0.8 m/s 이상 -> 1.0.
# 이번 branch는 ROBOT_VEL=0.20 m/s 실행조건을 기준으로 한다.
V_THREAT_REF = 0.80

# BASE -> movement bout hazard를 direct/social로 분리한다.
# robot: 거리+closing speed+개체 민감도
# social: previous-frame actual disturbance movement -> E_i excitation
# social range/path/provenance의 별도 cascade rule은 두지 않는다.
LAMBDA_ROBOT  = 20.0
LAMBDA_SOCIAL = 13.0
BETA = 0.60

# old code/logger compatibility only
LAMBDA_RESPONSE = LAMBDA_ROBOT

# Legacy compatibility only. Direction is taken directly from previous-frame movement;
# no extra directional social memory is used in v0.4.1.
TAU_FLOW = 0.12

# social cue는 이웃 source가 보인 즉시 완전히 켜지지 않고 이 시간척도로 축적/감쇠한다.
# 실제 flock-follow는 이 excitation이 남아 있을 때만 허용해 flow-only conveyor를 막는다.
# one-hop-per-frame 전파와 함께 섣부른 flock-wide activation을 막는 개발/튜닝값.
TAU_SOCIAL_EXCITATION = 0.25

# weak attraction은 social excitation이 거의 사라진 calm BASE에서만 허용한다.
# 개발/튜닝 threshold이며 생물학적 상수가 아니다.
CALM_SOCIAL_THRESHOLD = 0.05

# ----------------------------------------------------------------------------
# Habituation (robot cue only)
# ----------------------------------------------------------------------------
HABITUATION_ON = False
K_H   = 0.15
H_MIN = 0.30
# distance cue가 이 값을 위로 crossing하면 로봇 노출 1회로 집계
D_EXPOSURE_TH = 0.50

# ----------------------------------------------------------------------------
# Ordinary BASE movement
# ----------------------------------------------------------------------------
LAMBDA_BASE_MOVE = 0.04
BOUT_MIN = 0.5
BOUT_MAX = 2.0

# ----------------------------------------------------------------------------
# Stimulus-driven movement bout: move -> stop -> refractory -> re-evaluate cue
# ----------------------------------------------------------------------------
L_ESCAPE_MEAN = 0.15
L_ESCAPE_STD  = 0.07
L_ESCAPE_MIN  = 0.04
L_ESCAPE_MAX  = 0.40

T_REST_MIN = 0.35
T_REST_MAX = 1.20

# 거리기반 bout가 비정상적으로 오래 지속되는 경우만 끊는 safety timeout
T_ESCAPE_MAX = 3.0

# ESCAPE 중 perceived robot/social drive가 이 값 아래로 떨어지면,
# 최소 이동거리 L_ESCAPE_MIN을 이미 이동한 경우 조기 종료한다.
# 개발/튜닝값이며 생물학적 상수가 아니다.
H_STOP = 0.03

# ----------------------------------------------------------------------------
# Close / sudden BURST
# ----------------------------------------------------------------------------
T_BURST = 0.20
BURST_CLEARANCE = 0.15
# dQ_robot+/dt가 이 이상이면 갑작스러운 threat 증가로 간주
BURST_QDOT_TH = 0.80

# ----------------------------------------------------------------------------
# Physical contact / walls
# ----------------------------------------------------------------------------
# 실제 overlap correction. social sensitivity는 적용하지 않는다.
# dense contact chain에서는 adaptive iteration으로 물리 displacement가 군집 내부를 통과한다.
# 아래 두 값은 행동 파라미터가 아니라 수치 solver 개발값이다.
CONTACT_MAX_ITERS = 24
CONTACT_TOLERANCE = 0.0010   # 1 mm residual penetration tolerance
CONTACT_ITERS = CONTACT_MAX_ITERS   # old-code compatibility
MAX_OVERLAP_PUSH = 0.02

# active desired direction이 벽 밖을 향하면 normal 성분을 제거하고 tangent를 유지.
R_WALL_FLOW = 0.25

# ----------------------------------------------------------------------------
# Rare behavioral near-nonresponders
# 약 0.5%는 robot/social behavioral sensitivity를 모두 0으로 둔다.
# 실제 chicken/robot contact 및 crowd pressure에 의한 물리 이동은 정상.
# ----------------------------------------------------------------------------
LOW_RESPONDER_RATE        = 0.005
LOW_RESPONDER_RATE_JITTER = 0.001

# ----------------------------------------------------------------------------
# Spatial grid
# ----------------------------------------------------------------------------
CELL_SIZE = max(R_SOCIAL, R_C, R_FLOW, R_PRESSURE, R_MIN, 2 * CHICKEN_RADIUS) + 0.05

# ----------------------------------------------------------------------------
# Settling / Robot
# ----------------------------------------------------------------------------
SETTLE_TIME = 8.0

ROBOT_ENABLED = True
ROBOT_RADIUS  = 0.25
ROBOT_START   = np.array([-1.0, 2.25])
ROBOT_VEL     = np.array([0.20, 0.0])

# renderer의 기존 stimulus-ring 호환용.
# 이제 중심거리 기준 ring은 direct cue가 0이 되는 surface-clearance 경계를 표시한다.
R_STIM = ROBOT_RADIUS + CHICKEN_RADIUS + ROBOT_CUE_CLEARANCE
ROBOT_DRAW_STIM_RING = True

# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
PX_PER_M           = 80
CHICKEN_DRAW_SCALE = 1.0
MARGIN_PX          = 24
HUD_HEIGHT         = 64
TARGET_FPS         = 60
SIM_STEPS_PER_FRAME = 1

COL_BG         = (24, 26, 30)
COL_WALL       = (210, 210, 215)
COL_FEEDER_LN  = (196, 140, 70)
COL_DRINK_LN   = (80, 150, 210)
COL_PAN        = (120, 120, 128)
COL_PAN_EDGE   = (170, 170, 178)
COL_BASE       = (225, 214, 180)
COL_WALK       = (240, 160, 60)
COL_BURST      = (235, 60, 55)
COL_HEAD       = (230, 40, 40)
COL_ROBOT      = (70, 200, 230)
COL_ROBOT_RING = (70, 200, 230)
COL_HUD_TEXT   = (230, 230, 235)
COL_HUD_DIM    = (150, 150, 158)

# ----------------------------------------------------------------------------
# Logging / reproducibility
# ----------------------------------------------------------------------------
LOG_ENABLED = True
LOG_EVERY   = 1
LOG_DIR     = "logs"

RANDOM_SEED = 42
