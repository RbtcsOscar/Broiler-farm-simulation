"""
config.py
=========
v0.3.4: finite-range social wave + wall crowd relaxation.

핵심 원칙
- 첫 반응/재반응 모두 같은 확률 hazard 사용
- direct cue = 거리(clearance) + 실제 closing speed
- social activation = v1식 R_C 내 ESCAPE 비율
- social direction = 이웃의 실제 effective velocity flow를 robot threat-frame으로 정렬
- social wave는 source에서 누적된 propagation path에 따라 감쇠
- robot이 지나간 rear zone에서는 신규 behavioral social propagation을 차단
- crowd pressure는 behavioral social wave와 분리하여 벽/밀집에서 수동 재배치
- physical contact = 실제 overlap geometry correction

아래 값 중 행동 관련 숫자는 초기 개발/튜닝값이며 생물학적 상수로 간주하지 않는다.
모든 물리량은 SI 단위(m, s, m/s, rad).
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
R_SOCIAL = 0.55   # 자발적으로 걷는 BASE 개체의 약한 cohesion
R_C      = 0.40   # v1식 ESCAPE 상태 전파 반경
R_FLOW   = 0.45   # 실제 movement-flow 방향을 감지하는 이웃 반경

# behavioral social wave의 누적 경로거리 한계.
# 약 3 m는 현재 관찰을 맞추기 위한 개발/튜닝값이며 생물학적 상수가 아니다.
SOCIAL_PROP_RANGE = 3.0

# 밀집 crowd relaxation. behavioral ESCAPE와 분리된 수동/집단 재배치.
# 아래 값들은 v0.2 가이드의 초기 튜닝값을 시작점으로 사용한다.
R_PRESSURE = 0.17
PRESSURE_THRESHOLD = 0.20
V_CROWD_MAX = 0.07

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
# 0.4 m/s -> 0.5, 0.8 m/s 이상 -> 1.0.
# 속도 효과를 유지하면서 저속 비교실험에서 direct seed가 과도하게 약해지는 것을 막는다.
V_THREAT_REF = 0.80

# BASE -> movement bout hazard를 direct/social로 분리한다.
# robot: 거리+closing speed+개체 민감도
# social: v1식 ESCAPE 비율 C_i + 개체 social sensitivity
LAMBDA_ROBOT  = 20.0
LAMBDA_SOCIAL = 10.0
BETA = 0.60

# old code/logger compatibility only
LAMBDA_RESPONSE = LAMBDA_ROBOT

# 이웃 effective velocity의 짧은 방향 기억
TAU_FLOW = 0.12

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
# v1식 실제 overlap correction. social sensitivity는 적용하지 않는다.
CONTACT_ITERS = 2
MAX_OVERLAP_PUSH = 0.02

# active desired direction이 벽 밖을 향하면 normal 성분을 제거하고 tangent를 유지.
R_WALL_FLOW = 0.25

# ----------------------------------------------------------------------------
# Rare direct low responders
# direct robot sensitivity만 0. social-flow/contact는 정상.
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
ROBOT_VEL     = np.array([0.10, 0.0])

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