"""
stimulus.py
===========
v0.3.3: direct robot threat + v1-style social activation.

역할 분리
- Q_robot: 로봇에 의해 movement bout가 시작될 확률
- C_i: R_C 내 ESCAPE 이웃 비율 -> social activation
- flow vector: chickens.py에서 방향에만 사용
"""

import numpy as np
import config as C

EPS = 1e-9


def robot_threat(clearance, closing_speed):
    """Return (Q_robot, q_distance, q_closing).

    q_distance = 1 - clearance / ROBOT_CUE_CLEARANCE, clipped to [0, 1].
    q_closing = positive closing speed / V_THREAT_REF, clipped to [0, 1].
    로봇이 멀어지는 중이면 q_closing=0 -> direct threat=0.
    """
    q_distance = np.clip(
        1.0 - clearance / C.ROBOT_CUE_CLEARANCE,
        0.0, 1.0
    )
    v_close = np.maximum(0.0, closing_speed)
    q_closing = np.clip(
        v_close / max(C.V_THREAT_REF, EPS),
        0.0, 1.0
    )
    return q_distance * q_closing, q_distance, q_closing


def social_stimulus(escape_count, neighbor_count):
    """v1-style social activation fraction C_i = N_ESCAPE / N_neighbor."""
    c = np.zeros_like(escape_count, dtype=float)
    valid = neighbor_count > 0
    c[valid] = escape_count[valid] / neighbor_count[valid]
    return c


def flow_cue(flow_memory):
    """Return diagnostic magnitude + unit direction of actual velocity flow.

    Flow magnitude is NOT the main social trigger in v0.3.1.
    It determines propagated direction; C_i determines activation.
    """
    flow_speed = np.hypot(flow_memory[:, 0], flow_memory[:, 1])
    q_flow = np.clip(flow_speed / max(C.V_WALK, EPS), 0.0, 1.0)
    flow_dir = np.zeros_like(flow_memory)
    valid = flow_speed > EPS
    flow_dir[valid, 0] = flow_memory[valid, 0] / flow_speed[valid]
    flow_dir[valid, 1] = flow_memory[valid, 1] / flow_speed[valid]
    return q_flow, flow_dir, flow_speed


def response_probability(robot_sensitivity, social_sensitivity, habituation,
                         q_robot, social_fraction, dt):
    """Unified bout-start probability with separate robot/social rates.

    drive_robot  = s_robot * habituation * Q_robot
    drive_social = s_social * BETA * C_i

    hazard = LAMBDA_ROBOT*drive_robot + LAMBDA_SOCIAL*drive_social
    P = 1-exp(-hazard*dt)

    First response and post-rest response use the same equation.
    """
    drive_robot = (
        robot_sensitivity * habituation * np.maximum(0.0, q_robot)
    )
    drive_social = (
        social_sensitivity * C.BETA * np.maximum(0.0, social_fraction)
    )
    hazard_robot = C.LAMBDA_ROBOT * drive_robot
    hazard_social = C.LAMBDA_SOCIAL * drive_social
    hazard = hazard_robot + hazard_social
    p = 1.0 - np.exp(-hazard * dt)
    drive = drive_robot + drive_social
    return p, drive, drive_robot, drive_social, hazard_robot, hazard_social


def positive_rate(now, prev, dt):
    return np.maximum(0.0, (now - prev) / max(dt, EPS))