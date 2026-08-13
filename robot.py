"""
robot.py
========
1차 버전은 autonomous navigation 을 구현하지 않고, 미리 정한 trajectory 만 수행한다 (§30).

기본: 등속 직선 (straight line, constant speed).
필요하면 waypoints 로 piecewise 경로도 사용 가능.
"""

import numpy as np
import config as C


class Robot:
    def __init__(self):
        self.enabled = C.ROBOT_ENABLED
        self.start = np.array(C.ROBOT_START, dtype=float)
        self.vel = np.array(C.ROBOT_VEL, dtype=float)
        self.radius = C.ROBOT_RADIUS
        self.pos = self.start.copy()
        self.active = False          # settling 이후 활성화

        # piecewise 경로를 쓰려면 waypoints 를 채운다 (없으면 등속 직선)
        self.waypoints = None        # e.g. [(x,y,t), ...]

    def update(self, sim_time, dt):
        """현재 sim_time 에 맞춰 로봇 위치를 갱신하고 반환.
        비활성/미투입 상태면 None 을 반환한다 (닭에게 자극 없음)."""
        if not self.enabled:
            return None
        if sim_time < C.SETTLE_TIME:
            self.pos = self.start.copy()
            self.active = False
            return None

        self.active = True
        t = sim_time - C.SETTLE_TIME
        if self.waypoints is None:
            self.pos = self.start + self.vel * t
        else:
            self.pos = self._piecewise(t)

        # 축사 오른쪽 밖으로 완전히 나가면 자극 종료
        if self.pos[0] > C.WORLD_LENGTH + 1.0:
            self.active = False
            return None
        return self.pos.copy()

    def _piecewise(self, t):
        wp = self.waypoints
        for i in range(len(wp) - 1):
            x0, y0, t0 = wp[i]
            x1, y1, t1 = wp[i + 1]
            if t0 <= t <= t1:
                a = (t - t0) / max(t1 - t0, 1e-9)
                return np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0)])
        x, y, _ = wp[-1]
        return np.array([x, y])
