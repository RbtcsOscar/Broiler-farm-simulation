"""
world.py
========
환경의 geometry 만 관리한다 (가이드 §28).

핵심 원칙 (§20, §34):
    - visual_object 와 collision_object 를 분리한다.
    - 벽, 급이통  : collision = True  (hard obstacle)
    - 급이/급수 라인 : collision = False (표시만)

닭은 head 방향으로만 이동하므로, 벽/급이통 충돌은
"위치를 경계로 밀어내고, 필요하면 heading 을 안쪽으로 튕겨준다".
"""

import numpy as np
import config as C


class World:
    def __init__(self):
        self.L = C.WORLD_LENGTH
        self.W = C.WORLD_WIDTH
        self.rc = C.CHICKEN_RADIUS

        # ---- collision objects ----
        self.feeder_pans = self._create_feeder_pans()      # (Nf, 2)
        self.feeder_r = C.FEEDER_RADIUS

        # ---- visual-only objects ----
        self.feeder_line_y  = C.FEEDER_LINE_Y
        self.drinker_line_y = C.DRINKER_LINE_Y
        self.line_x0 = C.LINE_START_X
        self.line_x1 = C.LINE_START_X + C.LINE_LENGTH

    # ------------------------------------------------------------------ #
    # geometry 생성
    # ------------------------------------------------------------------ #
    def _create_feeder_pans(self):
        """급이통 10개: x_k = 1.2 + k*(15.6/11), k=1..10, y=1.5 (§3.4)."""
        k = np.arange(1, C.FEEDER_COUNT + 1)
        xs = C.LINE_START_X + k * (C.LINE_LENGTH / (C.FEEDER_COUNT + 1))
        ys = np.full_like(xs, C.FEEDER_LINE_Y, dtype=float)
        return np.column_stack([xs, ys])                   # (10, 2)

    # ------------------------------------------------------------------ #
    # 18. 닭-벽 충돌  (hard constraint: 절대 관통 불가)
    # ------------------------------------------------------------------ #
    def resolve_wall_collision(self, x, y, theta=None):
        """위치를 [r_c, L-r_c] x [r_c, W-r_c] 로 clamp 만 한다. (in-place)

        v0.2: heading 반사는 제거했다. 벽에서의 방향 처리(접선 flow)는
        ChickenManager._apply_wall_flow 가 desired direction 단계에서 담당한다.
        world 는 geometry hard constraint(관통 금지)만 책임진다.
        theta 인자는 하위호환을 위해 남겨두되 사용하지 않는다.
        """
        rc = self.rc
        hit = (x < rc) | (x > self.L - rc) | (y < rc) | (y > self.W - rc)
        np.clip(x, rc, self.L - rc, out=x)
        np.clip(y, rc, self.W - rc, out=y)
        return hit

    # ------------------------------------------------------------------ #
    # 19. 닭-급이통 충돌  (원형 hard obstacle)
    # ------------------------------------------------------------------ #
    def resolve_feeder_collision(self, x, y):
        """급이통과 겹친 닭을 경계(r_c + r_f)까지 반경 방향으로 밀어낸다. (in-place)"""
        min_d = self.rc + self.feeder_r          # 0.205 m
        pans = self.feeder_pans
        for k in range(pans.shape[0]):
            dx = x - pans[k, 0]
            dy = y - pans[k, 1]
            d = np.hypot(dx, dy)
            overlap = d < min_d
            if not np.any(overlap):
                continue
            # d==0 인 예외는 임의 방향으로 밀기
            safe = d[overlap]
            zero = safe < 1e-9
            ux = np.where(zero, 1.0, dx[overlap] / np.where(zero, 1.0, safe))
            uy = np.where(zero, 0.0, dy[overlap] / np.where(zero, 1.0, safe))
            x[overlap] = pans[k, 0] + ux * min_d
            y[overlap] = pans[k, 1] + uy * min_d

    # ------------------------------------------------------------------ #
    # 초기 배치 유효성
    # ------------------------------------------------------------------ #
    def is_inside_world(self, x, y):
        rc = self.rc
        return (x >= rc) & (x <= self.L - rc) & (y >= rc) & (y <= self.W - rc)

    def min_dist_to_pans(self, x, y):
        """각 점에서 가장 가까운 급이통 중심까지 거리."""
        dx = x[:, None] - self.feeder_pans[None, :, 0]
        dy = y[:, None] - self.feeder_pans[None, :, 1]
        return np.hypot(dx, dy).min(axis=1)
