"""
renderer.py
===========
Pygame 렌더러 (가이드 §31).

물리는 m 단위 그대로 두고, 여기서만 PX_PER_M 로 화면 좌표로 스케일한다.
닭   : 상태별 색 원 + 앞쪽(head 방향) 작은 빨간 삼각형
로봇 : 원 + heading + R_stim 링
급이/급수 라인 : dashed (visual only)
급이통 : 채워진 원 (hard obstacle)

world y 는 위쪽이 +y 가 되도록 화면에서 뒤집는다.
"""

import numpy as np
import pygame
import config as C

BASE, ESCAPE = 0, 1


class Renderer:
    def __init__(self, world, robot):
        pygame.init()
        self.world = world
        self.robot = robot
        self.px = C.PX_PER_M
        self.margin = C.MARGIN_PX
        self.hud_h = C.HUD_HEIGHT

        self.win_w = int(2 * self.margin + world.L * self.px)
        self.win_h = int(self.hud_h + 2 * self.margin + world.W * self.px)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("2D 육계 군집–로봇 상호작용 시뮬레이터")

        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.font_small = pygame.font.SysFont("consolas,menlo,monospace", 13)

        # 닭 글리프 크기(px)
        self.r_glyph = max(2.0, world.rc * self.px * C.CHICKEN_DRAW_SCALE)
        self.head_len = self.r_glyph * 1.05   # 몸통 밖으로 뻗는 부리 길이
        self.head_hw = self.r_glyph * 0.72    # 부리 밑변 half-width
        self.draw_head = True        # [g] 키로 토글

    # ---- 좌표 변환 (m -> px, y flip) ---- #
    def wx(self, x):
        return self.margin + x * self.px

    def wy(self, y):
        return self.hud_h + self.margin + (self.world.W - y) * self.px

    def _dashed_hline(self, y_world, x0, x1, color, dash=10, gap=8, width=2):
        sy = self.wy(y_world)
        sx0 = self.wx(x0)
        sx1 = self.wx(x1)
        x = sx0
        while x < sx1:
            xe = min(x + dash, sx1)
            pygame.draw.line(self.screen, color, (x, sy), (xe, sy), width)
            x = xe + gap

    # ------------------------------------------------------------------ #
    def draw(self, cm, robot_pos, sim_time, fps, paused):
        s = self.screen
        s.fill(C.COL_BG)

        # ---- 벽 (world 경계 사각형) ----
        rect = pygame.Rect(self.wx(0), self.wy(self.world.W),
                           self.world.L * self.px, self.world.W * self.px)
        pygame.draw.rect(s, C.COL_WALL, rect, width=3)

        # ---- 급이/급수 라인 (visual only, dashed) ----
        self._dashed_hline(self.world.feeder_line_y, self.world.line_x0,
                           self.world.line_x1, C.COL_FEEDER_LN)
        self._dashed_hline(self.world.drinker_line_y, self.world.line_x0,
                           self.world.line_x1, C.COL_DRINK_LN)

        # ---- 급이통 (hard obstacle) ----
        rf = self.world.feeder_r * self.px
        for (fx, fy) in self.world.feeder_pans:
            c = (int(self.wx(fx)), int(self.wy(fy)))
            pygame.draw.circle(s, C.COL_PAN, c, int(rf))
            pygame.draw.circle(s, C.COL_PAN_EDGE, c, int(rf), 2)

        # ---- 닭 ----
        self._draw_chickens(cm)

        # ---- 로봇 ----
        if robot_pos is not None:
            self._draw_robot(robot_pos)

        # ---- HUD ----
        self._draw_hud(cm, robot_pos, sim_time, fps, paused)

        pygame.display.flip()

    # ------------------------------------------------------------------ #
    def _draw_chickens(self, cm):
        s = self.screen
        sx = self.wx(cm.x)
        sy = self.wy(cm.y)
        cos_t = np.cos(cm.theta)
        sin_t = np.sin(cm.theta)      # world; 화면에서는 -sin (y flip)

        rg = self.r_glyph
        hl = self.head_len
        hw = self.head_hw

        state = cm.state
        burst_active = (cm.burst_timer > 0) & (state == ESCAPE)

        # 색: BASE / ESCAPE-WALK / ESCAPE-BURST
        col_base = C.COL_BASE
        col_walk = C.COL_WALK
        col_burst = C.COL_BURST
        col_head = C.COL_HEAD

        r_int = max(2, int(round(rg)))
        for i in range(cm.N):
            cx = sx[i]; cy = sy[i]
            if state[i] == BASE:
                col = col_base
            elif burst_active[i]:
                col = col_burst
            else:
                col = col_walk
            pygame.draw.circle(s, col, (int(cx), int(cy)), r_int)

            if not self.draw_head:
                continue

            # head 방향 단위벡터 (화면 좌표: y 뒤집힘)
            dx = cos_t[i]
            dy = -sin_t[i]
            px_ = -dy   # perpendicular
            py_ = dx
            tip = (cx + dx * (rg + hl), cy + dy * (rg + hl))
            b1 = (cx + dx * rg + px_ * hw, cy + dy * rg + py_ * hw)
            b2 = (cx + dx * rg - px_ * hw, cy + dy * rg - py_ * hw)
            pygame.draw.polygon(s, col_head, (tip, b1, b2))

    # ------------------------------------------------------------------ #
    def _draw_robot(self, robot_pos):
        s = self.screen
        cx = int(self.wx(robot_pos[0]))
        cy = int(self.wy(robot_pos[1]))
        rr = int(self.robot.radius * self.px)

        # 반응 영역 링 (v0.2.2: clearance 기반 onset. legacy 면 R_STIM)
        if C.ROBOT_DRAW_STIM_RING:
            if getattr(C, "USE_CLEARANCE_DIRECT_STIMULUS", True):
                ring_r = C.ROBOT_RADIUS + C.CHICKEN_RADIUS + C.DIRECT_ONSET_CLEARANCE
            else:
                ring_r = C.R_STIM
            ring = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*C.COL_ROBOT_RING, 40), (cx, cy),
                              int(ring_r * self.px))
            s.blit(ring, (0, 0))
            pygame.draw.circle(s, C.COL_ROBOT_RING, (cx, cy),
                              int(ring_r * self.px), 1)

        pygame.draw.circle(s, C.COL_ROBOT, (cx, cy), max(3, rr))
        # heading (진행 방향)
        v = self.robot.vel
        n = np.hypot(v[0], v[1]) + 1e-9
        hx = cx + (v[0] / n) * rr * 1.8
        hy = cy - (v[1] / n) * rr * 1.8
        pygame.draw.line(s, (20, 30, 40), (cx, cy), (hx, hy), 3)

    # ------------------------------------------------------------------ #
    def _draw_hud(self, cm, robot_pos, sim_time, fps, paused):
        s = self.screen
        esc = cm.escape_ratio() * 100.0
        vbar = cm.mean_speed()
        rob = "OUT" if robot_pos is None else f"({robot_pos[0]:.1f},{robot_pos[1]:.1f})"
        hab = "ON" if C.HABITUATION_ON else "OFF"
        pause_tag = "  [PAUSED]" if paused else ""

        line1 = (f" t={sim_time:6.2f}s   N={cm.N}   ESCAPE={esc:5.1f}%   "
                 f"v̄={vbar:4.2f}m/s   robot={rob}   habit={hab}   {fps:4.1f}fps{pause_tag}")
        line2 = (" [space] pause   [h] habituation   [r] reset   "
                 "[+/-] speed   [g] head on/off   [esc] quit   "
                 "BASE=beige  WALK=orange  BURST=red  head▶red")

        s.blit(self.font.render(line1, True, C.COL_HUD_TEXT), (10, 10))
        s.blit(self.font_small.render(line2, True, C.COL_HUD_DIM), (10, 36))

    def quit(self):
        pygame.quit()
