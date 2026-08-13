"""
simulation.py
=============
World / ChickenManager / Robot / Renderer / Logger 를 묶어 실행한다.

렌더 FPS 와 physics timestep(DT) 을 분리한다 (§25):
    한 프레임마다 SIM_STEPS_PER_FRAME 번 physics step 을 돌려 배속을 조절한다.
"""

import numpy as np
import pygame
import config as C

from world import World
from chickens import ChickenManager
from robot import Robot
from renderer import Renderer
from logger import Logger


class Simulation:
    def __init__(self, headless=False):
        self.headless = headless
        self.rng = np.random.default_rng(C.RANDOM_SEED)

        self.world = World()
        self.cm = ChickenManager(self.world, self.rng)
        self.robot = Robot()
        self.logger = Logger(C.LOG_ENABLED)

        self.sim_time = 0.0
        self.steps_per_frame = C.SIM_STEPS_PER_FRAME
        self.paused = False

        self.renderer = None if headless else Renderer(self.world, self.robot)

    # ------------------------------------------------------------------ #
    def reset(self):
        self.rng = np.random.default_rng(self.rng.integers(1 << 30))
        self.cm = ChickenManager(self.world, self.rng)
        self.robot = Robot()
        self.sim_time = 0.0
        if self.renderer:
            self.renderer.robot = self.robot

    # ------------------------------------------------------------------ #
    def physics_step(self):
        robot_pos = self.robot.update(self.sim_time, C.DT)
        info = self.cm.step(robot_pos, C.DT)
        self.sim_time += C.DT
        self.logger.log_step(self.sim_time, self.cm, robot_pos, info["S"])
        return robot_pos

    # ------------------------------------------------------------------ #
    def run(self, max_time=None):
        if self.headless:
            return self._run_headless(max_time)

        clock = pygame.time.Clock()
        running = True
        robot_pos = None
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    running = self._handle_key(e)

            if not self.paused:
                for _ in range(self.steps_per_frame):
                    robot_pos = self.physics_step()
                    if max_time and self.sim_time >= max_time:
                        running = False
                        break
            else:
                robot_pos = (None if not self.robot.active else self.robot.pos)

            fps = clock.get_fps()
            self.renderer.draw(self.cm, robot_pos, self.sim_time, fps, self.paused)
            clock.tick(C.TARGET_FPS)

        self.logger.close()
        self.renderer.quit()

    def _handle_key(self, e):
        if e.key == pygame.K_ESCAPE:
            return False
        elif e.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif e.key == pygame.K_h:
            C.HABITUATION_ON = not C.HABITUATION_ON
        elif e.key == pygame.K_r:
            self.reset()
        elif e.key == pygame.K_g:
            self.renderer.draw_head = not self.renderer.draw_head
        elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.steps_per_frame = min(20, self.steps_per_frame + 1)
        elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        return True

    # ------------------------------------------------------------------ #
    def _run_headless(self, max_time):
        max_time = max_time or 60.0
        n = int(max_time / C.DT)
        for _ in range(n):
            self.physics_step()
        self.logger.close()
        print(f"[headless] done: {self.sim_time:.1f}s, "
              f"escape_ratio={self.cm.escape_ratio()*100:.1f}%")
