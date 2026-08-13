"""
logger.py
=========
반드시 기록해야 할 데이터 (가이드 §32).

가벼운 CSV summary 를 매 step 기록하고,
필요하면 전체 상태 스냅샷(npz)을 주기적으로 저장한다.
"""

import os
import csv
import time
import numpy as np
import config as C


class Logger:
    def __init__(self, enabled=True, log_dir=None):
        self.enabled = enabled
        if not enabled:
            return
        self.dir = log_dir or C.LOG_DIR
        os.makedirs(self.dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.dir, f"summary_{stamp}.csv")
        self.snap_path = os.path.join(self.dir, f"snapshots_{stamp}.npz")

        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "t", "escape_ratio", "mean_speed",
            "robot_x", "robot_y", "robot_min_dist",
            "mean_nn_dist", "mean_S",
            "burst_count", "crowd_active_ratio", "near_wall_ratio",
            "reaction_rest_ratio", "bout_end_count",
            "crowd_move_ratio", "D_raw_max", "D_eff_max",
            "v_crowd_mean", "low_responder_count",
            "social_drive_max", "local_flow_speed_max",
            "push_present_ratio", "flow_present_ratio", "v_crowd_max",
        ])
        self._snaps = []       # (t, x, y, theta, state, S)
        self._snap_every = 20  # 1 s 마다 스냅샷 (20Hz 기준)
        self._k = 0

    def log_step(self, sim_time, cm, robot_pos, S):
        if not self.enabled:
            return
        if robot_pos is None:
            rx = ry = np.nan
            rmin = np.nan
        else:
            rx, ry = float(robot_pos[0]), float(robot_pos[1])
            rmin = float(np.hypot(cm.x - rx, cm.y - ry).min())

        finite_nn = cm.nn_dist[np.isfinite(cm.nn_dist)]
        mean_nn = float(finite_nn.mean()) if finite_nn.size else np.nan

        self._w.writerow([
            f"{sim_time:.3f}", f"{cm.escape_ratio():.4f}", f"{cm.mean_speed():.4f}",
            f"{rx:.3f}", f"{ry:.3f}", f"{rmin:.3f}",
            f"{mean_nn:.4f}", f"{float(S.mean()):.4f}",
            getattr(cm, "last_burst_count", 0),
            f"{getattr(cm, 'last_crowd_active_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_near_wall_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_reaction_rest_ratio', 0.0):.4f}",
            getattr(cm, "last_bout_end_count", 0),
            f"{getattr(cm, 'last_crowd_move_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_D_raw_max', 0.0):.4f}",
            f"{getattr(cm, 'last_D_eff_max', 0.0):.4f}",
            f"{getattr(cm, 'last_v_crowd_mean', 0.0):.4f}",
            getattr(cm, "low_responder_count", 0),
            f"{getattr(cm, 'last_social_drive_max', 0.0):.4f}",
            f"{getattr(cm, 'last_local_flow_speed_max', 0.0):.4f}",
            f"{getattr(cm, 'last_push_present_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_flow_present_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_v_crowd_max', 0.0):.4f}",
        ])

        self._k += 1
        if self._k % self._snap_every == 0:
            self._snaps.append((
                sim_time, cm.x.copy(), cm.y.copy(), cm.theta.copy(),
                cm.state.copy(), S.copy(),
            ))

    def close(self):
        if not self.enabled:
            return
        self._f.close()
        if self._snaps:
            t = np.array([s[0] for s in self._snaps])
            x = np.stack([s[1] for s in self._snaps])
            y = np.stack([s[2] for s in self._snaps])
            th = np.stack([s[3] for s in self._snaps])
            st = np.stack([s[4] for s in self._snaps])
            S = np.stack([s[5] for s in self._snaps])
            np.savez_compressed(self.snap_path, t=t, x=x, y=y,
                                theta=th, state=st, S=S)
        print(f"[logger] saved: {self.csv_path}")
        if self._snaps:
            print(f"[logger] saved: {self.snap_path}")
