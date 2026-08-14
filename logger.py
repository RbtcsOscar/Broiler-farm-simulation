"""
logger.py
=========
v0.4.16 front-wall social-split + robot-frame density diagnostics.

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
            "burst_count", "burst_start_count", "rush_candidate_count",
            "social_support_stop_count", "social_distance_stop_count", "social_density_stop_count",
            "crowd_active_ratio", "near_wall_ratio",
            "reaction_rest_ratio", "bout_end_count",
            "threat_loss_end_count", "direct_trigger_count", "social_trigger_count",
            "side_direct_trigger_count", "side_cue_active_ratio",
            "close_reaction_start_count", "social_direction_dominant_ratio",
            "robot_direction_weight_mean", "social_direction_weight_mean",
            "escape_forward_dominant_ratio", "low_sensitivity_escape_ratio",
            "direct_origin_escape_ratio", "social_origin_escape_ratio",
            "refill_active_ratio", "refill_eligible_ratio",
            "refill_centroid_offset_mean",
            "crowd_move_ratio", "D_raw_max", "D_eff_max",
            "v_crowd_mean", "low_sensitivity_count",
            "social_drive_max", "local_flow_speed_max",
            "push_present_ratio", "flow_present_ratio", "v_crowd_max",
            "perceived_threat_mean", "perceived_threat_max",
            "stop_threat_mean", "stop_threat_max",
            "social_excitation_mean", "social_excitation_max",
            "contact_social_source_ratio",
            "behavioral_source_ratio", "front_gate_ratio", "reverse_flow_ratio",
            "rear_recruit_block_ratio",
            "wall_split_active_ratio", "wall_social_conflict_ratio",
            "wall_split_up_ratio", "wall_split_down_ratio",
            "front_sector_count", "front_sector_density_ratio",
            "side_sector_count", "side_sector_density_ratio", "front_side_density_ratio",
            "escape_radial_cos_mean", "escape_forward_cos_mean",
            "escape_lateral_abs_mean",
            "escape_outward_lateral_mean", "escape_outward_lateral_ratio",
            "wake_count", "wake_density", "wake_density_ratio",
            "wake_far_count", "wake_far_density", "wake_far_density_ratio",
        ])
        self._snaps = []       # (t, x, y, theta, state, S, robot_xy)
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

        # Diagnostic robot-wake occupancy. The rectangle lies immediately
        # behind the robot and is used only to quantify post-pass refill.
        wake_count = 0
        wake_density = np.nan
        wake_density_ratio = np.nan
        wake_far_count = 0
        wake_far_density = np.nan
        wake_far_density_ratio = np.nan
        # Robot-frame occupancy diagnostics for the user's observed
        # 'front flock layer vs side accumulation' question.
        #
        # Equal-area sectors derived only from existing R_SOCIAL:
        # front: length 2R, width 2R
        # sides combined: longitudinal span 2R, two lateral strips of width R
        # => both areas = 4 R^2.
        front_sector_count = 0
        front_sector_density_ratio = np.nan
        side_sector_count = 0
        side_sector_density_ratio = np.nan
        front_side_density_ratio = np.nan

        if robot_pos is not None:
            t_hat = np.asarray(getattr(cm, "last_robot_forward", [1.0, 0.0]), dtype=float)
            tn = float(np.hypot(t_hat[0], t_hat[1]))
            if tn > 1e-9:
                t_hat = t_hat / tn
                n_hat = np.array([-t_hat[1], t_hat[0]])
                rear_offset = C.ROBOT_RADIUS + C.CHICKEN_RADIUS
                wake_len = float(getattr(C, "WAKE_DIAG_LENGTH", 1.0))
                wake_half = float(getattr(
                    C, "WAKE_DIAG_HALF_WIDTH", C.ROBOT_RADIUS + C.CHICKEN_RADIUS
                ))

                rel = np.column_stack([cm.x - rx, cm.y - ry])
                longitudinal = rel @ t_hat
                lateral = rel @ n_hat

                # Equal-area front/side occupancy sectors.
                Rsec = float(C.R_SOCIAL)
                body_offset = float(C.ROBOT_RADIUS + C.CHICKEN_RADIUS)

                front_mask = (
                    (longitudinal >= body_offset)
                    & (longitudinal <= body_offset + 2.0 * Rsec)
                    & (np.abs(lateral) <= Rsec)
                )
                side_mask = (
                    (np.abs(longitudinal) <= Rsec)
                    & (np.abs(lateral) >= body_offset)
                    & (np.abs(lateral) <= body_offset + Rsec)
                )

                # Report only if the front rectangle and both side strips are
                # entirely within the house, avoiding truncated-area bias.
                front_center = (
                    np.asarray(robot_pos, dtype=float)
                    + (body_offset + Rsec) * t_hat
                )
                front_corners = []
                for sgn_t in (-1.0, 1.0):
                    for sgn_n in (-1.0, 1.0):
                        front_corners.append(
                            front_center
                            + sgn_t * Rsec * t_hat
                            + sgn_n * Rsec * n_hat
                        )
                front_corners = np.asarray(front_corners)

                side_centers = [
                    np.asarray(robot_pos, dtype=float)
                    + (body_offset + 0.5 * Rsec) * n_hat,
                    np.asarray(robot_pos, dtype=float)
                    - (body_offset + 0.5 * Rsec) * n_hat,
                ]
                side_corners = []
                for sc in side_centers:
                    for sgn_t in (-1.0, 1.0):
                        for sgn_n in (-1.0, 1.0):
                            side_corners.append(
                                sc
                                + sgn_t * Rsec * t_hat
                                + sgn_n * 0.5 * Rsec * n_hat
                            )
                side_corners = np.asarray(side_corners)

                front_inside = (
                    np.all((front_corners[:, 0] >= 0.0) & (front_corners[:, 0] <= cm.world.L))
                    & np.all((front_corners[:, 1] >= 0.0) & (front_corners[:, 1] <= cm.world.W))
                )
                side_inside = (
                    np.all((side_corners[:, 0] >= 0.0) & (side_corners[:, 0] <= cm.world.L))
                    & np.all((side_corners[:, 1] >= 0.0) & (side_corners[:, 1] <= cm.world.W))
                )

                if front_inside and side_inside:
                    global_density = cm.N / max(cm.world.L * cm.world.W, 1e-9)
                    sector_area = max(4.0 * Rsec * Rsec, 1e-9)

                    front_sector_count = int(front_mask.sum())
                    side_sector_count = int(side_mask.sum())

                    front_density = front_sector_count / sector_area
                    side_density = side_sector_count / sector_area

                    front_sector_density_ratio = (
                        front_density / max(global_density, 1e-9)
                    )
                    side_sector_density_ratio = (
                        side_density / max(global_density, 1e-9)
                    )
                    front_side_density_ratio = (
                        front_density / max(side_density, 1e-9)
                    )

                wake_mask = (
                    (longitudinal <= -rear_offset)
                    & (longitudinal >= -(rear_offset + wake_len))
                    & (np.abs(lateral) <= wake_half)
                )

                # Only report density when the whole diagnostic rectangle is
                # inside the house, avoiding edge-area bias near entry/exit.
                center = np.asarray(robot_pos, dtype=float) - (rear_offset + 0.5 * wake_len) * t_hat
                corners = []
                for sgn_t in (-1.0, 1.0):
                    for sgn_n in (-1.0, 1.0):
                        corners.append(
                            center
                            + sgn_t * 0.5 * wake_len * t_hat
                            + sgn_n * wake_half * n_hat
                        )
                corners = np.asarray(corners)
                inside = (
                    np.all((corners[:, 0] >= 0.0) & (corners[:, 0] <= cm.world.L))
                    & np.all((corners[:, 1] >= 0.0) & (corners[:, 1] <= cm.world.W))
                )
                if inside:
                    wake_count = int(wake_mask.sum())
                    wake_area = max(wake_len * (2.0 * wake_half), 1e-9)
                    wake_density = wake_count / wake_area
                    global_density = cm.N / max(cm.world.L * cm.world.W, 1e-9)
                    wake_density_ratio = wake_density / max(global_density, 1e-9)

                    # Same-size band one additional wake length farther behind.
                    # At 0.20 m/s this corresponds to an older, post-pass region
                    # and is more informative about refill than the immediate wake.
                    far_mask = (
                        (longitudinal < -(rear_offset + wake_len))
                        & (longitudinal >= -(rear_offset + 2.0 * wake_len))
                        & (np.abs(lateral) <= wake_half)
                    )
                    far_center = (
                        np.asarray(robot_pos, dtype=float)
                        - (rear_offset + 1.5 * wake_len) * t_hat
                    )
                    far_corners = []
                    for sgn_t in (-1.0, 1.0):
                        for sgn_n in (-1.0, 1.0):
                            far_corners.append(
                                far_center
                                + sgn_t * 0.5 * wake_len * t_hat
                                + sgn_n * wake_half * n_hat
                            )
                    far_corners = np.asarray(far_corners)
                    far_inside = (
                        np.all((far_corners[:, 0] >= 0.0) & (far_corners[:, 0] <= cm.world.L))
                        & np.all((far_corners[:, 1] >= 0.0) & (far_corners[:, 1] <= cm.world.W))
                    )
                    if far_inside:
                        wake_far_count = int(far_mask.sum())
                        wake_far_density = wake_far_count / wake_area
                        wake_far_density_ratio = (
                            wake_far_density / max(global_density, 1e-9)
                        )

        self._w.writerow([
            f"{sim_time:.3f}", f"{cm.escape_ratio():.4f}", f"{cm.mean_speed():.4f}",
            f"{rx:.3f}", f"{ry:.3f}", f"{rmin:.3f}",
            f"{mean_nn:.4f}", f"{float(S.mean()):.4f}",
            getattr(cm, "last_burst_count", 0),
            getattr(cm, "last_burst_start_count", 0),
            getattr(cm, "last_rush_candidate_count", 0),
            getattr(cm, "last_social_support_stop_count", 0),
            getattr(cm, "last_social_distance_stop_count", 0),
            getattr(cm, "last_social_density_stop_count", 0),
            f"{getattr(cm, 'last_crowd_active_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_near_wall_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_reaction_rest_ratio', 0.0):.4f}",
            getattr(cm, "last_bout_end_count", 0),
            getattr(cm, "last_threat_loss_end_count", 0),
            getattr(cm, "last_direct_trigger_count", 0),
            getattr(cm, "last_social_trigger_count", 0),
            getattr(cm, "last_side_direct_trigger_count", 0),
            f"{getattr(cm, 'last_side_pass_cue_active_ratio', 0.0):.4f}",
            getattr(cm, "last_close_direct_override_count", 0),
            f"{getattr(cm, 'last_social_direction_dominant_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_robot_direction_weight_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_social_direction_weight_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_escape_forward_dominant_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_low_sensitivity_escape_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_direct_origin_escape_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_social_origin_escape_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_refill_active_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_refill_eligible_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_refill_centroid_offset_mean', 0.0):.4f}",
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
            f"{getattr(cm, 'last_perceived_threat_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_perceived_threat_max', 0.0):.4f}",
            f"{getattr(cm, 'last_stop_threat_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_stop_threat_max', 0.0):.4f}",
            f"{getattr(cm, 'last_social_excitation_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_social_excitation_max', 0.0):.4f}",
            f"{getattr(cm, 'last_contact_social_source_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_behavioral_source_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_front_gate_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_reverse_flow_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_rear_recruit_block_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_wall_split_active_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_wall_social_conflict_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_wall_split_up_ratio', 0.0):.4f}",
            f"{getattr(cm, 'last_wall_split_down_ratio', 0.0):.4f}",
            front_sector_count,
            f"{front_sector_density_ratio:.4f}" if np.isfinite(front_sector_density_ratio) else "nan",
            side_sector_count,
            f"{side_sector_density_ratio:.4f}" if np.isfinite(side_sector_density_ratio) else "nan",
            f"{front_side_density_ratio:.4f}" if np.isfinite(front_side_density_ratio) else "nan",
            f"{getattr(cm, 'last_escape_radial_cos_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_escape_forward_cos_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_escape_lateral_abs_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_escape_outward_lateral_mean', 0.0):.4f}",
            f"{getattr(cm, 'last_escape_outward_lateral_ratio', 0.0):.4f}",
            wake_count,
            f"{wake_density:.4f}" if np.isfinite(wake_density) else "nan",
            f"{wake_density_ratio:.4f}" if np.isfinite(wake_density_ratio) else "nan",
            wake_far_count,
            f"{wake_far_density:.4f}" if np.isfinite(wake_far_density) else "nan",
            f"{wake_far_density_ratio:.4f}" if np.isfinite(wake_far_density_ratio) else "nan",
        ])

        self._k += 1
        if self._k % self._snap_every == 0:
            robot_xy = (
                np.array([np.nan, np.nan], dtype=float)
                if robot_pos is None
                else np.asarray(robot_pos, dtype=float).copy()
            )
            self._snaps.append((
                sim_time, cm.x.copy(), cm.y.copy(), cm.theta.copy(),
                cm.state.copy(), S.copy(), robot_xy,
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
            robot_xy = np.stack([s[6] for s in self._snaps])
            np.savez_compressed(self.snap_path, t=t, x=x, y=y,
                                theta=th, state=st, S=S, robot_xy=robot_xy)
        print(f"[logger] saved: {self.csv_path}")
        if self._snaps:
            print(f"[logger] saved: {self.snap_path}")
