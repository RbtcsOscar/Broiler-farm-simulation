"""
chickens.py
===========
v0.3.3 threat-front advective crowd-flow + collision-angle seed model.

핵심 loop:
robot threat / v1-style social activation
 -> probabilistic short movement bout
 -> physical overlap displacement
 -> effective velocity(active + passive push)
 -> neighbor local flow
 -> individual social response
 -> next movement bout

BASE=0, ESCAPE=1은 renderer / logger 호환을 위해 유지한다.
ESCAPE는 "짧은 stimulus-driven movement bout" 상태로 해석한다.
"""

import numpy as np
import config as C
import stimulus as St
from spatial_grid import SpatialGrid

BASE, ESCAPE = 0, 1
EPS = 1e-9


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ChickenManager:
    def __init__(self, world, rng):
        self.world = world
        self.rng = rng
        self.N = C.CHICKEN_COUNT
        self.rc = C.CHICKEN_RADIUS
        self.grid = SpatialGrid(
            world.L,
            world.W,
            C.CELL_SIZE
        )

        N = self.N

        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.theta = np.zeros(N)
        self.speed = np.zeros(N)
        self.state = np.zeros(N, dtype=np.int8)

        # --------------------------------------------------------------
        # Individual responsiveness
        # --------------------------------------------------------------
        self.robot_sensitivity = rng.beta(
            C.ROBOT_SENS_ALPHA,
            C.ROBOT_SENS_BETA,
            size=N
        )
        self.social_sensitivity = rng.beta(
            C.SOCIAL_SENS_ALPHA,
            C.SOCIAL_SENS_BETA,
            size=N
        )

        # legacy/debug compatibility: "sensitivity" = robot sensitivity
        self.sensitivity = self.robot_sensitivity
        self.habituation = np.ones(N)

        # rare direct low responders
        p_low = rng.uniform(
            max(
                0.0,
                C.LOW_RESPONDER_RATE
                - C.LOW_RESPONDER_RATE_JITTER
            ),
            C.LOW_RESPONDER_RATE
            + C.LOW_RESPONDER_RATE_JITTER
        )
        n_low = int(round(N * p_low))

        self.low_responder = np.zeros(
            N,
            dtype=bool
        )

        if n_low > 0:
            idx = rng.choice(
                N,
                size=n_low,
                replace=False
            )
            self.low_responder[idx] = True
            self.robot_sensitivity[idx] = 0.0

        self.low_responder_count = int(
            self.low_responder.sum()
        )

        # --------------------------------------------------------------
        # Timers / bout bookkeeping
        # --------------------------------------------------------------
        self.escape_distance_left = np.zeros(N)
        self.escape_timer = np.zeros(N)
        self.reaction_rest_timer = np.zeros(N)
        self.base_move_timer = np.zeros(N)
        self.burst_timer = np.zeros(N)
        self.base_target_theta = rng.uniform(
            -np.pi,
            np.pi,
            size=N
        )

        # --------------------------------------------------------------
        # Robot exposure / history
        # --------------------------------------------------------------
        self.exposure_count = np.zeros(N)
        self.was_exposed = np.zeros(
            N,
            dtype=bool
        )
        self.prev_robot_threat = np.zeros(N)
        self.prev_robot_pos = None

        # diagnostic first response
        self.first_response_clearance = np.full(
            N,
            np.nan
        )
        self.first_direct_response_clearance = np.full(
            N,
            np.nan
        )

        # --------------------------------------------------------------
        # Neighbor / recursive flow state
        # --------------------------------------------------------------
        self.nn_dist = np.full(N, np.inf)
        self.nn_away = np.zeros((N, 2))

        # velocity generated during previous completed step
        self.effective_velocity = np.zeros((N, 2))
        self.push_velocity = np.zeros((N, 2))

        # current raw neighbor flow + short memory
        self.local_flow = np.zeros((N, 2))
        self.flow_memory = np.zeros((N, 2))
        self.flow_speed = np.zeros(N)

        # Finite-range behavioral social wave.
        # Direct robot responses are path=0 sources. Socially recruited birds
        # inherit min(source_path + pair_distance). inf means no active wave path.
        self.social_path = np.full(N, np.inf)
        self.social_path_candidate = np.full(N, np.inf)
        self.social_prop_gain = np.zeros(N)

        # Passive crowd relaxation: close-neighbor compression and its
        # low-density direction. This is independent of social sensitivity.
        self.crowd_pressure = np.zeros(N)
        self.crowd_vec = np.zeros((N, 2))

        # Robot-relative threat-front gate. Only birds that are not already
        # behind the advancing robot can seed/receive new behavioral social flow.
        # Physical contact displacement itself remains ungated.
        self.social_propagation_active = np.zeros(N, dtype=bool)
        self.last_robot_forward = np.array([1.0, 0.0], dtype=float)

        # compatibility with older renderer/logger/debug tools
        self.C_i = np.zeros(N)
        self.eC = np.zeros((N, 2))
        self.wall_flow_sign = rng.choice(
            [-1.0, 1.0],
            size=N
        )

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------
        self.last_burst_count = 0
        self.last_near_wall_ratio = 0.0
        self.last_reaction_rest_ratio = 0.0
        self.last_bout_end_count = 0

        self.last_direct_trigger_count = 0
        self.last_social_trigger_count = 0
        self.last_robot_cue_mean = 0.0
        self.last_robot_cue_max = 0.0
        self.last_flow_cue_mean = 0.0
        self.last_flow_cue_max = 0.0
        self.last_social_fraction_mean = 0.0
        self.last_social_fraction_max = 0.0
        self.last_social_drive_max = 0.0
        self.last_local_flow_speed_max = 0.0
        self.last_push_present_ratio = 0.0
        self.last_flow_present_ratio = 0.0
        self.last_social_follow_ratio = 0.0
        self.last_front_gate_ratio = 0.0
        self.last_rear_social_follow_ratio = 0.0
        self.last_reverse_flow_ratio = 0.0
        self.last_social_prop_gain_mean = 0.0
        self.last_social_prop_gain_max = 0.0
        self.last_social_path_max = 0.0
        self.last_crowd_pressure_mean = 0.0
        self.last_crowd_pressure_max = 0.0
        self.last_crowd_relax_ratio = 0.0
        self.last_D_raw_max = 0.0
        self.last_D_eff_max = 0.0

        self.last_social_push_ratio = 0.0
        self.last_social_push_mean = 0.0
        self.last_social_push_max = 0.0

        # old diagnostic names kept as aliases
        self.last_crowd_active_ratio = 0.0
        self.last_crowd_move_ratio = 0.0
        self.last_pressure_delta_mean = 0.0
        self.last_pressure_delta_max = 0.0
        self.last_push_delta_mean = 0.0
        self.last_push_delta_max = 0.0
        self.last_v_crowd_mean = 0.0

        self.initialize()

    # ------------------------------------------------------------------
    # Initial placement
    # ------------------------------------------------------------------
    def initialize(self):
        rng = self.rng
        rc = self.rc
        L = self.world.L
        W = self.world.W

        min_cc = 2.0 * rc
        min_cf = rc + C.FEEDER_RADIUS

        hcell = min_cc
        buckets = {}

        xs = np.empty(self.N)
        ys = np.empty(self.N)

        placed = 0
        attempts = 0
        max_attempts = self.N * 200

        pans = self.world.feeder_pans

        while (
            placed < self.N
            and attempts < max_attempts
        ):
            attempts += 1

            px = rng.uniform(
                rc,
                L - rc
            )
            py = rng.uniform(
                rc,
                W - rc
            )

            if np.any(
                (px - pans[:, 0]) ** 2
                + (py - pans[:, 1]) ** 2
                < min_cf ** 2
            ):
                continue

            cx = int(px / hcell)
            cy = int(py / hcell)

            ok = True

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = buckets.get(
                        (cx + dx, cy + dy)
                    )
                    if not b:
                        continue

                    for qx, qy in b:
                        if (
                            (px - qx) ** 2
                            + (py - qy) ** 2
                            < min_cc ** 2
                        ):
                            ok = False
                            break

                    if not ok:
                        break

                if not ok:
                    break

            if not ok:
                continue

            xs[placed] = px
            ys[placed] = py

            buckets.setdefault(
                (cx, cy),
                []
            ).append(
                (px, py)
            )

            placed += 1

        if placed < self.N:
            raise RuntimeError(
                f"초기 배치 실패: {placed}/{self.N}. "
                "밀도를 낮추거나 CHICKEN_COUNT를 줄이세요."
            )

        self.x[:] = xs
        self.y[:] = ys
        self.theta[:] = rng.uniform(
            -np.pi,
            np.pi,
            size=self.N
        )
        self.speed[:] = 0.0
        self.state[:] = BASE

    # ------------------------------------------------------------------
    # Neighbors: nearest neighbor + actual effective-velocity flow
    # ------------------------------------------------------------------
    def compute_neighbor_stats(self):
        N = self.N
        pos = np.column_stack(
            [self.x, self.y]
        )

        self.grid.build(
            self.x,
            self.y
        )

        self.nn_dist.fill(np.inf)
        self.nn_away.fill(0.0)
        self.local_flow.fill(0.0)
        self.social_path_candidate.fill(np.inf)
        self.social_prop_gain.fill(0.0)
        self.crowd_pressure.fill(0.0)
        self.crowd_vec.fill(0.0)

        # v1-style social activation state statistics
        neighbor_count = np.zeros(N, dtype=np.int32)
        escape_count = np.zeros(N, dtype=np.int32)
        edir_x = np.zeros(N)
        edir_y = np.zeros(N)
        state = self.state
        theta = self.theta

        for M, Nb in self.grid.neighborhoods():
            PM = pos[M]
            PNb = pos[Nb]

            diff = (
                PM[:, None, :]
                - PNb[None, :, :]
            )

            d = np.sqrt(
                (diff ** 2).sum(axis=2)
            )

            self_eq = (
                M[:, None]
                == Nb[None, :]
            )
            d[self_eq] = np.inf

            # nearest neighbor
            jmin = np.argmin(
                d,
                axis=1
            )
            row = np.arange(
                M.shape[0]
            )
            dmin = d[
                row,
                jmin
            ]

            self.nn_dist[M] = dmin

            away = diff[
                row,
                jmin,
                :
            ]

            good = (
                np.isfinite(dmin)
                & (dmin > EPS)
            )

            if np.any(good):
                self.nn_away[
                    M[good]
                ] = (
                    away[good]
                    / dmin[good, None]
                )

            # Previous-step disturbance velocity for social path / direction.
            nb_v = self.effective_velocity[Nb]
            nb_speed = np.hypot(nb_v[:, 0], nb_v[:, 1])

            # ----------------------------------------------------------
            # Finite-range social path propagation.
            # A source can be an ESCAPE bird or a BASE bird that is still
            # carrying behavioral social flow. Ordinary BASE wandering and
            # crowd-relaxation-only movement are never written into
            # effective_velocity, so they cannot extend this path.
            # ----------------------------------------------------------
            source_path_valid = (
                np.isfinite(self.social_path[Nb])
                & (self.social_path[Nb] < C.SOCIAL_PROP_RANGE)
                & self.social_propagation_active[Nb]
            )
            behavioral_source = source_path_valid & (
                (state[Nb] == ESCAPE)
                | (nb_speed > 1e-4)
            )

            within_path = d < max(C.R_C, C.R_FLOW)
            candidate = np.where(
                within_path & behavioral_source[None, :],
                d + self.social_path[Nb][None, :],
                np.inf
            )
            self.social_path_candidate[M] = candidate.min(axis=1)

            # v1-style behavioral social activation. ESCAPE state raises
            # reaction probability, but only while its inherited social path
            # remains inside the finite propagation range.
            within_c = d < C.R_C
            neighbor_count[M] = within_c.sum(axis=1)
            social_source = (
                (state[Nb] == ESCAPE)
                & source_path_valid
            )
            esc = within_c & social_source[None, :]
            escape_count[M] = esc.sum(axis=1)
            escf = esc.astype(np.float64)
            edir_x[M] = escf @ np.cos(theta[Nb])
            edir_y[M] = escf @ np.sin(theta[Nb])

            # ----------------------------------------------------------
            # Recursive directional crowd flow. Stationary neighbors do not
            # dilute a moving source; however, a source beyond the finite
            # behavioral path range cannot extend the threat wave.
            # ----------------------------------------------------------
            mask = d < C.R_FLOW
            w = np.where(mask, 1.0 - d / C.R_FLOW, 0.0)

            moving_nb = (nb_speed > 1e-4) & source_path_valid

            if np.any(moving_nb):
                unit_v = np.zeros_like(nb_v)
                unit_v[moving_nb, 0] = nb_v[moving_nb, 0] / nb_speed[moving_nb]
                unit_v[moving_nb, 1] = nb_v[moving_nb, 1] / nb_speed[moving_nb]

                activity = np.clip(nb_speed / C.V_WALK, 0.0, 1.0)
                wm = w * moving_nb[None, :]
                wa = w * activity[None, :]

                moving_wsum = wm.sum(axis=1)
                activity_wsum = wa.sum(axis=1)
                has_flow_neighbors = moving_wsum > EPS

                if np.any(has_flow_neighbors):
                    mean_activity = np.zeros(M.shape[0])
                    mean_activity[has_flow_neighbors] = (
                        activity_wsum[has_flow_neighbors]
                        / moving_wsum[has_flow_neighbors]
                    )

                    vx_dir = wa @ unit_v[:, 0]
                    vy_dir = wa @ unit_v[:, 1]
                    vec_norm = np.hypot(vx_dir, vy_dir)

                    coherence = np.zeros(M.shape[0])
                    active_vec = activity_wsum > EPS
                    coherence[active_vec] = np.clip(
                        vec_norm[active_vec] / activity_wsum[active_vec],
                        0.0,
                        1.0
                    )

                    q_local = np.clip(mean_activity * coherence, 0.0, 1.0)
                    good_dir = vec_norm > EPS

                    Mf = M[good_dir]
                    self.local_flow[Mf, 0] = (
                        C.V_WALK * q_local[good_dir]
                        * vx_dir[good_dir] / vec_norm[good_dir]
                    )
                    self.local_flow[Mf, 1] = (
                        C.V_WALK * q_local[good_dir]
                        * vy_dir[good_dir] / vec_norm[good_dir]
                    )

            # ----------------------------------------------------------
            # Passive crowd pressure / low-density direction.
            # This is geometry-only and independent of fear/social sensitivity.
            # ----------------------------------------------------------
            pressure_mask = d < C.R_PRESSURE
            wp = np.where(
                pressure_mask,
                1.0 - d / C.R_PRESSURE,
                0.0
            )
            dsafe = np.where(d > EPS, d, 1.0)
            pux = diff[:, :, 0] / dsafe
            puy = diff[:, :, 1] / dsafe
            px = (wp * pux).sum(axis=1)
            py = (wp * puy).sum(axis=1)
            self.crowd_pressure[M] = wp.sum(axis=1)
            pn = np.hypot(px, py)
            pg = pn > EPS
            if np.any(pg):
                Mp = M[pg]
                self.crowd_vec[Mp, 0] = px[pg] / pn[pg]
                self.crowd_vec[Mp, 1] = py[pg] / pn[pg]

        # v1 social activation fraction and ESCAPE-heading fallback direction.
        self.C_i[:] = St.social_stimulus(escape_count, neighbor_count)
        en = np.hypot(edir_x, edir_y)
        safe = en > EPS
        self.eC[:, 0] = np.where(safe, edir_x / np.where(safe, en, 1.0), 0.0)
        self.eC[:, 1] = np.where(safe, edir_y / np.where(safe, en, 1.0), 0.0)

        # Receiver-side attenuation from accumulated propagation path.
        finite_path = np.isfinite(self.social_path_candidate)
        self.social_prop_gain[finite_path] = np.clip(
            1.0
            - self.social_path_candidate[finite_path] / C.SOCIAL_PROP_RANGE,
            0.0,
            1.0
        )

    # ------------------------------------------------------------------
    # Ordinary BASE locomotion
    # ------------------------------------------------------------------
    def _base_desired_heading(
        self,
        dt,
        reaction_rest
    ):
        rng = self.rng

        is_base = (
            self.state == BASE
        )

        resting = (
            self.base_move_timer
            <= 0.0
        )

        p_start = (
            1.0
            - np.exp(
                -C.LAMBDA_BASE_MOVE
                * dt
            )
        )

        start = (
            is_base
            & resting
            & (~reaction_rest)
            & (
                rng.random(self.N)
                < p_start
            )
        )

        n_start = int(
            start.sum()
        )

        if n_start:
            self.base_move_timer[
                start
            ] = rng.uniform(
                C.BOUT_MIN,
                C.BOUT_MAX,
                size=n_start
            )

            self.base_target_theta[
                start
            ] = rng.uniform(
                -np.pi,
                np.pi,
                size=n_start
            )

        in_bout = (
            is_base
            & (
                self.base_move_timer
                > 0.0
            )
        )

        desired = self.theta.copy()
        desired[in_bout] = (
            self.base_target_theta[
                in_bout
            ]
        )

        too_close = (
            in_bout
            & (
                self.nn_dist
                < C.R_MIN
            )
        )

        isolated = (
            in_bout
            & (
                self.nn_dist
                > C.R_SOCIAL
            )
            & np.isfinite(
                self.nn_dist
            )
        )

        away_ang = np.arctan2(
            self.nn_away[:, 1],
            self.nn_away[:, 0]
        )

        desired[
            too_close
        ] = away_ang[
            too_close
        ]

        desired[
            isolated
        ] = wrap_to_pi(
            away_ang[
                isolated
            ]
            + np.pi
        )

        return desired, in_bout

    # ------------------------------------------------------------------
    # Robot incidence geometry:
    # radial-away + robot tangential component.
    # ------------------------------------------------------------------
    def _robot_geometry_direction(
        self,
        robot_pos,
        robot_vel
    ):
        geo = np.zeros(
            (self.N, 2)
        )
        radial = np.zeros(
            (self.N, 2)
        )

        if robot_pos is None:
            return geo, radial

        rx = self.x - robot_pos[0]
        ry = self.y - robot_pos[1]

        rnorm = (
            np.hypot(rx, ry)
            + EPS
        )

        radial[:, 0] = rx / rnorm
        radial[:, 1] = ry / rnorm

        rv = np.hypot(
            robot_vel[0],
            robot_vel[1]
        )

        if rv <= EPS:
            geo[:] = radial
            return geo, radial

        t_rx = robot_vel[0] / rv
        t_ry = robot_vel[1] / rv

        # Collision-angle seed: use the angle bisector between robot travel
        # and radial escape. Compared with the previous radial+tangential form,
        # this preserves substantially more lateral incidence-angle information
        # while retaining a forward component. Behind the robot direct threat is
        # already zero because closing_speed is zero.
        gx = t_rx + radial[:, 0]
        gy = t_ry + radial[:, 1]

        gnorm = np.hypot(
            gx,
            gy
        )

        valid = (
            gnorm > EPS
        )

        geo[valid, 0] = (
            gx[valid]
            / gnorm[valid]
        )
        geo[valid, 1] = (
            gy[valid]
            / gnorm[valid]
        )

        geo[
            ~valid
        ] = radial[
            ~valid
        ]

        return geo, radial

    # ------------------------------------------------------------------
    # Threat-frame social direction.
    # - reverse (robot-opposite) social flow is removed
    # - lateral flow is only allowed outward from the robot path
    # - forward component is at least as large as lateral component, so the
    #   propagated crowd wave cannot twist into a sideways/backward vortex
    # This uses geometry only; no extra gain parameter is introduced.
    # ------------------------------------------------------------------
    def _threat_align_social_direction(
        self,
        raw_dir,
        robot_pos,
        robot_vel,
        active_mask
    ):
        out = np.zeros_like(raw_dir)

        if robot_pos is None:
            return out

        rv = np.hypot(robot_vel[0], robot_vel[1])
        if rv <= EPS:
            return out

        tx = robot_vel[0] / rv
        ty = robot_vel[1] / rv
        nx = -ty
        ny = tx

        # Longitudinal and lateral components in the robot travel frame.
        parallel = raw_dir[:, 0] * tx + raw_dir[:, 1] * ty
        lateral = raw_dir[:, 0] * nx + raw_dir[:, 1] * ny

        relx = self.x - robot_pos[0]
        rely = self.y - robot_pos[1]
        cross_track = relx * nx + rely * ny

        # Above/below the robot path, inward lateral flow is discarded.
        # Near the centerline there is no preferred side, so the raw sign is kept.
        lateral_out = lateral.copy()
        off_center = np.abs(cross_track) > self.rc
        side = np.sign(cross_track[off_center])
        lateral_out[off_center] = (
            side
            * np.maximum(0.0, side * lateral[off_center])
        )

        # No backward behavioral propagation. Preserve the collision-angle
        # lateral component, but keep forward advection dominant (<=45 deg).
        forward = np.maximum(
            np.abs(parallel),
            np.abs(lateral_out)
        )

        gx = forward * tx + lateral_out * nx
        gy = forward * ty + lateral_out * ny
        gn = np.hypot(gx, gy)

        good = active_mask & (gn > EPS)
        out[good, 0] = gx[good] / gn[good]
        out[good, 1] = gy[good] / gn[good]

        return out

    # ------------------------------------------------------------------
    # Active desired vector wall projection.
    # Outward normal component is removed; tangent remains.
    # ------------------------------------------------------------------
    def _apply_wall_flow(
        self,
        vx,
        vy
    ):
        vx = vx.copy()
        vy = vy.copy()

        rc = self.rc

        d_left = self.x - rc
        d_right = (
            self.world.L - rc
        ) - self.x
        d_bottom = self.y - rc
        d_top = (
            self.world.W - rc
        ) - self.y

        D = np.stack(
            [
                d_left,
                d_right,
                d_bottom,
                d_top
            ],
            axis=1
        )

        wall_id = np.argmin(
            D,
            axis=1
        )
        wall_d = np.min(
            D,
            axis=1
        )

        near = (
            wall_d
            < C.R_WALL_FLOW
        )

        if not np.any(near):
            return vx, vy, near

        # inward normals: left/right/bottom/top
        normals = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ])

        nx = normals[
            wall_id,
            0
        ]
        ny = normals[
            wall_id,
            1
        ]

        dot = (
            vx * nx
            + vy * ny
        )

        outward = (
            near
            & (
                dot < 0.0
            )
        )

        vx[
            outward
        ] -= (
            dot[outward]
            * nx[outward]
        )

        vy[
            outward
        ] -= (
            dot[outward]
            * ny[outward]
        )

        # If wall projection removes the forward component, choose the
        # tangent that leads toward lower local density first. The tangential
        # component of crowd_vec already encodes left/right density imbalance.
        # If density is symmetric, preserve local behavioral flow; persistent
        # sign is only a final tie-breaker.
        norm = np.hypot(vx, vy)
        stuck = near & (norm < 1e-8)

        if np.any(stuck):
            tx = -ny
            ty = nx

            crowd_dot = (
                self.crowd_vec[:, 0] * tx
                + self.crowd_vec[:, 1] * ty
            )
            flow_dot = (
                self.flow_memory[:, 0] * tx
                + self.flow_memory[:, 1] * ty
            )

            sign = np.where(
                np.abs(crowd_dot) > EPS,
                np.sign(crowd_dot),
                np.where(
                    np.abs(flow_dot) > EPS,
                    np.sign(flow_dot),
                    self.wall_flow_sign
                )
            )

            vx[stuck] = sign[stuck] * tx[stuck]
            vy[stuck] = sign[stuck] * ty[stuck]

        return vx, vy, near

    # ------------------------------------------------------------------
    # Contact displacement wall projection.
    # Tangent component is preserved automatically by endpoint clipping.
    # If a corner removes almost all displacement, choose an allowed
    # cardinal direction closest to local flow/current heading.
    # ------------------------------------------------------------------
    def _project_contact_displacement(
        self,
        dx,
        dy
    ):
        orig_mag = np.hypot(
            dx,
            dy
        )

        x_new = np.clip(
            self.x + dx,
            self.rc,
            self.world.L - self.rc
        )
        y_new = np.clip(
            self.y + dy,
            self.rc,
            self.world.W - self.rc
        )

        pdx = x_new - self.x
        pdy = y_new - self.y

        remain = np.hypot(
            pdx,
            pdy
        )

        stuck = (
            (orig_mag > EPS)
            & (
                remain
                < 0.10 * orig_mag
            )
        )

        if not np.any(stuck):
            return pdx, pdy

        guide = self.flow_memory.copy()
        guide_norm = np.hypot(
            guide[:, 0],
            guide[:, 1]
        )

        no_guide = (
            guide_norm <= EPS
        )

        guide[
            no_guide,
            0
        ] = np.cos(
            self.theta[no_guide]
        )
        guide[
            no_guide,
            1
        ] = np.sin(
            self.theta[no_guide]
        )

        # Candidate directions: +x, -x, +y, -y.
        dirs = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ])

        for i in np.flatnonzero(
            stuck
        ):
            mag = orig_mag[i]

            best_score = -np.inf
            best_dx = 0.0
            best_dy = 0.0

            for ux, uy in dirs:
                tx = self.x[i] + ux * mag
                ty = self.y[i] + uy * mag

                # candidate must stay inside world
                if (
                    tx < self.rc
                    or tx > self.world.L - self.rc
                    or ty < self.rc
                    or ty > self.world.W - self.rc
                ):
                    continue

                score = (
                    guide[i, 0] * ux
                    + guide[i, 1] * uy
                )

                if score > best_score:
                    best_score = score
                    best_dx = ux * mag
                    best_dy = uy * mag

            pdx[i] = best_dx
            pdy[i] = best_dy

        return pdx, pdy

    # ------------------------------------------------------------------
    # v1-style actual-overlap correction.
    # No social gain, no soft shell, no sensitivity.
    # Returns accumulated chicken-contact displacement.
    # ------------------------------------------------------------------
    def resolve_contact_push(
        self,
        iterations=None
    ):
        if iterations is None:
            iterations = C.CONTACT_ITERS

        total_dx = np.zeros(
            self.N
        )
        total_dy = np.zeros(
            self.N
        )

        two_rc = (
            2.0 * self.rc
        )

        for _ in range(
            iterations
        ):
            self.grid.build(
                self.x,
                self.y
            )

            pos = np.column_stack(
                [self.x, self.y]
            )

            dispx = np.zeros(
                self.N
            )
            dispy = np.zeros(
                self.N
            )

            for M, Nb in self.grid.neighborhoods():
                PM = pos[M]
                PNb = pos[Nb]

                diff = (
                    PM[:, None, :]
                    - PNb[None, :, :]
                )

                d = np.sqrt(
                    (diff ** 2).sum(axis=2)
                )

                self_eq = (
                    M[:, None]
                    == Nb[None, :]
                )

                d[
                    self_eq
                ] = np.inf

                overlap = (
                    d < two_rc
                )

                if not np.any(
                    overlap
                ):
                    continue

                delta = np.where(
                    overlap,
                    two_rc - d,
                    0.0
                )

                dsafe = np.where(
                    d > EPS,
                    d,
                    1.0
                )

                ux = (
                    diff[:, :, 0]
                    / dsafe
                )
                uy = (
                    diff[:, :, 1]
                    / dsafe
                )

                zero = (
                    (d < EPS)
                    & overlap
                )

                if np.any(zero):
                    ux = np.where(
                        zero,
                        np.cos(
                            self.theta[M]
                        )[:, None],
                        ux
                    )
                    uy = np.where(
                        zero,
                        np.sin(
                            self.theta[M]
                        )[:, None],
                        uy
                    )

                dispx[M] += (
                    0.5
                    * delta
                    * ux
                ).sum(axis=1)

                dispy[M] += (
                    0.5
                    * delta
                    * uy
                ).sum(axis=1)

            mag = np.hypot(
                dispx,
                dispy
            )

            too_big = (
                mag
                > C.MAX_OVERLAP_PUSH
            )

            if np.any(
                too_big
            ):
                scale = (
                    C.MAX_OVERLAP_PUSH
                    / mag[too_big]
                )

                dispx[
                    too_big
                ] *= scale
                dispy[
                    too_big
                ] *= scale

            dispx, dispy = (
                self._project_contact_displacement(
                    dispx,
                    dispy
                )
            )

            self.x += dispx
            self.y += dispy

            total_dx += dispx
            total_dy += dispy

            # obstacle corrections can create a new chicken overlap,
            # so they stay inside the contact iteration.
            self.world.resolve_feeder_collision(
                self.x,
                self.y
            )
            self.world.resolve_wall_collision(
                self.x,
                self.y
            )

        return total_dx, total_dy

    # ------------------------------------------------------------------
    # One timestep
    # ------------------------------------------------------------------
    def step(
        self,
        robot_pos,
        dt
    ):
        N = self.N
        rng = self.rng

        # 1) Robot kinematics first: social propagation is defined in the
        # advancing robot-threat frame, not as an isotropic flock field.
        if robot_pos is not None:
            robot_pos = np.asarray(robot_pos, dtype=float)
            if self.prev_robot_pos is None:
                robot_vel = np.array(C.ROBOT_VEL, dtype=float)
            else:
                robot_vel = (
                    robot_pos - self.prev_robot_pos
                ) / max(dt, EPS)
        else:
            robot_vel = np.zeros(2)

        robot_speed = np.hypot(robot_vel[0], robot_vel[1])
        robot_frame_active = (
            robot_pos is not None
            and robot_speed > EPS
        )

        if robot_frame_active:
            tx = robot_vel[0] / robot_speed
            ty = robot_vel[1] / robot_speed
            self.last_robot_forward[:] = (tx, ty)

            relx = self.x - robot_pos[0]
            rely = self.y - robot_pos[1]
            longitudinal = relx * tx + rely * ty

            # No extra behavioral distance parameter: use the actual robot +
            # chicken geometry as the rear cutoff. Once the robot center has
            # passed a bird by more than the contact-scale geometry, that bird
            # cannot start/seed new behavioral social propagation.
            rear_cutoff = C.ROBOT_RADIUS + C.CHICKEN_RADIUS
            self.social_propagation_active[:] = (
                longitudinal >= -rear_cutoff
            )
            # A behavioral wave cannot remain attached to birds after the
            # advancing threat front has passed them. Passive crowd relaxation
            # is intentionally NOT cleared here.
            self.social_path[~self.social_propagation_active] = np.inf
        else:
            self.social_propagation_active.fill(False)
            self.social_path.fill(np.inf)

        # 2) Read previous-step disturbance velocity into the LOCAL field.
        # Sources already behind the robot are excluded in compute_neighbor_stats.
        self.compute_neighbor_stats()

        a = np.exp(-dt / max(C.TAU_FLOW, EPS))
        self.flow_memory[:] = (
            a * self.flow_memory
            + (1.0 - a) * self.local_flow
        )

        # Rear wake is behavioral memory, not physical momentum: clear it once
        # the threat front has passed. Existing active bouts may still decelerate
        # naturally, but cannot keep dragging the rear flock.
        self.flow_memory[~self.social_propagation_active] = 0.0

        q_flow, flow_dir, flow_speed = St.flow_cue(self.flow_memory)
        self.flow_speed[:] = flow_speed

        # 3) Robot geometry / closing speed / threat.
        if robot_pos is not None:
            dxr = self.x - robot_pos[0]
            dyr = self.y - robot_pos[1]

            center_d = np.hypot(
                dxr,
                dyr
            )

            clearance = (
                center_d
                - (
                    C.ROBOT_RADIUS
                    + C.CHICKEN_RADIUS
                )
            )

            geo_dir, radial = (
                self._robot_geometry_direction(
                    robot_pos,
                    robot_vel
                )
            )

            # Previous completed-step effective chicken velocity.
            rel_vx = (
                robot_vel[0]
                - self.effective_velocity[:, 0]
            )
            rel_vy = (
                robot_vel[1]
                - self.effective_velocity[:, 1]
            )

            closing_speed = np.maximum(
                0.0,
                radial[:, 0] * rel_vx
                + radial[:, 1] * rel_vy
            )
        else:
            clearance = np.full(
                N,
                np.inf
            )
            closing_speed = np.zeros(N)
            geo_dir = np.zeros(
                (N, 2)
            )

        (
            q_robot,
            q_distance,
            q_closing
        ) = St.robot_threat(
            clearance,
            closing_speed
        )

        # 4) Habituation is robot-only.
        now_exposed = (
            q_distance
            > C.D_EXPOSURE_TH
        )

        rising = (
            now_exposed
            & (~self.was_exposed)
        )

        self.exposure_count += rising
        self.was_exposed = now_exposed

        if C.HABITUATION_ON:
            self.habituation[:] = (
                C.H_MIN
                + (
                    1.0
                    - C.H_MIN
                )
                * np.exp(
                    -C.K_H
                    * self.exposure_count
                )
            )
        else:
            self.habituation.fill(1.0)

        # 5) Unified probabilistic movement-bout start.
        # Social activation is allowed only in the advancing threat-front.
        social_fraction_active = (
            self.C_i
            * self.social_prop_gain
            * self.social_propagation_active.astype(float)
        )

        (
            p_start,
            drive,
            drive_robot,
            drive_social,
            hazard_robot,
            hazard_social
        ) = St.response_probability(
            self.robot_sensitivity,
            self.social_sensitivity,
            self.habituation,
            q_robot,
            social_fraction_active,
            dt
        )

        is_base = (
            self.state == BASE
        )

        reaction_rest = (
            self.reaction_rest_timer
            > 0.0
        )

        can_start = (
            is_base
            & (~reaction_rest)
        )

        start = (
            can_start
            & (
                rng.random(N)
                < p_start
            )
        )

        # diagnostic source classification only.
        direct_start = (
            start
            & (
                hazard_robot
                >= hazard_social
            )
            & (
                drive_robot
                > 0.0
            )
        )

        social_start = (
            start
            & (~direct_start)
        )

        # Direct responses seed a new path at zero. Socially recruited birds
        # inherit the shortest accumulated path from the previous local wave.
        if np.any(direct_start):
            self.social_path[direct_start] = 0.0
        if np.any(social_start):
            inherited = self.social_path_candidate[social_start]
            self.social_path[social_start] = np.where(
                np.isfinite(inherited),
                inherited,
                np.inf
            )

        if np.any(direct_start):
            first_direct = (
                direct_start
                & np.isnan(
                    self.first_direct_response_clearance
                )
            )
            self.first_direct_response_clearance[
                first_direct
            ] = clearance[
                first_direct
            ]

        if np.any(start):
            n_start = int(
                start.sum()
            )

            self.state[
                start
            ] = ESCAPE

            L = rng.normal(
                C.L_ESCAPE_MEAN,
                C.L_ESCAPE_STD,
                size=n_start
            )

            self.escape_distance_left[
                start
            ] = np.clip(
                L,
                C.L_ESCAPE_MIN,
                C.L_ESCAPE_MAX
            )

            self.escape_timer[
                start
            ] = C.T_ESCAPE_MAX

            # stimulus-driven bout overrides ordinary BASE bout.
            self.base_move_timer[
                start
            ] = 0.0

            newly = (
                start
                & np.isnan(
                    self.first_response_clearance
                )
            )

            self.first_response_clearance[
                newly
            ] = clearance[
                newly
            ]

        is_escape = (
            self.state == ESCAPE
        )

        # 6) Sudden/close burst from robot threat rise.
        qdot_plus = St.positive_rate(
            q_robot,
            self.prev_robot_threat,
            dt
        )

        start_burst = (
            is_escape
            & (
                clearance
                < C.BURST_CLEARANCE
            )
            & (
                qdot_plus
                > C.BURST_QDOT_TH
            )
            & (
                self.burst_timer
                <= 0.0
            )
        )

        if np.any(
            start_burst
        ):
            self.burst_timer[
                start_burst
            ] = C.T_BURST

        burst_active = (
            is_escape
            & (
                self.burst_timer
                > 0.0
            )
        )

        # 7) Desired direction.
        # Activation and direction are intentionally separated:
        #   C_i -> whether social response starts (v1 mechanism)
        #   effective-velocity flow -> where the response goes
        # The raw local direction is then rectified into the advancing robot
        # threat frame: forward motion + outward lateral displacement only.
        raw_social_dir = flow_dir.copy()
        no_flow_dir = flow_speed <= EPS
        raw_social_dir[no_flow_dir] = self.eC[no_flow_dir]

        social_dir = self._threat_align_social_direction(
            raw_social_dir,
            robot_pos,
            robot_vel,
            self.social_propagation_active
        )

        robot_weight = drive_robot

        # Most birds should be carried strongly by a coherent flock flow; the
        # low tail still resists. This nonlinear mapping does not add a new
        # parameter: s=0 stays 0, s=1 stays 1, mid/high sensitivity is boosted.
        flow_compliance = 1.0 - (1.0 - self.social_sensitivity) ** 2

        # A sparse but clearly escaping neighbor must be able to entrain the
        # surrounding flock before the velocity average has fully built up.
        # Use the stronger of actual coherent velocity flow and sqrt(C_i).
        # sqrt only de-compresses the v1 fraction; it introduces no new gain.
        crowd_cue = np.maximum(
            q_flow,
            np.sqrt(np.clip(self.C_i, 0.0, 1.0))
        )
        flow_follow_drive = (
            flow_compliance
            * crowd_cue
            * self.social_prop_gain
            * self.social_propagation_active.astype(float)
        )
        social_weight = np.maximum(drive_social, flow_follow_drive)

        # Passive crowd relaxation is a separate geometry layer. It can remain
        # active after the threat has passed and does not depend on sensitivity.
        crowd_strength = np.clip(
            self.crowd_pressure - C.PRESSURE_THRESHOLD,
            0.0,
            1.0
        )
        crowd_active = crowd_strength > 0.0

        ux = (
            robot_weight * geo_dir[:, 0]
            + social_weight * social_dir[:, 0]
            + crowd_strength * self.crowd_vec[:, 0]
        )
        uy = (
            robot_weight * geo_dir[:, 1]
            + social_weight * social_dir[:, 1]
            + crowd_strength * self.crowd_vec[:, 1]
        )

        ux, uy, near_wall = (
            self._apply_wall_flow(
                ux,
                uy
            )
        )

        move_norm = np.hypot(
            ux,
            uy
        )

        esc_desired = (
            self.theta.copy()
        )

        valid = (
            move_norm > EPS
        )

        esc_desired[
            valid
        ] = np.arctan2(
            uy[valid],
            ux[valid]
        )

        # 8) Ordinary BASE heading.
        base_desired, in_base_bout = (
            self._base_desired_heading(
                dt,
                reaction_rest
            )
        )

        # BASE collective movement has two distinct causes:
        # 1) behavioral social follow (finite-path, sensitivity-dependent),
        # 2) passive crowd relaxation (geometry-only, can persist after threat).
        social_dir_norm = np.hypot(social_dir[:, 0], social_dir[:, 1])
        social_follow_base = (
            (self.state == BASE)
            & (~reaction_rest)
            & (flow_follow_drive > 1e-4)
            & (social_dir_norm > EPS)
        )
        crowd_move_base = (
            (self.state == BASE)
            & crowd_active
        )

        base_cx = (
            flow_follow_drive * social_dir[:, 0]
            + crowd_strength * self.crowd_vec[:, 0]
        )
        base_cy = (
            flow_follow_drive * social_dir[:, 1]
            + crowd_strength * self.crowd_vec[:, 1]
        )
        base_cx, base_cy, _ = self._apply_wall_flow(base_cx, base_cy)
        base_cn = np.hypot(base_cx, base_cy)
        base_collective = (
            (self.state == BASE)
            & (base_cn > EPS)
            & (social_follow_base | crowd_move_base)
        )
        if np.any(base_collective):
            base_desired[base_collective] = np.arctan2(
                base_cy[base_collective],
                base_cx[base_collective]
            )

        # A BASE behavioral follower inherits the local finite wave path and can
        # pass directional flow onward. crowd-only movers deliberately do not.
        if np.any(social_follow_base):
            inherited = self.social_path_candidate[social_follow_base]
            good_inherit = np.isfinite(inherited)
            idx_follow = np.flatnonzero(social_follow_base)
            self.social_path[idx_follow[good_inherit]] = inherited[good_inherit]

        desired_theta = np.where(
            is_escape,
            esc_desired,
            base_desired
        )

        # Once the robot has clearly passed a bird and direct threat is receding,
        # terminate the behavioral escape drive. The bird decelerates with the
        # normal stop time constant instead of continuing a rearward wake.
        receding_escape = (
            is_escape
            & (~self.social_propagation_active)
            & (q_robot <= EPS)
        )

        # 9) Target speed.
        v_target = np.where(
            in_base_bout,
            C.V_BASE,
            0.0
        )

        # Crowd-flow following is controlled by actual coherent movement, not
        # by the diluted ESCAPE fraction. Individual social sensitivity keeps a
        # small low-response tail while most birds are carried by the flow.
        v_social_follow = C.V_WALK * (
            np.clip(flow_follow_drive, 0.0, 1.0) ** 0.75
        )
        v_crowd = C.V_CROWD_MAX * crowd_strength
        v_target[social_follow_base] = np.maximum(
            v_target[social_follow_base],
            v_social_follow[social_follow_base]
        )
        v_target[crowd_move_base] = np.maximum(
            v_target[crowd_move_base],
            v_crowd[crowd_move_base]
        )

        tau = np.full(
            N,
            C.TAU_BASE
        )
        tau[social_follow_base] = C.TAU_WALK

        v_target = np.where(
            is_escape,
            np.where(
                burst_active,
                C.V_BURST,
                C.V_WALK
            ),
            v_target
        )

        tau = np.where(
            is_escape,
            np.where(
                burst_active,
                C.TAU_BURST,
                C.TAU_WALK
            ),
            tau
        )

        # refractory BASE chickens actively settle to stop
        v_target[
            reaction_rest
        ] = 0.0
        tau[
            reaction_rest
        ] = C.TAU_STOP

        # Reaction rest blocks active re-escape/social following, but does not
        # freeze a physically compressed BASE bird. Crowd relaxation remains.
        rest_crowd = reaction_rest & crowd_move_base
        v_target[rest_crowd] = np.maximum(
            v_target[rest_crowd],
            v_crowd[rest_crowd]
        )
        tau[rest_crowd] = C.TAU_BASE

        # Threat has passed: stop ESCAPE propulsion immediately and let speed
        # decay smoothly. State transition to BASE happens at bout-end handling.
        v_target[receding_escape] = 0.0
        tau[receding_escape] = C.TAU_STOP

        # Former social followers behind the robot should not coast for the much
        # slower TAU_BASE. If they are still above ordinary BASE speed, settle.
        rear_flow_coast = (
            (self.state == BASE)
            & (~self.social_propagation_active)
            & (~crowd_move_base)
            & (self.speed > C.V_BASE)
        )
        v_target[rear_flow_coast] = 0.0
        tau[rear_flow_coast] = C.TAU_STOP

        # 10) Heading dynamics.
        turn_active = (
            is_escape
            | in_base_bout
            | social_follow_base
            | crowd_move_base
        )

        dtheta = wrap_to_pi(
            desired_theta
            - self.theta
        )

        g_theta = np.where(
            turn_active,
            np.maximum(
                0.0,
                np.cos(dtheta)
            ),
            1.0
        )

        if np.any(
            turn_active
        ):
            omega = np.clip(
                C.K_THETA
                * dtheta[
                    turn_active
                ],
                -C.OMEGA_MAX,
                C.OMEGA_MAX
            )

            self.theta[
                turn_active
            ] = wrap_to_pi(
                self.theta[
                    turn_active
                ]
                + omega
                * dt
            )

        # 11) Active locomotion.
        v_star = (
            v_target
            * g_theta
        )

        self.speed += (
            (
                v_star
                - self.speed
            )
            * (
                1.0
                - np.exp(
                    -dt / tau
                )
            )
        )

        active_vx = (
            self.speed
            * np.cos(
                self.theta
            )
        )
        active_vy = (
            self.speed
            * np.sin(
                self.theta
            )
        )

        active_step_distance = (
            self.speed
            * dt
        )

        self.x += (
            active_vx
            * dt
        )
        self.y += (
            active_vy
            * dt
        )

        # hard obstacle clamp before chicken contact
        self.world.resolve_feeder_collision(
            self.x,
            self.y
        )
        self.world.resolve_wall_collision(
            self.x,
            self.y
        )

        # 12) v1-style physical overlap push.
        push_dx, push_dy = (
            self.resolve_contact_push(
                iterations=C.CONTACT_ITERS
            )
        )

        self.push_velocity[:, 0] = (
            push_dx
            / max(dt, EPS)
        )
        self.push_velocity[:, 1] = (
            push_dy
            / max(dt, EPS)
        )

        # 13) Recursive disturbance-flow source for next step.
        # Ordinary BASE wandering is not a social source. Existing birds behind
        # the robot may still physically move/finish a bout, but their movement
        # is not allowed to generate a new behavioral rear wake.
        propagation_gate = self.social_propagation_active.astype(float)
        disturbance_active = (
            (is_escape | social_follow_base).astype(float)
            * propagation_gate
        )

        self.effective_velocity[:, 0] = (
            disturbance_active * active_vx
            + propagation_gate * self.push_velocity[:, 0]
        )
        self.effective_velocity[:, 1] = (
            disturbance_active * active_vy
            + propagation_gate * self.push_velocity[:, 1]
        )

        # 14) Distance-based stimulus bout end.
        active_escape = (
            self.state == ESCAPE
        )

        self.escape_distance_left[
            active_escape
        ] -= (
            active_step_distance[
                active_escape
            ]
        )

        self.escape_timer[
            active_escape
        ] -= dt

        finished = (
            active_escape
            & (
                (
                    self.escape_distance_left
                    <= 0.0
                )
                | (
                    self.escape_timer
                    <= 0.0
                )
                | receding_escape
            )
        )

        bout_end_count = int(
            finished.sum()
        )

        if np.any(
            finished
        ):
            self.state[
                finished
            ] = BASE

            self.reaction_rest_timer[
                finished
            ] = rng.uniform(
                C.T_REST_MIN,
                C.T_REST_MAX,
                size=bout_end_count
            )

            self.base_move_timer[
                finished
            ] = 0.0

            self.escape_distance_left[
                finished
            ] = 0.0

        # 15) Timers.
        active_base_timer = (
            self.base_move_timer
            > 0.0
        )

        self.base_move_timer[
            active_base_timer
        ] = np.maximum(
            0.0,
            self.base_move_timer[
                active_base_timer
            ]
            - dt
        )

        self.burst_timer[:] = np.maximum(
            0.0,
            self.burst_timer
            - dt
        )

        self.reaction_rest_timer[:] = np.maximum(
            0.0,
            self.reaction_rest_timer
            - dt
        )

        # 16) Histories.
        self.prev_robot_threat[:] = q_robot

        if robot_pos is None:
            self.prev_robot_pos = None
        else:
            self.prev_robot_pos = robot_pos.copy()

        # 17) Diagnostics.
        push_mag = np.hypot(
            push_dx,
            push_dy
        )

        pushed = (
            push_mag > 1e-6
        )

        self.last_burst_count = int(
            burst_active.sum()
        )
        self.last_near_wall_ratio = float(
            near_wall.mean()
        )
        self.last_reaction_rest_ratio = float(
            reaction_rest.mean()
        )
        self.last_bout_end_count = bout_end_count

        self.last_direct_trigger_count = int(
            direct_start.sum()
        )
        self.last_social_trigger_count = int(
            social_start.sum()
        )

        self.last_robot_cue_mean = float(
            q_robot.mean()
        )
        self.last_robot_cue_max = float(
            q_robot.max()
        )
        self.last_flow_cue_mean = float(
            q_flow.mean()
        )
        self.last_flow_cue_max = float(
            q_flow.max()
        )
        self.last_social_fraction_mean = float(social_fraction_active.mean())
        self.last_social_fraction_max = float(social_fraction_active.max())
        self.last_social_drive_max = float(drive_social.max())
        self.last_local_flow_speed_max = float(flow_speed.max())
        self.last_push_present_ratio = float(pushed.mean())
        self.last_flow_present_ratio = float((flow_speed > 1e-4).mean())
        self.last_social_follow_ratio = float(social_follow_base.mean())
        self.last_social_prop_gain_mean = float(self.social_prop_gain.mean())
        self.last_social_prop_gain_max = float(self.social_prop_gain.max())
        finite_social_path = self.social_path[np.isfinite(self.social_path)]
        self.last_social_path_max = float(
            finite_social_path.max() if finite_social_path.size else 0.0
        )
        self.last_crowd_pressure_mean = float(self.crowd_pressure.mean())
        self.last_crowd_pressure_max = float(self.crowd_pressure.max())
        self.last_crowd_relax_ratio = float(crowd_move_base.mean())
        self.last_front_gate_ratio = float(self.social_propagation_active.mean())
        self.last_rear_social_follow_ratio = float((
            social_follow_base & (~self.social_propagation_active)
        ).mean())
        if robot_frame_active:
            raw_parallel = (
                flow_dir[:, 0] * self.last_robot_forward[0]
                + flow_dir[:, 1] * self.last_robot_forward[1]
            )
            raw_present = (flow_speed > 1e-4) & self.social_propagation_active
            if np.any(raw_present):
                self.last_reverse_flow_ratio = float(
                    (raw_parallel[raw_present] < 0.0).mean()
                )
            else:
                self.last_reverse_flow_ratio = 0.0
        else:
            self.last_reverse_flow_ratio = 0.0
        self.last_D_raw_max = float(q_distance.max())
        self.last_D_eff_max = float(q_robot.max())
        # compatibility aliases now point to passive crowd relaxation metrics
        self.last_crowd_active_ratio = float(crowd_active.mean())
        self.last_crowd_move_ratio = float(crowd_move_base.mean())
        self.last_v_crowd_mean = float(v_crowd.mean())

        self.last_social_push_ratio = float(
            pushed.mean()
        )
        self.last_social_push_mean = float(
            push_mag.mean()
        )
        self.last_social_push_max = float(
            push_mag.max()
        )

        self.last_crowd_active_ratio = (
            self.last_social_push_ratio
        )
        self.last_crowd_move_ratio = (
            self.last_social_push_ratio
        )

        # "S" is now the unified dimensionless response drive.
        return {
            "S": drive,
            "D": q_robot,
            "D_raw": q_distance,
            "C": social_fraction_active.copy(),
            "Q_flow": q_flow,
            "H_robot": hazard_robot,
            "H_social": hazard_social
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def escape_ratio(self):
        return float(
            (
                self.state
                == ESCAPE
            ).mean()
        )

    def mean_speed(self):
        return float(
            self.speed.mean()
        )