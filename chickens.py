"""
chickens.py
===========
v0.4.4 threat re-evaluation + dynamic escape direction + contact-observed social source.

핵심 loop:
robot threat / v1-style social activation
 -> probabilistic short movement bout
 -> physical overlap displacement
 -> behavioral velocity + actual contact push
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
            # Behavioral near-nonresponders: robot/social cues are both ignored.
            # They can still be moved by actual body contact and crowd pressure.
            self.robot_sensitivity[idx] = 0.0
            self.social_sensitivity[idx] = 0.0

        self.low_responder_count = int(
            self.low_responder.sum()
        )

        # --------------------------------------------------------------
        # Timers / bout bookkeeping
        # --------------------------------------------------------------
        self.escape_distance_left = np.zeros(N)
        self.escape_distance_travelled = np.zeros(N)
        self.escape_timer = np.zeros(N)
        # Compatibility-only field. v0.4.4 no longer persists an escape target;
        # desired escape direction is recomputed every frame and the actual body
        # heading remains smooth through K_THETA / OMEGA_MAX dynamics.
        self.escape_target_theta = rng.uniform(
            -np.pi,
            np.pi,
            size=N
        )
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

        # Local flock geometry within R_SOCIAL.
        # Cohesion is only toward neighbors already inside this radius;
        # birds outside the radius are not attracted in, preventing one
        # house-wide connected flock.
        self.flock_neighbor_count = np.zeros(N, dtype=np.int32)
        self.flock_centroid_dir = np.zeros((N, 2))

        # Previous completed-step disturbance velocities.
        # behavioral_velocity: voluntary robot/social movement; finite social path applies.
        # mechanical_velocity: actual contact/crowd displacement; no 3 m behavioral path limit.
        # effective_velocity is kept as a compatibility/perception sum.
        self.behavioral_velocity = np.zeros((N, 2))
        self.mechanical_velocity = np.zeros((N, 2))
        self.effective_velocity = np.zeros((N, 2))
        self.push_velocity = np.zeros((N, 2))

        # Explicit previous-frame source flags: a recruit from this step cannot
        # recruit the next bird until its actual movement is observed next step.
        self.behavioral_source_active = np.zeros(N, dtype=bool)
        self.mechanical_source_active = np.zeros(N, dtype=bool)

        # current raw neighbor flow + short direction memory
        self.local_flow = np.zeros((N, 2))
        self.flow_memory = np.zeros((N, 2))
        self.flow_speed = np.zeros(N)

        # Legacy compatibility fields. v0.4.1 simple-social does not use
        # path/provenance attenuation; these remain inert for existing loggers.
        self.social_path = np.full(N, np.inf)
        self.social_path_candidate = np.full(N, np.inf)
        self.social_prop_gain = np.ones(N)

        # Social excitation and local density.
        # local_density is a distribution metric [birds/m^2].
        # density_signal is a bounded local occupied-area proxy used only for
        # the already-retained density-dependent social path discount.
        self.social_source_mass = np.zeros(N)
        self.social_input = np.zeros(N)
        self.social_excitation = np.zeros(N)
        self.local_density = np.zeros(N)
        self.density_signal = np.zeros(N)
        self.mobility_factor = np.ones(N)

        # End-wall relief diagnostics.
        self.wall_relief_active = np.zeros(N, dtype=bool)
        self.wall_relief_dir = np.zeros((N, 2))
        self.wall_relief_gain = np.zeros(N)

        # Mechanical compression q_i and passive relaxation direction.
        # Keep crowd_pressure as a compatibility alias; it is NOT local density.
        self.mechanical_compression = np.zeros(N)
        self.crowd_pressure = self.mechanical_compression
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
        self.last_near_feeder_ratio = 0.0
        self.last_side_pass_cue_max = 0.0
        self.last_reaction_rest_ratio = 0.0
        self.last_bout_end_count = 0
        self.last_threat_loss_end_count = 0
        self.last_perceived_threat_mean = 0.0
        self.last_perceived_threat_max = 0.0
        self.last_contact_social_source_ratio = 0.0

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
        self.last_behavioral_source_ratio = 0.0
        self.last_mechanical_source_ratio = 0.0
        self.last_social_envelope_active_ratio = 0.0
        self.last_local_flock_member_ratio = 0.0
        self.last_local_flock_neighbor_mean = 0.0
        self.last_social_prop_gain_mean = 0.0
        self.last_social_prop_gain_max = 0.0
        self.last_social_path_max = 0.0
        self.last_social_source_mass_max = 0.0
        self.last_social_excitation_max = 0.0
        self.last_local_density_mean = 0.0
        self.last_local_density_max = 0.0
        self.last_density_signal_mean = 0.0
        self.last_density_signal_max = 0.0
        self.last_mobility_mean = 1.0
        self.last_mobility_min = 1.0
        self.last_wall_relief_ratio = 0.0
        self.last_contact_iters = 0
        self.last_contact_max_penetration = 0.0
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
        self.last_v_crowd_max = 0.0

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
        """
        Simple-social neighborhood update.

        Two social ingredients only:
        1) spacing geometry (nearest-neighbor attraction/repulsion is used later
           by _base_desired_heading);
        2) previous-frame actual disturbance movement -> social excitation and
           movement-following direction.

        No C_i, social-path, density path cost, upstream/provenance graph, or
        recruited-cohort memory is used here.
        """
        N = self.N
        pos = np.column_stack([self.x, self.y])

        self.grid.build(self.x, self.y)

        self.nn_dist.fill(np.inf)
        self.nn_away.fill(0.0)
        self.flock_neighbor_count.fill(0)
        self.flock_centroid_dir.fill(0.0)
        self.local_flow.fill(0.0)

        # Compatibility fields are deliberately inert.
        self.social_path.fill(np.inf)
        self.social_path_candidate.fill(np.inf)
        self.social_prop_gain.fill(1.0)
        self.C_i.fill(0.0)
        self.eC.fill(0.0)

        self.social_source_mass.fill(0.0)
        self.social_input.fill(0.0)
        self.local_density.fill(0.0)
        self.density_signal.fill(0.0)
        self.crowd_pressure.fill(0.0)
        self.crowd_vec.fill(0.0)

        for M, Nb in self.grid.neighborhoods():
            PM = pos[M]
            PNb = pos[Nb]

            diff = PM[:, None, :] - PNb[None, :, :]
            d = np.sqrt((diff ** 2).sum(axis=2))

            self_eq = M[:, None] == Nb[None, :]
            d[self_eq] = np.inf

            # ----------------------------------------------------------
            # Spacing geometry: nearest-neighbor direction.
            # Actual attraction/repulsion is applied only when a bird is already
            # in a voluntary BASE bout; spacing does NOT create locomotion by itself.
            # ----------------------------------------------------------
            jmin = np.argmin(d, axis=1)
            row = np.arange(M.shape[0])
            dmin = d[row, jmin]
            self.nn_dist[M] = dmin

            away = diff[row, jmin, :]
            good = np.isfinite(dmin) & (dmin > EPS)
            if np.any(good):
                self.nn_away[M[good]] = away[good] / dmin[good, None]

            # ----------------------------------------------------------
            # Local flock geometry.
            #
            # Only neighbors ALREADY within R_SOCIAL contribute. A bird with no
            # neighbors in this radius receives no cohesion, so separate local
            # flocks are not pulled together across empty space.
            # ----------------------------------------------------------
            flock_mask = d < C.R_SOCIAL
            flock_count = flock_mask.sum(axis=1)
            self.flock_neighbor_count[M] = flock_count

            # diff = self - neighbor, so -diff points from self toward neighbor.
            to_cx = (-diff[:, :, 0] * flock_mask).sum(axis=1)
            to_cy = (-diff[:, :, 1] * flock_mask).sum(axis=1)
            coh_norm = np.hypot(to_cx, to_cy)
            coh_good = (flock_count > 0) & (coh_norm > EPS)
            if np.any(coh_good):
                Mc = M[coh_good]
                self.flock_centroid_dir[Mc, 0] = (
                    to_cx[coh_good] / coh_norm[coh_good]
                )
                self.flock_centroid_dir[Mc, 1] = (
                    to_cy[coh_good] / coh_norm[coh_good]
                )

            # ----------------------------------------------------------
            # Local density rho_i: distribution diagnostic only.
            # ----------------------------------------------------------
            density_mask = d < C.R_DENSITY
            density_count = density_mask.sum(axis=1)
            local_density = density_count / (np.pi * C.R_DENSITY ** 2)
            self.local_density[M] = local_density
            self.density_signal[M] = np.clip(
                local_density * np.pi * self.rc ** 2,
                0.0,
                1.0
            )

            # ----------------------------------------------------------
            # Mechanical compression q_i: independent of social excitation.
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

            # ----------------------------------------------------------
            # Movement-following excitation.
            #
            # Source = previous completed-step voluntary disturbance movement
            # plus ACTUAL body-contact displacement.
            #
            # Passive crowd relaxation is deliberately excluded:
            #   behavioral_velocity + push_velocity      -> perceived by neighbors
            #   crowd_relax component of mechanical_velocity -> NOT a social source
            #
            # This allows 'pushed bird -> nearby bird notices motion' without
            # restoring crowd-relaxation -> social -> crowd positive feedback.
            # ----------------------------------------------------------
            nb_v = (
                self.behavioral_velocity[Nb]
                + self.push_velocity[Nb]
            )
            nb_speed = np.hypot(nb_v[:, 0], nb_v[:, 1])

            # Derived numerical floor, not an independent parameter.
            source_speed_floor = 0.5 * C.V_BASE
            moving_nb = nb_speed >= source_speed_floor

            within = d < C.R_FLOW
            w = np.where(
                within,
                1.0 - d / C.R_FLOW,
                0.0
            )

            source_w = w * moving_nb[None, :]
            source_mass = source_w.sum(axis=1)
            self.social_source_mass[M] = source_mass

            # S_i = 1 - exp(-A_i)
            self.social_input[M] = 1.0 - np.exp(-source_mass)

            # Direction = weighted mean unit direction of actually moving sources.
            if np.any(moving_nb):
                unit_v = np.zeros_like(nb_v)
                unit_v[moving_nb, 0] = nb_v[moving_nb, 0] / nb_speed[moving_nb]
                unit_v[moving_nb, 1] = nb_v[moving_nb, 1] / nb_speed[moving_nb]

                vx = source_w @ unit_v[:, 0]
                vy = source_w @ unit_v[:, 1]
                vn = np.hypot(vx, vy)

                good_flow = vn > EPS
                if np.any(good_flow):
                    Mf = M[good_flow]
                    # Magnitude carries source intensity/coherence only for
                    # diagnostics and movement-following strength.
                    q_local = np.clip(
                        source_mass[good_flow],
                        0.0,
                        1.0
                    )
                    self.local_flow[Mf, 0] = (
                        C.V_WALK * q_local * vx[good_flow] / vn[good_flow]
                    )
                    self.local_flow[Mf, 1] = (
                        C.V_WALK * q_local * vy[good_flow] / vn[good_flow]
                    )

    # ------------------------------------------------------------------
    # Ordinary BASE locomotion
    # ------------------------------------------------------------------
    def _base_desired_heading(
        self,
        dt,
        reaction_rest,
        attraction_allowed
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

        away_ang = np.arctan2(
            self.nn_away[:, 1],
            self.nn_away[:, 0]
        )

        # Strong local separation overrides the ordinary BASE target.
        desired[
            too_close
        ] = away_ang[
            too_close
        ]

        # Weak cohesion only within an already-existing local flock.
        # It does not start locomotion and it does not pull isolated flocks
        # together across distances > R_SOCIAL.
        cohesion_active = (
            in_bout
            & attraction_allowed
            & (~too_close)
            & (self.flock_neighbor_count > 0)
            & np.isfinite(self.nn_dist)
            & (self.nn_dist > C.R_MIN)
        )

        if np.any(cohesion_active):
            base_x = np.cos(desired)
            base_y = np.sin(desired)

            # Geometry-derived weight: almost zero near R_MIN, increasing only
            # as spacing opens inside R_SOCIAL. No new fitted gain parameter.
            coh_w = np.clip(
                (self.nn_dist - C.R_MIN)
                / max(C.R_SOCIAL - C.R_MIN, EPS),
                0.0,
                1.0
            )

            cx = (
                base_x
                + coh_w * self.flock_centroid_dir[:, 0]
            )
            cy = (
                base_y
                + coh_w * self.flock_centroid_dir[:, 1]
            )
            cn = np.hypot(cx, cy)
            good_coh = cohesion_active & (cn > EPS)
            desired[good_coh] = np.arctan2(
                cy[good_coh],
                cx[good_coh]
            )

        return desired, in_bout

    # ------------------------------------------------------------------
    # Threat-relative social direction projection.
    #
    # This is a geometric safety constraint, not social memory/alignment.
    # While an advancing robot threat exists:
    # - remove motion opposite robot travel,
    # - remove motion toward the robot,
    # - if nothing remains, use direct robot escape geometry.
    # ------------------------------------------------------------------
    def _remove_robot_inward_component(
        self,
        raw_dir,
        radial,
        active_mask
    ):
        """
        Minimal robot-relative safety constraint.

        raw_dir is formed first from robot + social flow + separation.
        During an actual direct robot cue, only the component pointing toward
        the robot is removed. Backward/sideways local flow is otherwise allowed.

            d = raw_dir · radial_away
            if d < 0:
                raw_dir <- raw_dir - d * radial_away

        No forward alignment, no leadership, no fallback to a 'correct' flock
        direction is imposed here.
        """
        out = raw_dir.copy()
        if not np.any(active_mask):
            return out

        dot = (
            out[:, 0] * radial[:, 0]
            + out[:, 1] * radial[:, 1]
        )
        inward = active_mask & (dot < 0.0)
        if np.any(inward):
            out[inward, 0] -= dot[inward] * radial[inward, 0]
            out[inward, 1] -= dot[inward] * radial[inward, 1]

        return out

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
    # world.py compatibility: new world accepts (x,y), older v1-style world
    # accepts (x,y,theta). This patch supports both.
    # ------------------------------------------------------------------
    def _resolve_world_wall(self):
        try:
            return self.world.resolve_wall_collision(self.x, self.y)
        except TypeError:
            return self.world.resolve_wall_collision(self.x, self.y, self.theta)

    # ------------------------------------------------------------------
    # End-wall relief field.
    #
    # At the wall in front of the robot:
    #   1) immediately at the wall -> move along the wall tangent toward
    #      lower local density;
    #   2) farther inside the WALL_RELIEF_DEPTH zone -> turn progressively
    #      toward -robot_direction so adjacent lanes are cleared backward.
    #
    # This is only enabled where crowd pressure exists AND an active
    # robot/social threat reaches the zone. It is not a permanent wall current.
    # ------------------------------------------------------------------
    def _front_wall_relief(
        self,
        robot_pos,
        robot_vel,
        threat_strength
    ):
        N = self.N
        out = np.zeros((N, 2))
        gain = np.zeros(N)
        active = np.zeros(N, dtype=bool)

        if robot_pos is None:
            return out, gain, active

        rv = np.hypot(robot_vel[0], robot_vel[1])
        if rv <= EPS:
            return out, gain, active

        tx = robot_vel[0] / rv
        ty = robot_vel[1] / rv

        # Axis-aligned house: choose the wall that the current robot heading
        # points toward most strongly.
        if abs(tx) >= abs(ty):
            if tx >= 0.0:
                front_dist = (self.world.L - self.rc) - self.x
                wall_tx, wall_ty = 0.0, 1.0
            else:
                front_dist = self.x - self.rc
                wall_tx, wall_ty = 0.0, 1.0
        else:
            if ty >= 0.0:
                front_dist = (self.world.W - self.rc) - self.y
                wall_tx, wall_ty = 1.0, 0.0
            else:
                front_dist = self.y - self.rc
                wall_tx, wall_ty = 1.0, 0.0

        # Must be compressed and currently reached by robot/social threat.
        crowd_active = self.crowd_pressure > C.PRESSURE_THRESHOLD
        reached = threat_strength > 1e-4
        active[:] = (
            (front_dist >= 0.0)
            & (front_dist < C.WALL_RELIEF_DEPTH)
            & crowd_active
            & reached
            & self.social_propagation_active
        )

        if not np.any(active):
            return out, gain, active

        # Tangent sign: lower-density direction first, then behavioral flow,
        # then persistent sign only as a tie-breaker.
        crowd_tan = (
            self.crowd_vec[:, 0] * wall_tx
            + self.crowd_vec[:, 1] * wall_ty
        )
        flow_tan = (
            self.flow_memory[:, 0] * wall_tx
            + self.flow_memory[:, 1] * wall_ty
        )
        sign = np.where(
            np.abs(crowd_tan) > EPS,
            np.sign(crowd_tan),
            np.where(
                np.abs(flow_tan) > EPS,
                np.sign(flow_tan),
                self.wall_flow_sign
            )
        )

        # Smoothly rotate:
        # wall (d=0) -> tangent,
        # inner edge (d=depth) -> -robot direction.
        # The centerline itself should clear SIDEWAYS rather than reverse into
        # the robot; the backward component grows only after a bird has moved
        # into an adjacent lane (cross-track outside robot body width).
        relx = self.x - robot_pos[0]
        rely = self.y - robot_pos[1]
        path_nx = -ty
        path_ny = tx
        cross_track = relx * path_nx + rely * path_ny
        side_factor = np.clip(
            np.abs(cross_track)
            / max(C.ROBOT_RADIUS + self.rc, EPS),
            0.0,
            1.0
        )

        geometric_tangent_w = np.clip(
            1.0 - front_dist / max(C.WALL_RELIEF_DEPTH, EPS),
            0.0,
            1.0
        )
        back_w = (1.0 - geometric_tangent_w) * side_factor
        tangent_w = 1.0 - back_w

        gx = tangent_w * sign * wall_tx - back_w * tx
        gy = tangent_w * sign * wall_ty - back_w * ty
        gn = np.hypot(gx, gy)

        good = active & (gn > EPS)
        out[good, 0] = gx[good] / gn[good]
        out[good, 1] = gy[good] / gn[good]

        # Keep the relief strength across the full zone. Direction changes from
        # tangent to backward with distance; if magnitude also decayed to zero,
        # the adjacent-lane backward-clearing part would disappear.
        crowd_strength = np.clip(
            self.crowd_pressure - C.PRESSURE_THRESHOLD,
            0.0,
            1.0
        )
        gain[active] = crowd_strength[active]

        return out, gain, active

    # ------------------------------------------------------------------
    # Feeder-pan tangential flow projection.
    #
    # Feed pans are hard circular obstacles. The old collision resolver only
    # projected a bird radially back to the pan boundary AFTER translation, so
    # a chicken whose desired heading pointed through a pan could repeatedly
    # walk into the same obstacle and appear stationary.
    #
    # Here the desired behavioral/crowd vector is projected onto the obstacle
    # tangent BEFORE locomotion. No new range parameter is introduced:
    # the existing R_WALL_FLOW is reused as a small steering shell.
    # ------------------------------------------------------------------
    def _apply_feeder_flow(
        self,
        vx,
        vy
    ):
        vx = vx.copy()
        vy = vy.copy()

        pans = getattr(
            self.world,
            "feeder_pans",
            None
        )
        if pans is None or len(pans) == 0:
            return vx, vy, np.zeros(self.N, dtype=bool)

        min_d = (
            self.rc
            + C.FEEDER_RADIUS
        )

        px = self.x[:, None] - pans[None, :, 0]
        py = self.y[:, None] - pans[None, :, 1]
        d = np.hypot(px, py)

        nearest = np.argmin(
            d,
            axis=1
        )
        row = np.arange(self.N)
        dmin = d[row, nearest]

        # existing wall-flow shell reused; not a new behavior parameter.
        near = (
            dmin
            < min_d + C.R_WALL_FLOW
        )

        if not np.any(near):
            return vx, vy, near

        nx = np.zeros(self.N)
        ny = np.zeros(self.N)
        good = dmin > EPS
        nx[good] = px[row[good], nearest[good]] / dmin[good]
        ny[good] = py[row[good], nearest[good]] / dmin[good]

        # n points outward from the pan. dot<0 means desired motion points
        # inward through the obstacle, so remove only that normal component.
        dot = (
            vx * nx
            + vy * ny
        )
        blocked = (
            near
            & (dot < 0.0)
        )

        vx[blocked] -= (
            dot[blocked]
            * nx[blocked]
        )
        vy[blocked] -= (
            dot[blocked]
            * ny[blocked]
        )

        # If projection removes almost the whole vector, choose the tangent
        # consistent with current local flow / robot travel. This makes birds
        # go around a feeder instead of being pinned against it.
        norm = np.hypot(vx, vy)
        stuck = (
            blocked
            & (norm < 1e-8)
        )

        if np.any(stuck):
            tx = -ny
            ty = nx

            flow_dot = (
                self.flow_memory[:, 0] * tx
                + self.flow_memory[:, 1] * ty
            )
            robot_dot = (
                self.last_robot_forward[0] * tx
                + self.last_robot_forward[1] * ty
            )

            sign = np.where(
                np.abs(flow_dot) > EPS,
                np.sign(flow_dot),
                np.where(
                    np.abs(robot_dot) > EPS,
                    np.sign(robot_dot),
                    self.wall_flow_sign
                )
            )

            vx[stuck] = sign[stuck] * tx[stuck]
            vy[stuck] = sign[stuck] * ty[stuck]

        return vx, vy, near

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
    # Robot/chicken hard body contact.
    # Behavioral near-nonresponders may wait until this happens, but the robot
    # cannot pass through them. Contact displacement is physical and is NOT
    # counted as voluntary escape distance.
    # ------------------------------------------------------------------
    def resolve_robot_contact(self, robot_pos):
        dx_out = np.zeros(self.N)
        dy_out = np.zeros(self.N)

        if robot_pos is None:
            return dx_out, dy_out

        rx = self.x - robot_pos[0]
        ry = self.y - robot_pos[1]
        d = np.hypot(rx, ry)
        min_d = C.ROBOT_RADIUS + self.rc

        hit = d < min_d
        if not np.any(hit):
            return dx_out, dy_out

        ux = np.zeros(self.N)
        uy = np.zeros(self.N)

        good = hit & (d > EPS)
        ux[good] = rx[good] / d[good]
        uy[good] = ry[good] / d[good]

        zero = hit & (~good)
        if np.any(zero):
            # If centers coincide, push opposite the robot travel direction.
            ux[zero] = -self.last_robot_forward[0]
            uy[zero] = -self.last_robot_forward[1]

        penetration = np.maximum(0.0, min_d - d)
        dx = penetration * ux
        dy = penetration * uy

        # Near a wall, convert blocked contact displacement into an allowed
        # tangent/low-density displacement instead of pinning the chicken.
        dx, dy = self._project_contact_displacement(dx, dy)

        self.x += dx
        self.y += dy
        self._resolve_world_wall()
        self.world.resolve_feeder_collision(self.x, self.y)

        return dx, dy

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
            iterations = C.CONTACT_MAX_ITERS

        total_dx = np.zeros(
            self.N
        )
        total_dy = np.zeros(
            self.N
        )

        two_rc = (
            2.0 * self.rc
        )

        used_iters = 0
        last_max_penetration = 0.0

        for _ in range(
            iterations
        ):
            used_iters += 1
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
            max_penetration = 0.0

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
                if np.any(overlap):
                    max_penetration = max(
                        max_penetration,
                        float(delta[overlap].max())
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

            last_max_penetration = max_penetration
            if max_penetration <= C.CONTACT_TOLERANCE:
                break

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
            self._resolve_world_wall()

        self.last_contact_iters = used_iters
        self.last_contact_max_penetration = last_max_penetration
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

            # Rear gate for NEW robot-induced social response. This is a robot
            # geometry condition, not social memory/provenance.
            rear_cutoff = C.ROBOT_RADIUS + C.CHICKEN_RADIUS
            self.social_propagation_active[:] = (
                longitudinal >= -rear_cutoff
            )
        else:
            self.social_propagation_active.fill(False)
            self.behavioral_source_active.fill(False)

        # 2) Read PREVIOUS-step sources. Newly recruited birds are invisible to
        # their next neighbor until their current-step movement has occurred.
        self.compute_neighbor_stats()

        # Delayed excitation only. This is the single social state retained.
        a_social = np.exp(-dt / max(C.TAU_SOCIAL_EXCITATION, EPS))
        self.social_excitation[:] = (
            a_social * self.social_excitation
            + (1.0 - a_social) * self.social_input
        )

        # No separate directional social memory: follow the previous-frame
        # movement field directly.
        self.flow_memory[:] = self.local_flow
        q_flow, flow_dir, flow_speed = St.flow_cue(self.local_flow)
        self.flow_speed[:] = flow_speed

        # 3) Robot geometry / closing speed / threat.
        q_side_pass = np.zeros(N)

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
            radial = np.zeros(
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

        # Chickens nearly beside a moving robot can have LOS closing_speed≈0
        # even though the robot is passing close by. This occurs around the
        # feeder/drinker-line bands (±0.75 m from the center route) and made
        # visibly nearby birds look behaviorally blind.
        #
        # Add a side-pass proximity cue without extending the existing direct
        # clearance cutoff:
        # - sqrt(q_distance): still exactly zero outside ROBOT_CUE_CLEARANCE,
        #   but less suppressive near the edge of that zone;
        # - side_fraction: strongest when the bird is lateral to robot travel,
        #   zero when directly in front/behind;
        # - rear/front gate prevents a new response after the robot has passed.
        if robot_frame_active:
            forward_dot = (
                radial[:, 0] * tx
                + radial[:, 1] * ty
            )
            side_fraction = np.sqrt(
                np.clip(
                    1.0 - forward_dot ** 2,
                    0.0,
                    1.0
                )
            )
            q_robot_speed = np.clip(
                robot_speed / max(C.V_THREAT_REF, EPS),
                0.0,
                1.0
            )
            q_side_pass = (
                np.sqrt(q_distance)
                * q_robot_speed
                * side_fraction
                * self.social_propagation_active.astype(float)
            )
            q_robot = np.maximum(
                q_robot,
                q_side_pass
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
        #
        # Social E_i is local moving-neighbor excitation, but NEW social ESCAPE
        # responses are additionally restricted to a simple robot-centered
        # envelope. This supplies finite spatial extent without path/provenance
        # graphs or flock-wide cascade memory.
        social_signal = self.social_excitation

        if robot_pos is not None:
            social_clearance = np.maximum(
                0.0,
                clearance
            )
            social_envelope = np.clip(
                1.0
                - social_clearance / max(C.SOCIAL_PROP_RANGE, EPS),
                0.0,
                1.0
            )
        else:
            social_envelope = np.zeros(N)

        # NEW recruitment is rear-gated.
        social_fraction_active = (
            social_signal
            * social_envelope
            * self.social_propagation_active.astype(float)
        )

        # ACTIVE-bout support is not hard rear-gated. This avoids an artificial
        # row of birds stopping exactly when the robot rear cutoff crosses them.
        social_fraction_support = (
            social_signal
            * social_envelope
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

        drive_social_support = (
            self.social_sensitivity
            * C.BETA
            * np.maximum(0.0, social_fraction_support)
        )
        perceived_threat = np.maximum(
            drive_robot,
            drive_social_support
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
            self.escape_distance_travelled[
                start
            ] = 0.0

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
        # Activation and direction remain separate. The raw social direction is
        # the observed previous-frame movement field; it is not rectified into
        # robot-forward motion.
        social_dir = flow_dir.copy()

        social_support = np.clip(
            social_signal,
            0.0,
            1.0
        )

        robot_weight = drive_robot
        social_weight = drive_social
        flow_follow_drive = np.zeros(N)

        # Passive crowd relaxation is a separate geometry layer. It can remain
        # active after the threat has passed and does not depend on sensitivity.
        crowd_strength = np.clip(
            self.crowd_pressure - C.PRESSURE_THRESHOLD,
            0.0,
            1.0
        )
        crowd_active = crowd_strength > 0.0

        # End-wall relief: threat reaching a compressed front wall rotates the
        # local social direction from wall-tangent to -robot-direction over
        # ~WALL_RELIEF_DEPTH, clearing adjacent lanes instead of forming a plug.
        threat_strength = np.maximum(
            q_robot,
            np.maximum(social_weight, flow_follow_drive)
        )
        if C.ENABLE_WALL_RELIEF:
            (
                relief_dir,
                relief_gain,
                wall_relief_active
            ) = self._front_wall_relief(
                robot_pos,
                robot_vel,
                threat_strength
            )
        else:
            relief_dir = np.zeros((N, 2))
            relief_gain = np.zeros(N)
            wall_relief_active = np.zeros(N, dtype=bool)
        self.wall_relief_dir[:] = relief_dir
        self.wall_relief_gain[:] = relief_gain
        self.wall_relief_active[:] = wall_relief_active

        # Threat response direction = robot + social movement-following
        # + short-range separation. Weak attraction is OFF here.
        sep_strength = np.clip(
            (C.R_MIN - self.nn_dist) / max(C.R_MIN, EPS),
            0.0,
            1.0
        )
        ux = (
            robot_weight * geo_dir[:, 0]
            + social_weight * social_dir[:, 0]
            + sep_strength * self.nn_away[:, 0]
            + relief_gain * relief_dir[:, 0]
        )
        uy = (
            robot_weight * geo_dir[:, 1]
            + social_weight * social_dir[:, 1]
            + sep_strength * self.nn_away[:, 1]
            + relief_gain * relief_dir[:, 1]
        )

        raw_dir = np.column_stack([ux, uy])
        direct_safety_active = (
            is_escape
            & (q_robot > EPS)
        )
        raw_dir = self._remove_robot_inward_component(
            raw_dir,
            radial,
            direct_safety_active
        )
        ux = raw_dir[:, 0]
        uy = raw_dir[:, 1]

        ux, uy, near_wall = (
            self._apply_wall_flow(
                ux,
                uy
            )
        )
        ux, uy, near_feeder = (
            self._apply_feeder_flow(
                ux,
                uy
            )
        )

        move_norm = np.hypot(
            ux,
            uy
        )

        # v0.4.4: no persistent target heading. Recompute the desired direction
        # every frame from current robot geometry + current social flow +
        # separation, then let body heading dynamics provide the only memory.
        esc_desired = self.theta.copy()
        valid = move_norm > EPS
        esc_desired[valid] = np.arctan2(
            uy[valid],
            ux[valid]
        )

        # 8) Ordinary BASE heading.
        # Weak attraction is only a calm flock-distribution tendency.
        calm_attraction_allowed = (
            (q_robot <= EPS)
            & (self.social_excitation <= C.CALM_SOCIAL_THRESHOLD)
            & (~reaction_rest)
        )
        base_desired, in_base_bout = (
            self._base_desired_heading(
                dt,
                reaction_rest,
                calm_attraction_allowed
            )
        )

        # Social excitation only changes finite ESCAPE bout-start probability.
        # It never creates continuous BASE locomotion.
        social_dir_norm = np.hypot(
            social_dir[:, 0],
            social_dir[:, 1]
        )
        social_follow_base = np.zeros(N, dtype=bool)
        social_follow_speed_raw = np.zeros(N)

        # Passive crowd relaxation no longer becomes a BASE behavioral mode.
        # It is integrated later in the mechanical layer and therefore does not
        # change heading or behavioral bout state.
        crowd_relax_active = crowd_active

        desired_theta = np.where(
            is_escape,
            esc_desired,
            base_desired
        )

        # Rear gate applies to NEW recruitment only. Active ESCAPE bouts are
        # terminated by their own distance/timer or perceived-threat loss below.

        # 9) Target speed.
        v_target = np.where(
            in_base_bout,
            C.V_BASE,
            0.0
        )

        v_social_follow = np.zeros(N)
        v_crowd = C.V_CROWD_MAX * crowd_strength
        tau = np.full(
            N,
            C.TAU_BASE
        )

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

        # STOP/refractory has priority over social excitation.
        rest_social_follow = np.zeros(N, dtype=bool)

        # Passive mechanical crowd relaxation is applied after locomotion and
        # is therefore unaffected by behavioral refractory.

        # BASE birds behind the threat front settle instead of being carried
        # by residual neighbor motion.
        rear_flow_coast = (
            (self.state == BASE)
            & (~self.social_propagation_active)
            & (self.speed > C.V_BASE)
        )
        v_target[rear_flow_coast] = 0.0
        tau[rear_flow_coast] = C.TAU_STOP

        # Density-jam mobility:
        # signal transmission above is NOT reduced by high density; only actual
        # voluntary locomotion becomes difficult once pressure exceeds JAM_PRESSURE.
        jam_excess = np.maximum(
            0.0,
            self.crowd_pressure - C.JAM_PRESSURE
        )
        self.mobility_factor[:] = 1.0 / (1.0 + jam_excess ** 2)

        voluntary_move = (
            is_escape
            | in_base_bout
        )
        v_target[voluntary_move] *= self.mobility_factor[voluntary_move]

        # 10) Heading dynamics.
        # A chicken with essentially zero locomotion demand must NOT rotate in
        # place just because a weak social/crowd direction exists.
        locomotion_state = (
            is_escape
            | in_base_bout
        )
        turn_active = (
            locomotion_state
            & (v_target >= 0.5 * C.V_BASE)
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

        # hard obstacle clamp before passive/contact mechanics
        self.world.resolve_feeder_collision(
            self.x,
            self.y
        )
        self._resolve_world_wall()

        # 12) Passive crowd relaxation = mechanical displacement.
        #
        # It does not rotate heading, does not start/extend an ESCAPE bout, and
        # does not depend on behavioral sensitivity. The direction is the local
        # short-range pressure gradient projected by the same geometric
        # wall/feeder constraints.
        crowd_dir_x = self.crowd_vec[:, 0].copy()
        crowd_dir_y = self.crowd_vec[:, 1].copy()
        crowd_dir_x, crowd_dir_y, _ = self._apply_wall_flow(
            crowd_dir_x,
            crowd_dir_y
        )
        crowd_dir_x, crowd_dir_y, _ = self._apply_feeder_flow(
            crowd_dir_x,
            crowd_dir_y
        )
        crowd_dir_norm = np.hypot(crowd_dir_x, crowd_dir_y)
        good_crowd_dir = crowd_dir_norm > EPS
        crowd_dir_x[good_crowd_dir] /= crowd_dir_norm[good_crowd_dir]
        crowd_dir_y[good_crowd_dir] /= crowd_dir_norm[good_crowd_dir]
        crowd_dir_x[~good_crowd_dir] = 0.0
        crowd_dir_y[~good_crowd_dir] = 0.0

        crowd_relax_vx = v_crowd * crowd_dir_x
        crowd_relax_vy = v_crowd * crowd_dir_y
        crowd_relax_dx = crowd_relax_vx * dt
        crowd_relax_dy = crowd_relax_vy * dt

        self.x += crowd_relax_dx
        self.y += crowd_relax_dy

        self.world.resolve_feeder_collision(
            self.x,
            self.y
        )
        self._resolve_world_wall()

        # 13) Physical contacts.
        # Low behavioral responders may reach robot contact; contact itself is
        # geometry-only and independent of sensitivity.
        robot_push_dx_1, robot_push_dy_1 = self.resolve_robot_contact(robot_pos)

        chicken_push_dx, chicken_push_dy = (
            self.resolve_contact_push(
                iterations=C.CONTACT_MAX_ITERS
            )
        )

        # Neighbor correction can push a chicken back into the robot, so enforce
        # robot body contact once more.
        robot_push_dx_2, robot_push_dy_2 = self.resolve_robot_contact(robot_pos)

        push_dx = robot_push_dx_1 + chicken_push_dx + robot_push_dx_2
        push_dy = robot_push_dy_1 + chicken_push_dy + robot_push_dy_2

        # Robot/chicken contact is a physical disturbance. push_velocity is
        # deliberately exposed to social perception on the NEXT frame; passive
        # crowd_relax velocity remains excluded from that perception.
        self.push_velocity[:, 0] = (
            push_dx / max(dt, EPS)
        )
        self.push_velocity[:, 1] = (
            push_dy / max(dt, EPS)
        )

        # 14) Separate behavioral and mechanical movement.
        # Social perception next frame uses behavioral_velocity + push_velocity.
        # mechanical_velocity additionally contains passive crowd relaxation.
        behavioral_move_mask = is_escape.copy()
        self.behavioral_velocity[:, 0] = (
            behavioral_move_mask.astype(float) * active_vx
        )
        self.behavioral_velocity[:, 1] = (
            behavioral_move_mask.astype(float) * active_vy
        )

        self.mechanical_velocity[:, 0] = (
            self.push_velocity[:, 0]
            + crowd_relax_vx
        )
        self.mechanical_velocity[:, 1] = (
            self.push_velocity[:, 1]
            + crowd_relax_vy
        )
        self.effective_velocity[:] = (
            self.behavioral_velocity + self.mechanical_velocity
        )

        # Consumed only next frame -> explicit one-hop-per-frame causality.
        self.behavioral_source_active[:] = behavioral_move_mask
        self.mechanical_source_active[:] = (
            np.hypot(
                self.mechanical_velocity[:, 0],
                self.mechanical_velocity[:, 1]
            ) > 1e-4
        )

        # 15) Distance-based stimulus bout end.
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
        self.escape_distance_travelled[
            active_escape
        ] += (
            active_step_distance[
                active_escape
            ]
        )

        self.escape_timer[
            active_escape
        ] -= dt

        distance_finished = (
            self.escape_distance_left <= 0.0
        )
        timer_finished = (
            self.escape_timer <= 0.0
        )
        threat_loss_finished = (
            active_escape
            & (
                self.escape_distance_travelled
                >= C.L_ESCAPE_MIN
            )
            & (
                perceived_threat
                < C.H_STOP
            )
        )

        finished = (
            active_escape
            & (
                distance_finished
                | timer_finished
                | threat_loss_finished
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
            self.escape_distance_travelled[
                finished
            ] = 0.0

        # 16) Timers.
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

        # 17) Histories.
        self.prev_robot_threat[:] = q_robot

        if robot_pos is None:
            self.prev_robot_pos = None
        else:
            self.prev_robot_pos = robot_pos.copy()

        # 18) Diagnostics.
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
        self.last_near_feeder_ratio = float(
            near_feeder.mean()
        )
        self.last_side_pass_cue_max = float(
            q_side_pass.max()
        )
        self.last_reaction_rest_ratio = float(
            reaction_rest.mean()
        )
        self.last_bout_end_count = bout_end_count
        self.last_threat_loss_end_count = int(
            threat_loss_finished.sum()
        )
        self.last_perceived_threat_mean = float(
            perceived_threat.mean()
        )
        self.last_perceived_threat_max = float(
            perceived_threat.max()
        )
        self.last_contact_social_source_ratio = float(
            (
                np.hypot(
                    self.push_velocity[:, 0],
                    self.push_velocity[:, 1]
                ) >= 0.5 * C.V_BASE
            ).mean()
        )

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
        self.last_social_follow_ratio = 0.0
        self.last_social_prop_gain_mean = 1.0
        self.last_social_prop_gain_max = 1.0
        self.last_social_path_max = 0.0
        self.last_social_source_mass_max = float(self.social_source_mass.max())
        self.last_social_excitation_max = float(self.social_excitation.max())
        self.last_local_density_mean = float(self.local_density.mean())
        self.last_local_density_max = float(self.local_density.max())
        self.last_density_signal_mean = float(self.density_signal.mean())
        self.last_density_signal_max = float(self.density_signal.max())
        self.last_mobility_mean = float(self.mobility_factor.mean())
        self.last_mobility_min = float(self.mobility_factor.min())
        self.last_wall_relief_ratio = float(self.wall_relief_active.mean())
        self.last_crowd_pressure_mean = float(self.crowd_pressure.mean())
        self.last_crowd_pressure_max = float(self.crowd_pressure.max())
        self.last_crowd_relax_ratio = float(crowd_relax_active.mean())
        self.last_front_gate_ratio = float(self.social_propagation_active.mean())
        self.last_rear_social_follow_ratio = 0.0
        self.last_behavioral_source_ratio = float(
            self.behavioral_source_active.mean()
        )
        self.last_mechanical_source_ratio = float(
            self.mechanical_source_active.mean()
        )
        self.last_social_envelope_active_ratio = float(
            (social_envelope > 0.0).mean()
        )
        self.last_local_flock_member_ratio = float(
            (self.flock_neighbor_count > 0).mean()
        )
        self.last_local_flock_neighbor_mean = float(
            self.flock_neighbor_count.mean()
        )
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
        # Diagnostics must remain semantically separate:
        # - crowd_active: pressure above the passive-relaxation threshold
        # - crowd_move: actual nonzero passive crowd-relaxation velocity
        # - push_present/social_push: physical contact-solver displacement
        self.last_crowd_active_ratio = float(
            crowd_active.mean()
        )

        crowd_move_actual = (
            np.hypot(
                crowd_relax_vx,
                crowd_relax_vy
            ) > 1e-4
        )
        self.last_crowd_move_ratio = float(
            crowd_move_actual.mean()
        )
        self.last_crowd_relax_ratio = (
            self.last_crowd_move_ratio
        )

        self.last_v_crowd_mean = float(
            v_crowd.mean()
        )
        self.last_v_crowd_max = float(
            v_crowd.max()
        )

        self.last_social_push_ratio = float(
            pushed.mean()
        )
        self.last_social_push_mean = float(
            push_mag.mean()
        )
        self.last_social_push_max = float(
            push_mag.max()
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
