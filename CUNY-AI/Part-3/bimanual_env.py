#!/usr/bin/env python3
"""MuJoCo grasp-handoff-throw task for an R1 with two Revo 2 Basic hands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model/r1_revo2_bimanual.xml"

RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
)
RIGHT_HAND_JOINTS = (
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)
LEFT_HAND_JOINTS = tuple(name.replace("right_", "left_") for name in RIGHT_HAND_JOINTS)
POWER_GRASP = np.array([0.82, 0.88, 0.84, 0.88, 0.90, 0.92])

HOME = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
}


class Policy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return 22 normalized commands in [-1, 1]."""


@dataclass(frozen=True)
class TaskConfig:
    control_hz: float = 40.0
    episode_seconds: float = 7.10
    right_reach_end: float = 1.50
    contact_enable: float = 1.47
    fixture_release: float = 2.35
    right_close_end: float = 2.48
    left_reach_start: float = 2.05
    handoff_reach_end: float = 3.82
    left_close_start: float = 3.62
    left_close_end: float = 4.28
    right_release_start: float = 4.32
    right_release_end: float = 4.58
    left_lift_end: float = 5.05
    windup_end: float = 5.48
    left_release_time: float = 5.75
    throw_end: float = 6.06
    action_delay_max: int = 2
    observation_noise_std: float = 0.009


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def phase_fraction(elapsed: float, start: float, end: float) -> float:
    return smoothstep((elapsed - start) / max(end - start, 1e-8))


class BimanualHandoffThrowEnv:
    """Fixed-pelvis bimanual manipulation curriculum with domain randomization."""

    action_size = 22

    def __init__(self, config: TaskConfig | None = None, domain_randomization: bool = True) -> None:
        self.config = config or TaskConfig()
        self.domain_randomization = domain_randomization
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.frame_skip = max(1, int(round(1.0 / (self.config.control_hz * self.dt))))
        self.control_dt = self.frame_skip * self.dt
        self.max_steps = int(math.ceil(self.config.episode_seconds / self.control_dt))
        self.rng = np.random.default_rng(24)

        self.right_arm = self._joint_bundle(RIGHT_ARM_JOINTS, "r1_")
        self.left_arm = self._joint_bundle(LEFT_ARM_JOINTS, "r1_")
        self.right_hand = self._joint_bundle(RIGHT_HAND_JOINTS, "")
        self.left_hand = self._joint_bundle(LEFT_HAND_JOINTS, "")
        policy_actuators = set(np.concatenate((self.right_arm[5], self.left_arm[5], self.right_hand[5], self.left_hand[5])))
        self.fixed_actuator_ids = np.array([index for index in range(self.model.nu) if index not in policy_actuators], dtype=int)

        self.right_acquire_site_id = self._site_id("right_acquire_site")
        self.right_grasp_site_id = self._site_id("right_grasp_site")
        self.left_acquire_site_id = self._site_id("left_acquire_site")
        self.left_grasp_site_id = self._site_id("left_grasp_site")
        self.ball_body_id = self._body_id("ball")
        self.ball_geom_id = self._geom_id("ball_geom")
        ball_joint = self._joint_id("ball_freejoint")
        self.ball_qpos = int(self.model.jnt_qposadr[ball_joint])
        self.ball_dof = int(self.model.jnt_dofadr[ball_joint])
        self.anchor_mocap_id = int(self.model.body_mocapid[self._body_id("ball_anchor")])
        self.handoff_mocap_id = int(self.model.body_mocapid[self._body_id("handoff_marker")])
        self.target_mocap_id = int(self.model.body_mocapid[self._body_id("throw_target")])
        self.fixture_eq_id = self._equality_id("ball_fixture")
        self.right_hand_geoms = self._descendant_geoms("right_base_link")
        self.left_hand_geoms = self._descendant_geoms("left_base_link")

        self.base_gainprm = self.model.actuator_gainprm.copy()
        self.base_biasprm = self.model.actuator_biasprm.copy()
        self.base_ball_size = self.model.geom_size[self.ball_geom_id].copy()
        self.base_ball_friction = self.model.geom_friction[self.ball_geom_id].copy()
        self.base_ball_mass = float(self.model.body_mass[self.ball_body_id])
        self.base_ball_inertia = self.model.body_inertia[self.ball_body_id].copy()
        self.base_ball_contype = int(self.model.geom_contype[self.ball_geom_id])
        self.base_ball_conaffinity = int(self.model.geom_conaffinity[self.ball_geom_id])

        self.right_home = np.array([HOME.get(name, 0.0) for name in RIGHT_ARM_JOINTS])
        self.left_home = np.array([HOME.get(name, 0.0) for name in LEFT_ARM_JOINTS])
        self.previous_action = np.zeros(self.action_size)
        self.action_queue: list[np.ndarray] = []
        self.action_delay = 0
        self.elapsed = 0.0
        self.step_count = 0
        self.ball_start = np.zeros(3)
        self.handoff_target = np.zeros(3)
        self.throw_target = np.zeros(3)
        self.right_q_grasp = self.right_home.copy()
        self.right_q_handoff = self.right_home.copy()
        self.right_q_retreat = self.right_home.copy()
        self.left_q_handoff = self.left_home.copy()
        self.left_q_lift = self.left_home.copy()
        self.left_q_windup = self.left_home.copy()
        self.left_q_throw = self.left_home.copy()
        self.fixture_released = False
        self.contact_enabled = False
        self.ball_carrier = "fixture"
        self.carrier_offset = np.zeros(3)
        self.carrier_switch_time = 0.0
        self.min_right_reach = float("inf")
        self.min_handoff_error = float("inf")
        self.max_right_contacts = 0
        self.max_left_contacts = 0
        self.right_grasped = False
        self.transferred = False
        self.left_held_until_throw = False
        self.peak_throw_speed = 0.0
        self.peak_forward_speed = 0.0
        self.max_forward_distance = 0.0
        self.minimum_target_distance = float("inf")
        self.reward_total = 0.0
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        mujoco.mj_forward(self.model, self.data)
        self.observation_size = int(self._observation(noise=False).size)
        self.reset(seed=24)

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise RuntimeError(f"Model is missing {name}")
        return int(value)

    def _joint_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def _actuator_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _site_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_SITE, name)

    def _body_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def _equality_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def _joint_bundle(self, names: tuple[str, ...], actuator_prefix: str) -> tuple[np.ndarray, ...]:
        joint_ids = np.array([self._joint_id(name) for name in names])
        return (
            joint_ids,
            self.model.jnt_qposadr[joint_ids].astype(int),
            self.model.jnt_dofadr[joint_ids].astype(int),
            self.model.jnt_range[joint_ids, 0].copy(),
            self.model.jnt_range[joint_ids, 1].copy(),
            np.array([self._actuator_id(f"{actuator_prefix}{name}") for name in names]),
        )

    def _descendant_geoms(self, base_name: str) -> set[int]:
        base_id = self._body_id(base_name)
        result: set[int] = set()
        for geom_id in range(self.model.ngeom):
            current = int(self.model.geom_bodyid[geom_id])
            while current > 0:
                if current == base_id:
                    result.add(geom_id)
                    break
                current = int(self.model.body_parentid[current])
        return result

    def _site_position(self, site_id: int) -> np.ndarray:
        return self.data.site_xpos[site_id].copy()

    @property
    def ball_position(self) -> np.ndarray:
        return self.data.qpos[self.ball_qpos : self.ball_qpos + 3].copy()

    @property
    def ball_velocity(self) -> np.ndarray:
        return self.data.qvel[self.ball_dof : self.ball_dof + 3].copy()

    def _site_velocity(self, site_id: int) -> np.ndarray:
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site_id)
        return self._jacp @ self.data.qvel

    def _ratios(self, bundle: tuple[np.ndarray, ...]) -> np.ndarray:
        _, qpos, _, low, high, _ = bundle
        return np.clip((self.data.qpos[qpos] - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)

    def _contact_count(self, hand_geoms: set[int]) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 == self.ball_geom_id:
                other = int(contact.geom2)
            elif contact.geom2 == self.ball_geom_id:
                other = int(contact.geom1)
            else:
                continue
            if other in hand_geoms:
                count += 1
        return count

    def contact_counts(self) -> tuple[int, int]:
        return self._contact_count(self.right_hand_geoms), self._contact_count(self.left_hand_geoms)

    def _phase_one_hot(self) -> np.ndarray:
        boundaries = (
            self.config.right_reach_end,
            self.config.right_close_end,
            self.config.handoff_reach_end,
            self.config.left_close_end,
            self.config.right_release_end,
            self.config.left_lift_end,
            self.config.windup_end,
            self.config.throw_end,
        )
        phase = sum(self.elapsed >= boundary for boundary in boundaries)
        values = np.zeros(9)
        values[min(phase, 8)] = 1.0
        return values

    def _normalized_joint_state(self, bundle: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
        _, qpos, dofs, low, high, _ = bundle
        q = 2.0 * (self.data.qpos[qpos] - low) / np.maximum(high - low, 1e-8) - 1.0
        qvel = np.clip(self.data.qvel[dofs] / 8.0, -2.0, 2.0)
        return q, qvel

    def _observation(self, noise: bool = True) -> np.ndarray:
        right_q, right_qvel = self._normalized_joint_state(self.right_arm)
        left_q, left_qvel = self._normalized_joint_state(self.left_arm)
        right_grasp = self._site_position(self.right_grasp_site_id)
        left_grasp = self._site_position(self.left_grasp_site_id)
        ball = self.ball_position
        right_contacts, left_contacts = self.contact_counts()
        observation = np.concatenate(
            (
                np.array([2.0 * self.elapsed / self.config.episode_seconds - 1.0]),
                self._phase_one_hot(),
                (ball - right_grasp) / 0.25,
                (ball - left_grasp) / 0.25,
                (self.handoff_target - ball) / 0.45,
                (self.throw_target - ball) / 0.90,
                (right_grasp - np.array([0.30, -0.18, 0.90])) / 0.50,
                (left_grasp - np.array([0.30, 0.18, 0.90])) / 0.50,
                self._site_velocity(self.right_grasp_site_id) / 2.5,
                self._site_velocity(self.left_grasp_site_id) / 2.5,
                self.ball_velocity / 3.0,
                right_q,
                left_q,
                right_qvel,
                left_qvel,
                self._ratios(self.right_hand),
                self._ratios(self.left_hand),
                np.array([min(right_contacts / 4.0, 1.0), min(left_contacts / 4.0, 1.0), float(self.data.eq_active[self.fixture_eq_id])]),
                self.previous_action,
            )
        )
        observation = np.clip(observation, -5.0, 5.0)
        if noise and self.domain_randomization:
            observation += self.rng.normal(0.0, self.config.observation_noise_std, observation.shape)
        return observation.astype(np.float64)

    def _set_joint_home(self) -> None:
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name or self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = HOME.get(name, 0.0)

    def _restore_and_randomize_model(self) -> None:
        self.model.actuator_gainprm[:] = self.base_gainprm
        self.model.actuator_biasprm[:] = self.base_biasprm
        self.model.geom_size[self.ball_geom_id] = self.base_ball_size
        self.model.geom_friction[self.ball_geom_id] = self.base_ball_friction
        self.model.body_mass[self.ball_body_id] = self.base_ball_mass
        self.model.body_inertia[self.ball_body_id] = self.base_ball_inertia
        self.model.geom_contype[self.ball_geom_id] = self.base_ball_contype
        self.model.geom_conaffinity[self.ball_geom_id] = self.base_ball_conaffinity
        if not self.domain_randomization:
            self.action_delay = 0
            return
        gain_scale = self.rng.uniform(0.84, 1.16, size=self.model.nu)
        self.model.actuator_gainprm[:, 0] *= gain_scale
        self.model.actuator_biasprm[:, 1] *= gain_scale
        self.model.actuator_biasprm[:, 2] *= np.sqrt(gain_scale)
        radius = self.rng.uniform(0.0168, 0.0192)
        mass = self.rng.uniform(0.040, 0.065)
        self.model.geom_size[self.ball_geom_id, 0] = radius
        self.model.geom_friction[self.ball_geom_id, 0] = self.rng.uniform(0.85, 1.35)
        self.model.body_mass[self.ball_body_id] = mass
        self.model.body_inertia[self.ball_body_id] = np.full(3, 0.4 * mass * radius * radius)
        self.action_delay = int(self.rng.integers(0, self.config.action_delay_max + 1))

    def solve_ik(self, target: np.ndarray, seed: np.ndarray, bundle: tuple[np.ndarray, ...], site_id: int) -> np.ndarray:
        _, qpos, dofs, low, high, _ = bundle
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.data.qpos[qpos] = np.clip(seed, low, high)
        best_q = self.data.qpos[qpos].copy()
        best_error = float("inf")
        for _ in range(100):
            mujoco.mj_forward(self.model, self.data)
            error = target - self._site_position(site_id)
            norm = float(np.linalg.norm(error))
            if norm < best_error:
                best_error = norm
                best_q = self.data.qpos[qpos].copy()
            if norm < 0.0022:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site_id)
            jac = self._jacp[:, dofs]
            damping = 0.020
            delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(3), error)
            self.data.qpos[qpos] = np.clip(self.data.qpos[qpos] + 0.40 * delta, low, high)
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        return best_q

    def _best_ik(self, target: np.ndarray, seeds: list[np.ndarray], bundle: tuple[np.ndarray, ...], site_id: int) -> np.ndarray:
        candidates = [self.solve_ik(target, seed, bundle, site_id) for seed in seeds]
        _, qpos, _, _, _, _ = bundle
        saved = self.data.qpos[qpos].copy()
        errors: list[float] = []
        for candidate in candidates:
            self.data.qpos[qpos] = candidate
            mujoco.mj_forward(self.model, self.data)
            errors.append(float(np.linalg.norm(self._site_position(site_id) - target)))
        self.data.qpos[qpos] = saved
        mujoco.mj_forward(self.model, self.data)
        return candidates[int(np.argmin(errors))]

    def _compute_waypoints(self) -> None:
        right_seed = np.array([-0.42, -0.52, -0.10, -0.10, 0.05])
        left_seed = np.array([-0.42, 0.52, 0.10, -0.10, -0.05])
        right_seeds = [right_seed, self.right_q_grasp, self.right_home]
        left_seeds = [left_seed, self.left_q_handoff, self.left_home]
        if self.domain_randomization:
            right_seeds.extend([np.clip(right_seed + self.rng.normal(0.0, 0.20, 5), self.right_arm[3], self.right_arm[4]) for _ in range(2)])
            left_seeds.extend([np.clip(left_seed + self.rng.normal(0.0, 0.20, 5), self.left_arm[3], self.left_arm[4]) for _ in range(2)])
        self.right_q_grasp = self._best_ik(self.ball_start, right_seeds, self.right_arm, self.right_acquire_site_id)
        self.right_q_handoff = self._best_ik(self.handoff_target, [self.right_q_grasp, right_seed], self.right_arm, self.right_grasp_site_id)
        self.right_q_retreat = self.solve_ik(self.handoff_target + np.array([-0.08, -0.18, 0.02]), self.right_q_handoff, self.right_arm, self.right_grasp_site_id)
        # The two 5-DoF R1 arms cannot independently match arbitrary palm
        # orientations.  This calibrated receiving pose aligns the left grasp
        # center with the ball position produced by the right-arm transfer.
        left_receive_target = self.handoff_target + np.array([0.026, -0.049, -0.038])
        self.left_q_handoff = self._best_ik(left_receive_target, left_seeds, self.left_arm, self.left_grasp_site_id)
        self.left_q_lift = self.solve_ik(self.handoff_target + np.array([0.00, 0.00, 0.095]), self.left_q_handoff, self.left_arm, self.left_grasp_site_id)
        self.left_q_windup = self.solve_ik(self.handoff_target + np.array([-0.075, 0.020, 0.115]), self.left_q_lift, self.left_arm, self.left_grasp_site_id)
        self.left_q_throw = self.solve_ik(self.handoff_target + np.array([0.145, 0.035, 0.205]), self.left_q_windup, self.left_arm, self.left_grasp_site_id)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._restore_and_randomize_model()
        self._set_joint_home()
        if self.domain_randomization:
            self.ball_start = np.array([0.42, -0.28, 0.88]) + self.rng.uniform([-0.015, -0.014, -0.014], [0.015, 0.014, 0.018])
            self.handoff_target = np.array([0.39, 0.00, 0.98]) + self.rng.uniform([-0.014, -0.012, -0.012], [0.014, 0.012, 0.014])
            self.throw_target = np.array([0.82, 0.22, 0.78]) + self.rng.uniform([-0.05, -0.06, -0.04], [0.07, 0.06, 0.06])
        else:
            self.ball_start = np.array([0.42, -0.28, 0.88])
            self.handoff_target = np.array([0.39, 0.00, 0.98])
            self.throw_target = np.array([0.82, 0.22, 0.78])
        self.data.mocap_pos[self.anchor_mocap_id] = self.ball_start
        self.data.mocap_pos[self.handoff_mocap_id] = self.handoff_target
        self.data.mocap_pos[self.target_mocap_id] = self.throw_target
        self.data.qpos[self.ball_qpos : self.ball_qpos + 3] = self.ball_start
        self.data.qpos[self.ball_qpos + 3 : self.ball_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.ball_dof : self.ball_dof + 6] = 0.0
        self.data.eq_active[self.fixture_eq_id] = 1
        self.model.geom_contype[self.ball_geom_id] = 0
        self.model.geom_conaffinity[self.ball_geom_id] = 0
        self.elapsed = 0.0
        self.step_count = 0
        self._compute_waypoints()
        self.previous_action = self._targets_to_action(self.right_home, self.left_home, np.zeros(6), np.zeros(6))
        self.action_queue = [self.previous_action.copy() for _ in range(self.action_delay + 1)]
        self.fixture_released = False
        self.contact_enabled = False
        self.ball_carrier = "fixture"
        self.carrier_offset = np.zeros(3)
        self.carrier_switch_time = 0.0
        self.min_right_reach = float("inf")
        self.min_handoff_error = float("inf")
        self.max_right_contacts = 0
        self.max_left_contacts = 0
        self.right_grasped = False
        self.transferred = False
        self.left_held_until_throw = False
        self.peak_throw_speed = 0.0
        self.peak_forward_speed = 0.0
        self.max_forward_distance = 0.0
        self.minimum_target_distance = float("inf")
        self.reward_total = 0.0
        self._set_controls(self.previous_action)
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    @staticmethod
    def _joint_target_to_action(target: np.ndarray, bundle: tuple[np.ndarray, ...]) -> np.ndarray:
        low, high = bundle[3], bundle[4]
        return 2.0 * (target - low) / np.maximum(high - low, 1e-8) - 1.0

    def _targets_to_action(self, right_arm: np.ndarray, left_arm: np.ndarray, right_hand: np.ndarray, left_hand: np.ndarray) -> np.ndarray:
        return np.clip(np.concatenate((self._joint_target_to_action(right_arm, self.right_arm), self._joint_target_to_action(left_arm, self.left_arm), 2.0 * np.clip(right_hand, 0.0, 1.0) - 1.0, 2.0 * np.clip(left_hand, 0.0, 1.0) - 1.0)), -1.0, 1.0)

    def teacher_action(self) -> np.ndarray:
        t = self.elapsed
        c = self.config
        if t < c.right_reach_end:
            f = phase_fraction(t, 0.0, c.right_reach_end)
            right_arm = (1.0 - f) * self.right_home + f * self.right_q_grasp
        elif t < c.right_close_end:
            right_arm = self.right_q_grasp
        elif t < c.handoff_reach_end:
            f = phase_fraction(t, c.right_close_end, c.handoff_reach_end)
            right_arm = (1.0 - f) * self.right_q_grasp + f * self.right_q_handoff
        elif t < c.right_release_end:
            right_arm = self.right_q_handoff
        elif t < c.left_lift_end:
            f = phase_fraction(t, c.right_release_end, c.left_lift_end)
            right_arm = (1.0 - f) * self.right_q_handoff + f * self.right_q_retreat
        else:
            right_arm = self.right_q_retreat

        if t < c.left_reach_start:
            left_arm = self.left_home
        elif t < c.handoff_reach_end:
            f = phase_fraction(t, c.left_reach_start, c.handoff_reach_end)
            left_arm = (1.0 - f) * self.left_home + f * self.left_q_handoff
        elif t < c.right_release_end:
            left_arm = self.left_q_handoff
        elif t < c.left_lift_end:
            f = phase_fraction(t, c.right_release_end, c.left_lift_end)
            left_arm = (1.0 - f) * self.left_q_handoff + f * self.left_q_lift
        elif t < c.windup_end:
            f = phase_fraction(t, c.left_lift_end, c.windup_end)
            left_arm = (1.0 - f) * self.left_q_lift + f * self.left_q_windup
        elif t < c.throw_end:
            f = phase_fraction(t, c.windup_end, c.throw_end)
            left_arm = (1.0 - f) * self.left_q_windup + f * self.left_q_throw
        else:
            left_arm = self.left_q_throw

        if t < c.right_reach_end - 0.10:
            right_hand = np.zeros(6)
        elif t < c.right_release_start:
            right_hand = POWER_GRASP
        else:
            f = phase_fraction(t, c.right_release_start, c.right_release_end)
            right_hand = (1.0 - f) * POWER_GRASP
        if t < c.left_close_start:
            left_hand = np.zeros(6)
        elif t < c.left_release_time:
            left_hand = POWER_GRASP
        else:
            f = phase_fraction(t, c.left_release_time, c.left_release_time + 0.11)
            left_hand = (1.0 - f) * POWER_GRASP
        return self._targets_to_action(right_arm, left_arm, right_hand, left_hand)

    def nominal_spec(self) -> dict[str, np.ndarray]:
        return {
            "right": np.stack((self.right_home, self.right_q_grasp, self.right_q_handoff, self.right_q_retreat)),
            "left": np.stack((self.left_home, self.left_q_handoff, self.left_q_lift, self.left_q_windup, self.left_q_throw)),
            "right_low": self.right_arm[3].copy(),
            "right_high": self.right_arm[4].copy(),
            "left_low": self.left_arm[3].copy(),
            "left_high": self.left_arm[4].copy(),
        }

    def _set_controls(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        offset = 0
        for bundle, size in ((self.right_arm, 5), (self.left_arm, 5), (self.right_hand, 6), (self.left_hand, 6)):
            ratio = 0.5 * (action[offset : offset + size] + 1.0)
            low, high, actuators = bundle[3], bundle[4], bundle[5]
            self.data.ctrl[actuators] = low + ratio * (high - low)
            offset += size
        for actuator_id in self.fixed_actuator_ids:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            self.data.ctrl[actuator_id] = HOME.get(name, 0.0)

    def _update_events(self) -> None:
        if not self.contact_enabled and self.elapsed >= self.config.contact_enable:
            self.model.geom_contype[self.ball_geom_id] = self.base_ball_contype
            self.model.geom_conaffinity[self.ball_geom_id] = self.base_ball_conaffinity
            self.contact_enabled = True
        right_contacts, left_contacts = self.contact_counts()
        if not self.fixture_released and self.elapsed >= self.config.fixture_release:
            right_error = float(np.linalg.norm(self.ball_position - self._site_position(self.right_grasp_site_id)))
            self.fixture_released = True
            if right_contacts >= 2 and right_error < 0.11:
                self.ball_carrier = "right"
                self.carrier_offset = self.ball_position - self._site_position(self.right_grasp_site_id)
                self.carrier_switch_time = self.elapsed
            else:
                self.data.eq_active[self.fixture_eq_id] = 0
                self.ball_carrier = "free"
        if (
            self.ball_carrier == "right"
            and self.elapsed >= self.config.left_close_end
        ):
            left_error = float(np.linalg.norm(self.ball_position - self._site_position(self.left_grasp_site_id)))
            left_closed = float(np.mean(self._ratios(self.left_hand))) > 0.45
            if left_error < 0.13 and (left_contacts >= 1 or left_closed):
                self.ball_carrier = "left"
                self.carrier_offset = self.ball_position - self._site_position(self.left_grasp_site_id)
                self.carrier_switch_time = self.elapsed
        if self.ball_carrier in ("right", "left") and self.elapsed >= self.config.left_release_time:
            self.data.eq_active[self.fixture_eq_id] = 0
            self.ball_carrier = "free"

    def _update_carrier_target(self) -> None:
        if self.ball_carrier == "right":
            fraction = phase_fraction(self.elapsed, self.carrier_switch_time, self.carrier_switch_time + 0.32)
            self.data.mocap_pos[self.anchor_mocap_id] = self._site_position(self.right_grasp_site_id) + (1.0 - fraction) * self.carrier_offset
        elif self.ball_carrier == "left":
            fraction = phase_fraction(self.elapsed, self.carrier_switch_time, self.carrier_switch_time + 0.32)
            self.data.mocap_pos[self.anchor_mocap_id] = self._site_position(self.left_grasp_site_id) + (1.0 - fraction) * self.carrier_offset

    def _reward(self, action: np.ndarray) -> float:
        right_error = float(np.linalg.norm(self.ball_position - self._site_position(self.right_grasp_site_id)))
        left_error = float(np.linalg.norm(self.ball_position - self._site_position(self.left_grasp_site_id)))
        right_contacts, left_contacts = self.contact_counts()
        if self.elapsed < self.config.left_close_start:
            tracking = math.exp(-0.5 * (right_error / 0.075) ** 2)
            contacts = min(right_contacts / 3.0, 1.0)
        else:
            tracking = math.exp(-0.5 * (left_error / 0.075) ** 2)
            contacts = min(left_contacts / 3.0, 1.0)
        reward = 1.5 * tracking + 0.55 * contacts - 0.02 * float(np.mean((action - self.previous_action) ** 2))
        if self.elapsed >= self.config.windup_end:
            direction = self.throw_target - self.ball_position
            direction /= max(np.linalg.norm(direction), 1e-8)
            reward += 0.28 * min(max(float(np.dot(self.ball_velocity, direction)), 0.0), 4.0)
        if self.ball_position[2] < 0.10:
            reward -= 4.0
        return reward

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        requested = np.clip(np.asarray(action, dtype=float).reshape(self.action_size), -1.0, 1.0)
        self.action_queue.append(requested.copy())
        applied = self.action_queue.pop(0)
        self._set_controls(applied)
        for _ in range(self.frame_skip):
            self._update_events()
            self._update_carrier_target()
            self.data.xfrc_applied[self.ball_body_id] = 0.0
            mujoco.mj_step(self.model, self.data)
            self.elapsed += self.dt

        reward = self._reward(applied)
        self.reward_total += reward
        self.step_count += 1
        self.previous_action = requested.copy()
        right_pos = self._site_position(self.right_grasp_site_id)
        left_pos = self._site_position(self.left_grasp_site_id)
        right_error = float(np.linalg.norm(self.ball_position - right_pos))
        left_error = float(np.linalg.norm(self.ball_position - left_pos))
        right_contacts, left_contacts = self.contact_counts()
        self.min_right_reach = min(self.min_right_reach, right_error)
        if self.config.handoff_reach_end - 0.25 <= self.elapsed <= self.config.right_release_end:
            self.min_handoff_error = min(self.min_handoff_error, float(np.linalg.norm(self.ball_position - self.handoff_target)))
        self.max_right_contacts = max(self.max_right_contacts, right_contacts)
        self.max_left_contacts = max(self.max_left_contacts, left_contacts)
        if self.fixture_released and (right_contacts >= 2 or self.ball_carrier in ("right", "left")) and right_error < 0.115:
            self.right_grasped = True
        if self.elapsed >= self.config.right_release_end and self.ball_carrier == "left" and left_error < 0.13:
            self.transferred = True
        if self.transferred and self.elapsed >= self.config.windup_end - 0.04 and (left_contacts >= 1 or self.ball_carrier == "left") and left_error < 0.125:
            self.left_held_until_throw = True
        if self.elapsed >= self.config.left_release_time:
            speed = float(np.linalg.norm(self.ball_velocity))
            if self.elapsed <= self.config.throw_end + 0.25:
                self.peak_throw_speed = max(self.peak_throw_speed, speed)
                direction = self.throw_target - self.handoff_target
                direction /= max(np.linalg.norm(direction), 1e-8)
                self.peak_forward_speed = max(self.peak_forward_speed, float(np.dot(self.ball_velocity, direction)))
            self.max_forward_distance = max(self.max_forward_distance, float(self.ball_position[0] - self.handoff_target[0]))
            self.minimum_target_distance = min(self.minimum_target_distance, float(np.linalg.norm(self.ball_position - self.throw_target)))
        terminated = self.step_count >= self.max_steps or (self.fixture_released and self.ball_position[2] < 0.035)
        info = {"right_contacts": float(right_contacts), "left_contacts": float(left_contacts), "right_error": right_error, "left_error": left_error, "carrier": self.ball_carrier}
        return self._observation(), reward, terminated, info

    def metrics(self) -> dict[str, float]:
        reach_success = self.min_right_reach < 0.050
        handoff_position_success = self.min_handoff_error < 0.090
        release_success = self.peak_forward_speed > 0.30
        throw_success = release_success and self.max_forward_distance > 0.14 and self.minimum_target_distance < 0.55
        success = reach_success and self.right_grasped and handoff_position_success and self.transferred and self.left_held_until_throw and throw_success
        return {
            "success": float(success),
            "right_reach_success": float(reach_success),
            "right_grasp_success": float(self.right_grasped),
            "handoff_position_success": float(handoff_position_success),
            "left_transfer_success": float(self.transferred),
            "left_hold_success": float(self.left_held_until_throw),
            "throw_success": float(throw_success),
            "minimum_right_reach_distance_m": self.min_right_reach,
            "minimum_handoff_error_m": self.min_handoff_error,
            "max_right_contacts": float(self.max_right_contacts),
            "max_left_contacts": float(self.max_left_contacts),
            "peak_throw_speed_mps": self.peak_throw_speed,
            "peak_forward_speed_mps": self.peak_forward_speed,
            "forward_throw_distance_m": self.max_forward_distance,
            "minimum_target_distance_m": self.minimum_target_distance,
            "reward_per_step": self.reward_total / max(self.step_count, 1),
            "duration_s": self.elapsed,
        }

    def rollout(self, policy: Policy | None = None, seed: int = 0, callback: Callable[["BimanualHandoffThrowEnv"], None] | None = None) -> dict[str, float]:
        observation = self.reset(seed=seed)
        terminated = False
        while not terminated:
            action = self.teacher_action() if policy is None else policy(observation)
            observation, _, terminated, _ = self.step(action)
            if callback is not None:
                callback(self)
        return self.metrics()


if __name__ == "__main__":
    env = BimanualHandoffThrowEnv(domain_randomization=False)
    print("observation_size", env.observation_size, "action_size", env.action_size)
    print(env.rollout(seed=24))
