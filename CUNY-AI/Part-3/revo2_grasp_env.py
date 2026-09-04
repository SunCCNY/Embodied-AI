#!/usr/bin/env python3
"""Ball-grasp training environment for the BrainCo Revo2 Basic hand.

The environment intentionally has no Gymnasium dependency.  It exposes the
small reset/step/rollout interface needed by the bundled NumPy CEM trainer,
while keeping the MuJoCo task usable from other learning frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol

import mujoco
import numpy as np

from revo2_sim import ACTUATOR_SUFFIXES, POSES


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCENES = {
    "right": ROOT / "vendor/Revo2_xml/xml_right/scene_train.xml",
    "left": ROOT / "vendor/Revo2_xml/xml_left/scene_train.xml",
}

FEATURE_NAMES = (
    "target_error_x",
    "target_error_y",
    "target_error_z",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "target_velocity_x",
    "target_velocity_y",
    "target_velocity_z",
    "ball_angular_velocity_x",
    "ball_angular_velocity_y",
    "ball_angular_velocity_z",
    "thumb_opposition_ratio",
    "thumb_flexion_ratio",
    "index_flexion_ratio",
    "middle_flexion_ratio",
    "ring_flexion_ratio",
    "pinky_flexion_ratio",
    "gravity_fraction",
)


class Policy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return six actuator commands normalized to the interval [0, 1]."""


@dataclass(frozen=True)
class GraspTaskConfig:
    task: str = "hold"
    episode_seconds: float = 4.5
    control_hz: float = 50.0
    acquire_seconds: float = 1.50
    gravity_ramp_seconds: float = 0.50
    actuator_slew_seconds: float = 1.50
    move_amplitude: float = 0.006
    move_period_seconds: float = 2.0
    disturbance_force: float = 0.06
    disturbance_seconds: float = 0.12

    def __post_init__(self) -> None:
        if self.task not in {"hold", "move"}:
            raise ValueError("task must be 'hold' or 'move'")
        if self.episode_seconds <= 0 or self.control_hz <= 0:
            raise ValueError("episode_seconds and control_hz must be positive")


class Revo2BallGraspEnv:
    """Closed-loop MuJoCo task for acquiring and retaining a free ball."""

    observation_size = len(FEATURE_NAMES)
    action_size = len(ACTUATOR_SUFFIXES)

    def __init__(
        self,
        hand: str = "right",
        config: GraspTaskConfig | None = None,
    ) -> None:
        if hand not in TRAINING_SCENES:
            raise ValueError("hand must be 'right' or 'left'")

        self.hand = hand
        self.config = config or GraspTaskConfig()
        self.model = mujoco.MjModel.from_xml_path(str(TRAINING_SCENES[hand]))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.frame_skip = max(
            1, int(round(1.0 / (self.config.control_hz * self.dt)))
        )
        self.control_dt = self.frame_skip * self.dt
        self.max_control_steps = int(
            math.ceil(self.config.episode_seconds / self.control_dt)
        )

        actuator_names = [f"{hand}_{suffix}" for suffix in ACTUATOR_SUFFIXES]
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                for name in actuator_names
            ],
            dtype=int,
        )
        if np.any(self.actuator_ids < 0):
            raise RuntimeError(
                f"Missing expected {hand}-hand actuators: {actuator_names}"
            )

        self.ctrl_low = self.model.actuator_ctrlrange[
            self.actuator_ids, 0
        ].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[
            self.actuator_ids, 1
        ].copy()

        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0].astype(int)
        self.actuated_qpos_adr = self.model.jnt_qposadr[joint_ids].astype(int)
        joint_range = self.model.jnt_range[joint_ids]
        self.joint_low = joint_range[:, 0].copy()
        self.joint_high = joint_range[:, 1].copy()

        self.ball_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, "grasp_object"
        )
        self.ball_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, "grasp_object_geom"
        )
        ball_joint_id = int(self.model.body_jntadr[self.ball_body_id])
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint_id])
        self.ball_dof_adr = int(self.model.jnt_dofadr[ball_joint_id])

        self.target_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, "grasp_target"
        )
        self.target_mocap_id = int(
            self.model.body_mocapid[self.target_body_id]
        )
        if self.target_mocap_id < 0:
            raise RuntimeError("grasp_target must be a mocap body")

        excluded = {
            self.ball_geom_id,
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "scene_floor"),
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "grasp_target_geom"),
        }
        self.hand_geom_ids = set(range(self.model.ngeom)) - excluded

        nominal_ball = self.model.qpos0[
            self.ball_qpos_adr : self.ball_qpos_adr + 3
        ].copy()
        # The official power grasp settles the sphere about 22 mm deeper into
        # the palm than its open-hand spawn point.
        self.hold_center = nominal_ball.copy()
        self.hold_center[0] *= 0.46
        self.hold_center[2] -= 0.023

        self.rng = np.random.default_rng(0)
        self.elapsed = 0.0
        self.control_step = 0
        self.previous_action = POSES["open"].copy()
        self.disturbance_direction = np.array([1.0, 0.0, 0.0])
        self.reward_total = 0.0
        self.hold_distance_total = 0.0
        self.hold_samples = 0
        self.max_contacts = 0
        self.dropped = False
        self.move_ball_x_min = float("inf")
        self.move_ball_x_max = float("-inf")
        self.move_target_x_min = float("inf")
        self.move_target_x_max = float("-inf")
        self.move_error_squared = 0.0
        self.move_samples = 0
        self.reset(seed=0)

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"Model is missing required object: {name}")
        return int(object_id)

    def ratio_to_ctrl(self, ratio: np.ndarray) -> np.ndarray:
        ratio = np.asarray(ratio, dtype=float)
        return self.ctrl_low + np.clip(ratio, 0.0, 1.0) * (
            self.ctrl_high - self.ctrl_low
        )

    def gravity_fraction(self, elapsed: float | None = None) -> float:
        elapsed = self.elapsed if elapsed is None else elapsed
        if elapsed <= self.config.acquire_seconds:
            return 0.0
        ramp_time = elapsed - self.config.acquire_seconds
        return float(
            np.clip(ramp_time / self.config.gravity_ramp_seconds, 0.0, 1.0)
        )

    def target_position(self, elapsed: float | None = None) -> np.ndarray:
        elapsed = self.elapsed if elapsed is None else elapsed
        target = self.hold_center.copy()
        if self.config.task == "move":
            move_start = (
                self.config.acquire_seconds
                + self.config.gravity_ramp_seconds
                + 0.25
            )
            if elapsed > move_start:
                phase = (
                    2.0
                    * math.pi
                    * (elapsed - move_start)
                    / self.config.move_period_seconds
                )
                target[0] += self.config.move_amplitude * math.sin(phase)
        return target

    def movement_started(self, elapsed: float | None = None) -> bool:
        elapsed = self.elapsed if elapsed is None else elapsed
        move_start = (
            self.config.acquire_seconds
            + self.config.gravity_ramp_seconds
            + 0.25
        )
        return self.config.task == "move" and elapsed > move_start

    def target_velocity(self, elapsed: float | None = None) -> np.ndarray:
        elapsed = self.elapsed if elapsed is None else elapsed
        velocity = np.zeros(3)
        if self.config.task == "move":
            move_start = (
                self.config.acquire_seconds
                + self.config.gravity_ramp_seconds
                + 0.25
            )
            if elapsed > move_start:
                omega = 2.0 * math.pi / self.config.move_period_seconds
                phase = omega * (elapsed - move_start)
                velocity[0] = (
                    self.config.move_amplitude * omega * math.cos(phase)
                )
        return velocity

    @property
    def ball_position(self) -> np.ndarray:
        return self.data.qpos[
            self.ball_qpos_adr : self.ball_qpos_adr + 3
        ].copy()

    @property
    def ball_velocity(self) -> np.ndarray:
        return self.data.qvel[
            self.ball_dof_adr : self.ball_dof_adr + 3
        ].copy()

    @property
    def ball_angular_velocity(self) -> np.ndarray:
        return self.data.qvel[
            self.ball_dof_adr + 3 : self.ball_dof_adr + 6
        ].copy()

    def joint_ratios(self) -> np.ndarray:
        qpos = self.data.qpos[self.actuated_qpos_adr]
        span = np.maximum(self.joint_high - self.joint_low, 1e-8)
        return np.clip((qpos - self.joint_low) / span, 0.0, 1.0)

    def contact_count(self) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 == self.ball_geom_id:
                other = int(contact.geom2)
            elif contact.geom2 == self.ball_geom_id:
                other = int(contact.geom1)
            else:
                continue
            if other in self.hand_geom_ids:
                count += 1
        return count

    def observation(self) -> np.ndarray:
        target = self.target_position()
        target_velocity = self.target_velocity()
        observation = np.concatenate(
            (
                (target - self.ball_position) / 0.03,
                self.ball_velocity / 0.50,
                target_velocity / 0.03,
                self.ball_angular_velocity / 10.0,
                self.joint_ratios(),
                np.array([self.gravity_fraction()]),
            )
        )
        return np.clip(observation, -5.0, 5.0).astype(np.float64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        initial_ball = self.model.qpos0[
            self.ball_qpos_adr : self.ball_qpos_adr + 3
        ].copy()
        # The sphere begins in a narrow but non-zero spawn distribution.  A
        # wider reset distribution tends to place this 36 mm sphere inside
        # the official collision meshes, producing non-physical launch
        # impulses before the policy can react.
        initial_ball += self.rng.normal(
            loc=0.0, scale=np.array([0.00035, 0.00035, 0.00025])
        )
        self.data.qpos[
            self.ball_qpos_adr : self.ball_qpos_adr + 3
        ] = initial_ball
        self.data.qpos[
            self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7
        ] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[
            self.ball_dof_adr : self.ball_dof_adr + 6
        ] = self.rng.normal(0.0, 0.0005, size=6)

        direction = self.rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-8)
        self.disturbance_direction = np.array(
            [direction[0], direction[1], 0.15]
        )
        self.disturbance_direction /= np.linalg.norm(
            self.disturbance_direction
        )

        self.model.opt.gravity[:] = 0.0
        self.data.ctrl[self.actuator_ids] = self.ratio_to_ctrl(POSES["open"])
        self.data.xfrc_applied[:] = 0.0
        self.elapsed = 0.0
        self.control_step = 0
        self.previous_action = POSES["open"].copy()
        self.reward_total = 0.0
        self.hold_distance_total = 0.0
        self.hold_samples = 0
        self.max_contacts = 0
        self.dropped = False
        self.move_ball_x_min = float("inf")
        self.move_ball_x_max = float("-inf")
        self.move_target_x_min = float("inf")
        self.move_target_x_max = float("-inf")
        self.move_error_squared = 0.0
        self.move_samples = 0
        self._update_target_marker()
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def _update_target_marker(self) -> None:
        self.data.mocap_pos[self.target_mocap_id] = self.target_position()
        self.data.mocap_quat[self.target_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

    def _disturbance_active(self) -> bool:
        start = (
            self.config.acquire_seconds
            + self.config.gravity_ramp_seconds
            + 0.55
        )
        return start <= self.elapsed < start + self.config.disturbance_seconds

    def _physics_step(self, action: np.ndarray) -> None:
        gravity_fraction = self.gravity_fraction()
        self.model.opt.gravity[:] = np.array(
            [0.0, 0.0, -9.81 * gravity_fraction]
        )

        desired_ctrl = self.ratio_to_ctrl(action)
        current_ctrl = self.data.ctrl[self.actuator_ids]
        max_delta = (
            (self.ctrl_high - self.ctrl_low)
            * POSES["power"]
            * self.dt
            / self.config.actuator_slew_seconds
        )
        self.data.ctrl[self.actuator_ids] = current_ctrl + np.clip(
            desired_ctrl - current_ctrl, -max_delta, max_delta
        )

        self.data.xfrc_applied[:] = 0.0
        if self._disturbance_active():
            self.data.xfrc_applied[self.ball_body_id, :3] = (
                self.config.disturbance_force * self.disturbance_direction
            )

        self._update_target_marker()
        mujoco.mj_step(self.model, self.data)
        self.elapsed += self.dt

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        target = self.target_position()
        distance = float(np.linalg.norm(self.ball_position - target))
        speed = float(np.linalg.norm(self.ball_velocity))
        contacts = self.contact_count()
        track_reward = math.exp(-0.5 * (distance / 0.024) ** 2)
        contact_reward = min(contacts / 4.0, 1.0)
        smoothness = float(np.mean((action - self.previous_action) ** 2))

        reward = (
            3.0 * track_reward
            + 0.45 * contact_reward
            - 0.08 * min(speed, 2.0)
            - 0.04 * smoothness
        )
        if self.movement_started():
            x_error = float(self.ball_position[0] - target[0])
            directional_reward = math.exp(
                -0.5 * (x_error / 0.006) ** 2
            )
            reward += 1.5 * directional_reward
            self.move_ball_x_min = min(
                self.move_ball_x_min, float(self.ball_position[0])
            )
            self.move_ball_x_max = max(
                self.move_ball_x_max, float(self.ball_position[0])
            )
            self.move_target_x_min = min(
                self.move_target_x_min, float(target[0])
            )
            self.move_target_x_max = max(
                self.move_target_x_max, float(target[0])
            )
            self.move_error_squared += x_error * x_error
            self.move_samples += 1
        gravity_on = self.gravity_fraction() >= 0.99
        if gravity_on:
            self.hold_distance_total += distance
            self.hold_samples += 1

        distance_from_palm = float(
            np.linalg.norm(self.ball_position - self.hold_center)
        )
        dropped = (
            gravity_on
            and (
                self.ball_position[2] < 0.005
                or distance_from_palm > 0.14
            )
        )
        if dropped:
            reward -= 30.0

        self.max_contacts = max(self.max_contacts, contacts)
        return reward, {
            "distance": distance,
            "speed": speed,
            "contacts": float(contacts),
            "track_reward": track_reward,
            "dropped": float(dropped),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        action = np.clip(
            np.asarray(action, dtype=float), 0.0, 1.0
        ).reshape(self.action_size)
        for _ in range(self.frame_skip):
            self._physics_step(action)

        reward, info = self._reward(action)
        self.reward_total += reward
        self.control_step += 1
        self.previous_action = action.copy()
        self.dropped = bool(info["dropped"])
        terminated = self.dropped or self.control_step >= self.max_control_steps
        return self.observation(), reward, terminated, info

    def metrics(self) -> dict[str, float]:
        mean_hold_distance = (
            self.hold_distance_total / self.hold_samples
            if self.hold_samples
            else float("inf")
        )
        final_distance = float(
            np.linalg.norm(self.ball_position - self.target_position())
        )
        final_contacts = self.contact_count()
        completed = self.control_step >= self.max_control_steps
        move_ball_range = (
            self.move_ball_x_max - self.move_ball_x_min
            if self.move_samples
            else 0.0
        )
        move_target_range = (
            self.move_target_x_max - self.move_target_x_min
            if self.move_samples
            else 0.0
        )
        move_tracking_rmse = (
            math.sqrt(self.move_error_squared / self.move_samples)
            if self.move_samples
            else 0.0
        )
        movement_success = (
            self.config.task != "move"
            or (
                move_ball_range >= 0.002
                and move_tracking_rmse < 0.008
            )
        )
        success = (
            completed
            and not self.dropped
            and mean_hold_distance < 0.040
            and final_distance < 0.050
            and final_contacts >= 2
            and movement_success
        )
        return {
            "score": self.reward_total / max(self.control_step, 1),
            "success": float(success),
            "dropped": float(self.dropped),
            "mean_hold_distance": mean_hold_distance,
            "final_distance": final_distance,
            "final_contacts": float(final_contacts),
            "max_contacts": float(self.max_contacts),
            "duration": self.elapsed,
            "move_ball_range": move_ball_range,
            "move_target_range": move_target_range,
            "move_tracking_rmse": move_tracking_rmse,
        }

    def rollout(
        self,
        policy: Policy,
        seed: int,
        callback: Callable[["Revo2BallGraspEnv"], None] | None = None,
    ) -> dict[str, float]:
        observation = self.reset(seed=seed)
        terminated = False
        while not terminated:
            action = policy(observation)
            observation, _, terminated, _ = self.step(action)
            if callback is not None:
                callback(self)
        return self.metrics()
