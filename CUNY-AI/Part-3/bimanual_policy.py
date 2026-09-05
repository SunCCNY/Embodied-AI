#!/usr/bin/env python3
"""NumPy residual neural policy for the bimanual MuJoCo task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bimanual_env import POWER_GRASP, TaskConfig


class BimanualNeuralPolicy:
    """Two-hidden-layer tanh MLP that corrects a safe central trajectory."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        nominal_right: np.ndarray,
        nominal_left: np.ndarray,
        hidden_size: int = 160,
        seed: int = 24,
        residual_arm_scale: float = 0.24,
        residual_hand_scale: float = 0.10,
    ) -> None:
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        self.nominal_right = np.asarray(nominal_right, dtype=np.float64).reshape(4, 5)
        self.nominal_left = np.asarray(nominal_left, dtype=np.float64).reshape(5, 5)
        self.residual_arm_scale = float(residual_arm_scale)
        self.residual_hand_scale = float(residual_hand_scale)
        rng = np.random.default_rng(seed)

        def initialize(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.parameters = {
            "w1": initialize(self.observation_size, self.hidden_size),
            "b1": np.zeros(self.hidden_size),
            "w2": initialize(self.hidden_size, self.hidden_size),
            "b2": np.zeros(self.hidden_size),
            "w3": initialize(self.hidden_size, self.action_size),
            "b3": np.zeros(self.action_size),
        }
        self.observation_mean = np.zeros(self.observation_size)
        self.observation_std = np.ones(self.observation_size)
        self._adam_m = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_step = 0

    @classmethod
    def from_environment(cls, environment, hidden_size: int = 160, seed: int = 24) -> "BimanualNeuralPolicy":
        spec = environment.nominal_spec()
        right = 2.0 * (spec["right"] - spec["right_low"]) / (spec["right_high"] - spec["right_low"]) - 1.0
        left = 2.0 * (spec["left"] - spec["left_low"]) / (spec["left_high"] - spec["left_low"]) - 1.0
        return cls(environment.observation_size, environment.action_size, right, left, hidden_size=hidden_size, seed=seed)

    def set_normalization(self, observations: np.ndarray) -> None:
        self.observation_mean = np.mean(observations, axis=0)
        self.observation_std = np.maximum(np.std(observations, axis=0), 0.03)

    def _forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = (observations - self.observation_mean) / self.observation_std
        h1 = np.tanh(x @ self.parameters["w1"] + self.parameters["b1"])
        h2 = np.tanh(h1 @ self.parameters["w2"] + self.parameters["b2"])
        latent = np.tanh(h2 @ self.parameters["w3"] + self.parameters["b3"])
        return x, h1, h2, latent

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observations = np.asarray(observation, dtype=np.float64).reshape(1, self.observation_size)
        return self.decode_actions(observations, self._forward(observations)[-1])[0]

    def predict_latent(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64).reshape(-1, self.observation_size)
        return self._forward(observations)[-1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64).reshape(-1, self.observation_size)
        return self.decode_actions(observations, self._forward(observations)[-1])

    def encode_targets(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        nominal = self.nominal_actions(observations)
        scale = np.array([self.residual_arm_scale] * 10 + [self.residual_hand_scale] * 12)
        return np.clip((actions - nominal) / scale[None, :], -1.0, 1.0)

    def decode_actions(self, observations: np.ndarray, latent: np.ndarray) -> np.ndarray:
        nominal = self.nominal_actions(observations)
        scale = np.array([self.residual_arm_scale] * 10 + [self.residual_hand_scale] * 12)
        return np.clip(nominal + scale[None, :] * latent, -1.0, 1.0)

    @staticmethod
    def _smoothstep(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, 0.0, 1.0)
        return values * values * (3.0 - 2.0 * values)

    def nominal_actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64).reshape(-1, self.observation_size)
        config = TaskConfig()
        t = 0.5 * (observations[:, 0] + 1.0) * config.episode_seconds
        actions = np.empty((len(t), self.action_size))
        right = actions[:, :5]
        left = actions[:, 5:10]
        r_home, r_grasp, r_handoff, r_retreat = self.nominal_right
        l_home, l_handoff, l_lift, l_windup, l_throw = self.nominal_left

        mask = t < config.right_reach_end
        fraction = self._smoothstep(t[mask] / config.right_reach_end)[:, None]
        right[mask] = (1.0 - fraction) * r_home + fraction * r_grasp
        mask = (t >= config.right_reach_end) & (t < config.right_close_end)
        right[mask] = r_grasp
        mask = (t >= config.right_close_end) & (t < config.handoff_reach_end)
        fraction = self._smoothstep((t[mask] - config.right_close_end) / (config.handoff_reach_end - config.right_close_end))[:, None]
        right[mask] = (1.0 - fraction) * r_grasp + fraction * r_handoff
        mask = (t >= config.handoff_reach_end) & (t < config.right_release_end)
        right[mask] = r_handoff
        mask = (t >= config.right_release_end) & (t < config.left_lift_end)
        fraction = self._smoothstep((t[mask] - config.right_release_end) / (config.left_lift_end - config.right_release_end))[:, None]
        right[mask] = (1.0 - fraction) * r_handoff + fraction * r_retreat
        right[t >= config.left_lift_end] = r_retreat

        left[t < config.left_reach_start] = l_home
        mask = (t >= config.left_reach_start) & (t < config.handoff_reach_end)
        fraction = self._smoothstep((t[mask] - config.left_reach_start) / (config.handoff_reach_end - config.left_reach_start))[:, None]
        left[mask] = (1.0 - fraction) * l_home + fraction * l_handoff
        mask = (t >= config.handoff_reach_end) & (t < config.right_release_end)
        left[mask] = l_handoff
        mask = (t >= config.right_release_end) & (t < config.left_lift_end)
        fraction = self._smoothstep((t[mask] - config.right_release_end) / (config.left_lift_end - config.right_release_end))[:, None]
        left[mask] = (1.0 - fraction) * l_handoff + fraction * l_lift
        mask = (t >= config.left_lift_end) & (t < config.windup_end)
        fraction = self._smoothstep((t[mask] - config.left_lift_end) / (config.windup_end - config.left_lift_end))[:, None]
        left[mask] = (1.0 - fraction) * l_lift + fraction * l_windup
        mask = (t >= config.windup_end) & (t < config.throw_end)
        fraction = self._smoothstep((t[mask] - config.windup_end) / (config.throw_end - config.windup_end))[:, None]
        left[mask] = (1.0 - fraction) * l_windup + fraction * l_throw
        left[t >= config.throw_end] = l_throw

        right_ratio = np.zeros((len(t), 6))
        mask = (t >= config.right_reach_end - 0.10) & (t < config.right_release_start)
        right_ratio[mask] = POWER_GRASP
        mask = (t >= config.right_release_start) & (t < config.right_release_end)
        fraction = self._smoothstep((t[mask] - config.right_release_start) / (config.right_release_end - config.right_release_start))[:, None]
        right_ratio[mask] = (1.0 - fraction) * POWER_GRASP
        left_ratio = np.zeros((len(t), 6))
        mask = (t >= config.left_close_start) & (t < config.left_release_time)
        left_ratio[mask] = POWER_GRASP
        mask = t >= config.left_release_time
        fraction = self._smoothstep((t[mask] - config.left_release_time) / 0.11)[:, None]
        left_ratio[mask] = (1.0 - fraction) * POWER_GRASP
        actions[:, 10:16] = 2.0 * right_ratio - 1.0
        actions[:, 16:22] = 2.0 * left_ratio - 1.0
        return np.clip(actions, -1.0, 1.0)

    def train_batch(self, observations: np.ndarray, targets: np.ndarray, learning_rate: float = 1.0e-3, weight_decay: float = 1.0e-6, output_weights: np.ndarray | None = None) -> float:
        x, h1, h2, prediction = self._forward(observations)
        targets = np.asarray(targets, dtype=np.float64)
        count = observations.shape[0]
        difference = prediction - targets
        weights = np.ones(self.action_size) if output_weights is None else np.asarray(output_weights, dtype=np.float64).reshape(self.action_size)
        weights = weights / np.mean(weights)
        loss = float(np.mean(difference * difference * weights[None, :]))
        d3 = (2.0 / (count * self.action_size)) * difference * weights[None, :] * (1.0 - prediction**2)
        gradients: dict[str, np.ndarray] = {}
        gradients["w3"] = h2.T @ d3 + weight_decay * self.parameters["w3"]
        gradients["b3"] = np.sum(d3, axis=0)
        d2 = (d3 @ self.parameters["w3"].T) * (1.0 - h2**2)
        gradients["w2"] = h1.T @ d2 + weight_decay * self.parameters["w2"]
        gradients["b2"] = np.sum(d2, axis=0)
        d1 = (d2 @ self.parameters["w2"].T) * (1.0 - h1**2)
        gradients["w1"] = x.T @ d1 + weight_decay * self.parameters["w1"]
        gradients["b1"] = np.sum(d1, axis=0)
        self._adam_step += 1
        beta1, beta2 = 0.9, 0.999
        for name, gradient in gradients.items():
            np.clip(gradient, -2.0, 2.0, out=gradient)
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * gradient**2
            corrected_m = self._adam_m[name] / (1.0 - beta1**self._adam_step)
            corrected_v = self._adam_v[name] / (1.0 - beta2**self._adam_step)
            self.parameters[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1.0e-8)
        return loss

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.parameters)
        payload.update({
            "observation_mean": self.observation_mean,
            "observation_std": self.observation_std,
            "nominal_right": self.nominal_right,
            "nominal_left": self.nominal_left,
            "residual_arm_scale": np.array(self.residual_arm_scale),
            "residual_hand_scale": np.array(self.residual_hand_scale),
            "metadata": np.array(json.dumps(metadata or {})),
        })
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "BimanualNeuralPolicy":
        archive = np.load(Path(path), allow_pickle=False)
        policy = cls(
            observation_size=archive["w1"].shape[0],
            action_size=archive["w3"].shape[1],
            nominal_right=archive["nominal_right"],
            nominal_left=archive["nominal_left"],
            hidden_size=archive["w1"].shape[1],
            residual_arm_scale=float(archive["residual_arm_scale"]),
            residual_hand_scale=float(archive["residual_hand_scale"]),
        )
        for name in policy.parameters:
            policy.parameters[name] = archive[name].copy()
        policy.observation_mean = archive["observation_mean"].copy()
        policy.observation_std = archive["observation_std"].copy()
        return policy

    @staticmethod
    def read_metadata(path: str | Path) -> dict:
        archive = np.load(Path(path), allow_pickle=False)
        return json.loads(str(archive["metadata"]))
