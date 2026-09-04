#!/usr/bin/env python3
"""Small NumPy neural-network policy for the Revo2 ball-grasp task."""

from __future__ import annotations

import numpy as np

from revo2_grasp_env import FEATURE_NAMES
from revo2_sim import POSES


class NeuralGraspPolicy:
    """One-hidden-layer tanh MLP with six sigmoid actuator outputs."""

    policy_type = "mlp_tanh"

    def __init__(
        self,
        parameters: np.ndarray,
        observation_size: int = len(FEATURE_NAMES),
        hidden_size: int = 16,
    ) -> None:
        self.observation_size = int(observation_size)
        self.hidden_size = int(hidden_size)
        expected = self.parameter_count(self.observation_size, self.hidden_size)
        packed = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if packed.size != expected:
            raise ValueError(
                f"Expected {expected} neural policy parameters, "
                f"got {packed.size}"
            )

        offset = 0
        size = self.hidden_size * self.observation_size
        self.input_weights = packed[offset : offset + size].reshape(
            self.hidden_size, self.observation_size
        )
        offset += size
        self.hidden_bias = packed[offset : offset + self.hidden_size]
        offset += self.hidden_size
        size = 6 * self.hidden_size
        self.output_weights = packed[offset : offset + size].reshape(
            6, self.hidden_size
        )
        offset += size
        self.output_bias = packed[offset : offset + 6]

    @staticmethod
    def parameter_count(observation_size: int, hidden_size: int) -> int:
        return (
            hidden_size * observation_size
            + hidden_size
            + 6 * hidden_size
            + 6
        )

    @classmethod
    def initialized(
        cls,
        rng: np.random.Generator,
        observation_size: int = len(FEATURE_NAMES),
        hidden_size: int = 16,
    ) -> "NeuralGraspPolicy":
        input_weights = rng.normal(
            0.0, np.sqrt(1.0 / observation_size),
            size=(hidden_size, observation_size),
        )
        hidden_bias = np.zeros(hidden_size)
        output_weights = rng.normal(0.0, 0.015, size=(6, hidden_size))
        power = np.clip(POSES["power"], 0.02, 0.98)
        output_bias = np.log(power / (1.0 - power))
        return cls(
            cls.pack(
                input_weights,
                hidden_bias,
                output_weights,
                output_bias,
            ),
            observation_size=observation_size,
            hidden_size=hidden_size,
        )

    @staticmethod
    def pack(
        input_weights: np.ndarray,
        hidden_bias: np.ndarray,
        output_weights: np.ndarray,
        output_bias: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(input_weights).reshape(-1),
                np.asarray(hidden_bias).reshape(-1),
                np.asarray(output_weights).reshape(-1),
                np.asarray(output_bias).reshape(-1),
            )
        ).astype(np.float64)

    @property
    def parameters(self) -> np.ndarray:
        return self.pack(
            self.input_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
        )

    def forward_batch(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        observations = np.asarray(observations, dtype=np.float64)
        hidden = np.tanh(
            observations @ self.input_weights.T + self.hidden_bias
        )
        logits = hidden @ self.output_weights.T + self.output_bias
        logits = np.clip(logits, -12.0, 12.0)
        actions = 1.0 / (1.0 + np.exp(-logits))
        return hidden, actions

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        _, actions = self.forward_batch(
            np.asarray(observation, dtype=np.float64).reshape(
                1, self.observation_size
            )
        )
        return actions[0]

    def binary_cross_entropy_gradient(
        self,
        observations: np.ndarray,
        target_actions: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return imitation loss and the packed parameter gradient."""
        observations = np.asarray(observations, dtype=np.float64)
        target_actions = np.asarray(target_actions, dtype=np.float64)
        hidden, actions = self.forward_batch(observations)
        actions_safe = np.clip(actions, 1e-7, 1.0 - 1e-7)
        loss = -float(
            np.mean(
                target_actions * np.log(actions_safe)
                + (1.0 - target_actions) * np.log(1.0 - actions_safe)
            )
        )

        scale = 1.0 / (observations.shape[0] * 6)
        d_logits = (actions - target_actions) * scale
        output_weights_gradient = d_logits.T @ hidden
        output_bias_gradient = np.sum(d_logits, axis=0)
        d_hidden = d_logits @ self.output_weights
        d_hidden_pre_activation = d_hidden * (1.0 - hidden * hidden)
        input_weights_gradient = (
            d_hidden_pre_activation.T @ observations
        )
        hidden_bias_gradient = np.sum(
            d_hidden_pre_activation, axis=0
        )
        gradient = self.pack(
            input_weights_gradient,
            hidden_bias_gradient,
            output_weights_gradient,
            output_bias_gradient,
        )
        return loss, gradient
