#!/usr/bin/env python3
"""Train a compact closed-loop Revo2 ball-grasp policy with NumPy CEM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from revo2_grasp_env import (
    FEATURE_NAMES,
    GraspTaskConfig,
    Revo2BallGraspEnv,
)
from revo2_sim import POSES, ROOT


class LinearGraspPolicy:
    """Six sigmoid actuator heads driven by normalized MuJoCo state."""

    def __init__(
        self,
        parameters: np.ndarray,
        observation_size: int = len(FEATURE_NAMES),
    ) -> None:
        expected = 6 * (observation_size + 1)
        parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if parameters.size != expected:
            raise ValueError(
                f"Expected {expected} policy parameters, got {parameters.size}"
            )
        self.observation_size = observation_size
        packed = parameters.reshape(6, observation_size + 1)
        self.bias = packed[:, 0]
        self.weights = packed[:, 1:]

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        logits = self.bias + self.weights @ observation
        logits = np.clip(logits, -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-logits))

    @property
    def parameters(self) -> np.ndarray:
        return np.column_stack((self.bias, self.weights)).reshape(-1)


def initial_parameters() -> np.ndarray:
    power = np.clip(POSES["power"], 0.02, 0.98)
    bias = np.log(power / (1.0 - power))
    weights = np.zeros((6, len(FEATURE_NAMES)))
    return np.column_stack((bias, weights)).reshape(-1)


def evaluate(
    env: Revo2BallGraspEnv,
    parameters: np.ndarray,
    seeds: list[int],
) -> tuple[float, list[dict[str, float]]]:
    policy = LinearGraspPolicy(parameters)
    metrics = [env.rollout(policy, seed=seed) for seed in seeds]
    score = float(np.mean([item["score"] for item in metrics]))
    return score, metrics


def save_policy(
    output: Path,
    parameters: np.ndarray,
    hand: str,
    task: str,
    training: dict[str, float | int | str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        parameters=np.asarray(parameters, dtype=np.float64),
        hand=np.array(hand),
        task=np.array(task),
        observation_size=np.array(len(FEATURE_NAMES), dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES),
        training_json=np.array(json.dumps(training, sort_keys=True)),
    )
    output.with_suffix(".json").write_text(
        json.dumps(training, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--task", choices=("hold", "move"), default="hold")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        help="warm-start from a saved .npz policy (useful for move training)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output .npz path (defaults to policies/revo2_<hand>_<task>.npz)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small smoke-training run: 3 iterations, 12 candidates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.iterations = 3
        args.population = 12
        args.episodes = 1
    if args.iterations <= 0 or args.population < 4 or args.episodes <= 0:
        raise ValueError("iterations/episodes must be positive; population >= 4")
    if not 0.05 <= args.elite_fraction <= 0.5:
        raise ValueError("--elite-fraction must be between 0.05 and 0.5")

    output = args.output or (
        ROOT / "policies" / f"revo2_{args.hand}_{args.task}.npz"
    )
    output = output.resolve()
    config = GraspTaskConfig(task=args.task)
    env = Revo2BallGraspEnv(hand=args.hand, config=config)
    rng = np.random.default_rng(args.seed)

    if args.initial_policy:
        with np.load(args.initial_policy) as saved:
            mean = np.asarray(saved["parameters"], dtype=np.float64)
    else:
        mean = initial_parameters()

    packed = mean.reshape(6, len(FEATURE_NAMES) + 1)
    std = np.full_like(packed, 0.055)
    std[:, 0] = 0.16
    if args.task == "move":
        # The hold policy already owns the stable grasp.  Explore the two
        # feedback channels that can create useful side-to-side modulation
        # much more strongly than unrelated state features.
        std[:] = 0.025
        std[:, 0] = 0.08
        std[:, 1 + FEATURE_NAMES.index("target_error_x")] = 1.25
        std[:, 1 + FEATURE_NAMES.index("target_velocity_x")] = 0.45
    std = std.reshape(-1)
    elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))

    baseline_seeds = [args.seed + 90_000 + index for index in range(5)]
    best_score, baseline_metrics = evaluate(env, mean, baseline_seeds)
    warm_start_score = best_score
    best_parameters = mean.copy()
    baseline_success = float(
        np.mean([item["success"] for item in baseline_metrics])
    )
    print(
        f"Warm start: score={best_score:.4f}, "
        f"success={baseline_success:.0%}"
    )

    started = time.monotonic()
    for iteration in range(args.iterations):
        candidates = rng.normal(
            loc=mean, scale=std, size=(args.population, mean.size)
        )
        candidates[0] = mean
        seeds = [
            args.seed + iteration * 1_000 + episode
            for episode in range(args.episodes)
        ]
        scores = np.empty(args.population)
        for index, candidate in enumerate(candidates):
            scores[index], _ = evaluate(env, candidate, seeds)

        elite_indices = np.argsort(scores)[-elite_count:]
        elites = candidates[elite_indices]
        elite_mean = np.mean(elites, axis=0)
        elite_std = np.std(elites, axis=0)
        mean = 0.25 * mean + 0.75 * elite_mean
        std = np.clip(0.35 * std + 0.65 * elite_std, 0.012, 0.45)

        iteration_best = int(np.argmax(scores))
        if scores[iteration_best] > best_score:
            best_score = float(scores[iteration_best])
            best_parameters = candidates[iteration_best].copy()

        print(
            f"Iteration {iteration + 1:02d}/{args.iterations}: "
            f"best={scores[iteration_best]:.4f}, "
            f"elite_mean={np.mean(scores[elite_indices]):.4f}, "
            f"global_best={best_score:.4f}"
        )

    validation_seeds = [args.seed + 100_000 + index for index in range(12)]
    validation_score, validation_metrics = evaluate(
        env, best_parameters, validation_seeds
    )
    success_rate = float(
        np.mean([item["success"] for item in validation_metrics])
    )
    drop_rate = float(
        np.mean([item["dropped"] for item in validation_metrics])
    )
    mean_distance = float(
        np.mean(
            [item["mean_hold_distance"] for item in validation_metrics]
        )
    )
    move_range = float(
        np.mean([item["move_ball_range"] for item in validation_metrics])
    )
    move_rmse = float(
        np.mean([item["move_tracking_rmse"] for item in validation_metrics])
    )
    elapsed = time.monotonic() - started

    training = {
        "algorithm": "cross_entropy_method_linear_feedback",
        "hand": args.hand,
        "task": args.task,
        "seed": args.seed,
        "iterations": args.iterations,
        "population": args.population,
        "episodes_per_candidate": args.episodes,
        "baseline_score": warm_start_score,
        "validation_score": validation_score,
        "validation_success_rate": success_rate,
        "validation_drop_rate": drop_rate,
        "validation_mean_hold_distance_m": mean_distance,
        "validation_move_range_m": move_range,
        "validation_move_tracking_rmse_m": move_rmse,
        "training_seconds": elapsed,
        "policy_parameters": int(best_parameters.size),
        "disturbance_force_n": config.disturbance_force,
        "gravity_m_per_s2": 9.81,
    }
    save_policy(
        output,
        best_parameters,
        hand=args.hand,
        task=args.task,
        training=training,
    )

    print(
        "\nValidation: "
        f"score={validation_score:.4f}, success={success_rate:.0%}, "
        f"drops={drop_rate:.0%}, "
        f"mean hold error={mean_distance * 1000:.1f} mm"
    )
    if args.task == "move":
        print(
            f"Movement: ball range={move_range * 1000:.1f} mm, "
            f"tracking RMSE={move_rmse * 1000:.1f} mm"
        )
    print(f"Saved policy: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
