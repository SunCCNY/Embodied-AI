#!/usr/bin/env python3
"""Train a neural Revo2 ball-grasp policy with NumPy and MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from neural_grasp_policy import NeuralGraspPolicy
from revo2_grasp_env import (
    FEATURE_NAMES,
    GraspTaskConfig,
    Revo2BallGraspEnv,
)
from revo2_sim import ROOT
from train_grasp_policy import LinearGraspPolicy


def evaluate(
    env: Revo2BallGraspEnv,
    policy_factory: Callable[[], NeuralGraspPolicy],
    seeds: list[int],
) -> tuple[float, list[dict[str, float]]]:
    policy = policy_factory()
    metrics = [env.rollout(policy, seed=seed) for seed in seeds]
    score = float(np.mean([item["score"] for item in metrics]))
    return score, metrics


def collect_teacher_data(
    env: Revo2BallGraspEnv,
    teacher: LinearGraspPolicy,
    seeds: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for seed in seeds:
        observation = env.reset(seed=seed)
        terminated = False
        while not terminated:
            action = teacher(observation)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _, terminated, _ = env.step(action)

    base_observations = np.asarray(observations)
    base_actions = np.asarray(actions)
    jittered_observations = np.clip(
        base_observations
        + rng.normal(0.0, 0.025, size=base_observations.shape),
        -5.0,
        5.0,
    )
    jittered_actions = np.asarray(
        [teacher(observation) for observation in jittered_observations]
    )
    return (
        np.concatenate((base_observations, jittered_observations)),
        np.concatenate((base_actions, jittered_actions)),
    )


def distill_teacher(
    policy: NeuralGraspPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    batch_size: int = 256,
    learning_rate: float = 0.012,
) -> tuple[NeuralGraspPolicy, float]:
    parameters = policy.parameters.copy()
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    latest_loss = float("nan")

    for step in range(1, steps + 1):
        indices = rng.integers(
            0, observations.shape[0],
            size=min(batch_size, observations.shape[0]),
        )
        current = NeuralGraspPolicy(
            parameters,
            observation_size=policy.observation_size,
            hidden_size=policy.hidden_size,
        )
        latest_loss, gradient = current.binary_cross_entropy_gradient(
            observations[indices], actions[indices]
        )
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = (
            beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        )
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        parameters -= (
            learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + epsilon)
        )

    return (
        NeuralGraspPolicy(
            parameters,
            observation_size=policy.observation_size,
            hidden_size=policy.hidden_size,
        ),
        latest_loss,
    )


def exploration_std(
    observation_size: int, hidden_size: int
) -> np.ndarray:
    input_weights = np.full((hidden_size, observation_size), 0.035)
    hidden_bias = np.full(hidden_size, 0.035)
    output_weights = np.full((6, hidden_size), 0.040)
    output_bias = np.full(6, 0.060)
    return NeuralGraspPolicy.pack(
        input_weights, hidden_bias, output_weights, output_bias
    )


def save_policy(
    output: Path,
    policy: NeuralGraspPolicy,
    hand: str,
    task: str,
    training: dict[str, float | int | str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        parameters=policy.parameters,
        policy_type=np.array(policy.policy_type),
        hidden_size=np.array(policy.hidden_size, dtype=np.int64),
        hand=np.array(hand),
        task=np.array(task),
        observation_size=np.array(policy.observation_size, dtype=np.int64),
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
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--distill-episodes", type=int, default=10)
    parser.add_argument("--distill-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        help="linear .npz policy used as a safe imitation teacher",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output .npz path (defaults to "
            "policies/revo2_<hand>_<task>_neural.npz)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small smoke-training run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.iterations = 2
        args.population = 8
        args.episodes = 1
        args.distill_episodes = 2
        args.distill_steps = 80
    if args.hidden_size < 2:
        raise ValueError("--hidden-size must be at least 2")
    if args.iterations <= 0 or args.population < 4 or args.episodes <= 0:
        raise ValueError("iterations/episodes must be positive; population >= 4")
    if not 0.05 <= args.elite_fraction <= 0.5:
        raise ValueError("--elite-fraction must be between 0.05 and 0.5")

    output = args.output or (
        ROOT
        / "policies"
        / f"revo2_{args.hand}_{args.task}_neural.npz"
    )
    output = output.resolve()
    config = GraspTaskConfig(task=args.task)
    env = Revo2BallGraspEnv(hand=args.hand, config=config)
    rng = np.random.default_rng(args.seed)
    neural = NeuralGraspPolicy.initialized(
        rng,
        observation_size=len(FEATURE_NAMES),
        hidden_size=args.hidden_size,
    )

    teacher_path = ""
    distillation_loss = float("nan")
    if args.initial_policy:
        teacher_path = str(args.initial_policy.resolve())
        with np.load(args.initial_policy) as saved:
            teacher = LinearGraspPolicy(
                np.asarray(saved["parameters"], dtype=np.float64)
            )
            teacher_hand = str(saved["hand"].item())
            teacher_task = str(saved["task"].item())
        if (teacher_hand, teacher_task) != (args.hand, args.task):
            raise ValueError(
                "The teacher policy hand/task must match this training run"
            )
        print(
            f"Collecting {args.distill_episodes} teacher episodes from "
            f"{args.initial_policy}..."
        )
        observations, actions = collect_teacher_data(
            env,
            teacher,
            [
                args.seed + 50_000 + index
                for index in range(args.distill_episodes)
            ],
            rng,
        )
        neural, distillation_loss = distill_teacher(
            neural,
            observations,
            actions,
            rng,
            steps=args.distill_steps,
        )
        imitation_error = float(
            np.mean(
                np.abs(neural.forward_batch(observations)[1] - actions)
            )
        )
        print(
            f"Imitation initialization: BCE={distillation_loss:.5f}, "
            f"mean action error={imitation_error:.4f}"
        )

    mean = neural.parameters.copy()
    std = exploration_std(len(FEATURE_NAMES), args.hidden_size)
    elite_count = max(
        2, int(math.ceil(args.population * args.elite_fraction))
    )
    monitor_seeds = [
        args.seed + 90_000 + index for index in range(5)
    ]
    best_score, baseline_metrics = evaluate(
        env,
        lambda: NeuralGraspPolicy(
            mean, len(FEATURE_NAMES), args.hidden_size
        ),
        monitor_seeds,
    )
    warm_start_score = best_score
    best_parameters = mean.copy()
    baseline_success = float(
        np.mean([item["success"] for item in baseline_metrics])
    )
    print(
        f"Neural warm start: score={best_score:.4f}, "
        f"success={baseline_success:.0%}, "
        f"parameters={mean.size}"
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
            scores[index], _ = evaluate(
                env,
                lambda candidate=candidate: NeuralGraspPolicy(
                    candidate, len(FEATURE_NAMES), args.hidden_size
                ),
                seeds,
            )

        elite_indices = np.argsort(scores)[-elite_count:]
        elites = candidates[elite_indices]
        elite_mean = np.mean(elites, axis=0)
        elite_std = np.std(elites, axis=0)
        mean = 0.30 * mean + 0.70 * elite_mean
        std = np.clip(0.40 * std + 0.60 * elite_std, 0.006, 0.18)

        monitor_score, monitor_metrics = evaluate(
            env,
            lambda: NeuralGraspPolicy(
                mean, len(FEATURE_NAMES), args.hidden_size
            ),
            monitor_seeds,
        )
        monitor_success = float(
            np.mean([item["success"] for item in monitor_metrics])
        )
        if monitor_score > best_score:
            best_score = monitor_score
            best_parameters = mean.copy()

        print(
            f"Iteration {iteration + 1:02d}/{args.iterations}: "
            f"batch_best={np.max(scores):.4f}, "
            f"elite_mean={np.mean(scores[elite_indices]):.4f}, "
            f"monitor={monitor_score:.4f}, "
            f"success={monitor_success:.0%}, "
            f"global_best={best_score:.4f}"
        )

    validation_seeds = [
        args.seed + 100_000 + index for index in range(23)
    ]
    validation_score, validation_metrics = evaluate(
        env,
        lambda: NeuralGraspPolicy(
            best_parameters, len(FEATURE_NAMES), args.hidden_size
        ),
        validation_seeds,
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
    successful_distances = [
        item["mean_hold_distance"]
        for item in validation_metrics
        if item["success"]
    ]
    successful_mean_distance = (
        float(np.mean(successful_distances))
        if successful_distances
        else float("inf")
    )
    elapsed = time.monotonic() - started
    policy = NeuralGraspPolicy(
        best_parameters, len(FEATURE_NAMES), args.hidden_size
    )
    training = {
        "algorithm": "cross_entropy_method_mlp_feedback",
        "policy_type": policy.policy_type,
        "hand": args.hand,
        "task": args.task,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "iterations": args.iterations,
        "population": args.population,
        "episodes_per_candidate": args.episodes,
        "teacher_policy": teacher_path,
        "distillation_episodes": (
            args.distill_episodes if args.initial_policy else 0
        ),
        "distillation_steps": (
            args.distill_steps if args.initial_policy else 0
        ),
        "distillation_bce": distillation_loss,
        "baseline_score": warm_start_score,
        "validation_episodes": len(validation_seeds),
        "validation_score": validation_score,
        "validation_success_rate": success_rate,
        "validation_drop_rate": drop_rate,
        "validation_mean_hold_distance_m": mean_distance,
        "validation_successful_mean_hold_distance_m": (
            successful_mean_distance
        ),
        "training_seconds": elapsed,
        "policy_parameters": int(best_parameters.size),
        "disturbance_force_n": config.disturbance_force,
        "gravity_m_per_s2": 9.81,
    }
    save_policy(output, policy, args.hand, args.task, training)
    print(
        "\nNeural validation: "
        f"score={validation_score:.4f}, success={success_rate:.0%}, "
        f"drops={drop_rate:.0%}, "
        f"mean hold error={mean_distance * 1000:.1f} mm"
    )
    print(f"Saved neural policy: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
