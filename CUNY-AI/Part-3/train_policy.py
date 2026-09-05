#!/usr/bin/env python3
"""Train the robust dual-Revo grasp-handoff-throw neural policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from bimanual_env import BimanualHandoffThrowEnv
from bimanual_policy import BimanualNeuralPolicy


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policies/r1_dual_revo2_handoff_throw.npz"
SUMMARY_PATH = ROOT / "results/training_summary.json"
CURVE_PATH = ROOT / "results/training_curve.csv"
EVALUATION_PATH = ROOT / "results/neural_policy_evaluation.json"


def collect_teacher_data(env: BimanualHandoffThrowEnv, episodes: int, first_seed: int, policy: BimanualNeuralPolicy | None = None, teacher_probability: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    for episode in range(episodes):
        observation = env.reset(seed=first_seed + episode)
        terminated = False
        while not terminated:
            teacher = env.teacher_action()
            observations.append(observation.copy())
            labels.append(teacher.copy())
            executed = teacher if policy is None or env.rng.random() < teacher_probability else policy(observation)
            observation, _, terminated, _ = env.step(executed)
        metrics.append(env.metrics())
        if (episode + 1) % 20 == 0:
            rate = np.mean([row["success"] for row in metrics[-20:]])
            print(f"Collected {episode + 1}/{episodes} episodes; recent success={100 * rate:.0f}%", flush=True)
    return np.asarray(observations), np.asarray(labels), metrics


def train_steps(policy: BimanualNeuralPolicy, observations: np.ndarray, actions: np.ndarray, steps: int, batch_size: int, learning_rate: float, rng: np.random.Generator, curve: list[dict[str, float]], stage: int) -> None:
    encoded = policy.encode_targets(observations, actions)
    phase_ids = np.argmax(observations[:, 1:10], axis=1)
    phase_buckets = [np.flatnonzero(phase_ids == phase) for phase in range(9)]
    output_weights = np.array([4.0] * 10 + [1.6] * 12)
    for step in range(steps):
        sampled_phases = rng.integers(0, 9, size=batch_size)
        indices = np.empty(batch_size, dtype=int)
        for phase, bucket in enumerate(phase_buckets):
            selected = sampled_phases == phase
            if np.any(selected):
                indices[selected] = rng.choice(bucket, size=int(np.sum(selected)), replace=True)
        loss = policy.train_batch(observations[indices], encoded[indices], learning_rate=learning_rate, output_weights=output_weights)
        if step % 50 == 0 or step == steps - 1:
            count = min(2048, len(observations))
            validation_indices = rng.choice(len(observations), size=count, replace=False)
            error = policy.predict_latent(observations[validation_indices]) - encoded[validation_indices]
            curve.append({"stage": float(stage), "step": float(step), "training_mse": loss, "validation_mse": float(np.mean(error**2)), "validation_mae": float(np.mean(np.abs(error)))})
        if (step + 1) % 500 == 0:
            print(f"Training stage {stage}: {step + 1}/{steps}, loss={loss:.6f}", flush=True)


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in rows[0]:
        finite = [row[key] for row in rows if math.isfinite(row[key])]
        result[key] = float(np.mean(finite)) if finite else float("nan")
    return result


def evaluate(policy: BimanualNeuralPolicy, episodes: int, first_seed: int) -> tuple[dict[str, float], list[dict[str, float]]]:
    env = BimanualHandoffThrowEnv(domain_randomization=True)
    rows = [env.rollout(policy=policy, seed=first_seed + index) for index in range(episodes)]
    return mean_metrics(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-episodes", type=int, default=140)
    parser.add_argument("--initial-steps", type=int, default=3000)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=40)
    parser.add_argument("--dagger-steps", type=int, default=1500)
    parser.add_argument("--evaluation-episodes", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1.1e-3)
    parser.add_argument("--output", type=Path, default=POLICY_PATH)
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(2401)
    nominal_env = BimanualHandoffThrowEnv(domain_randomization=False)
    env = BimanualHandoffThrowEnv(domain_randomization=True)
    observations, actions, teacher_rows = collect_teacher_data(env, args.teacher_episodes, first_seed=24000)
    policy = BimanualNeuralPolicy.from_environment(nominal_env, hidden_size=args.hidden_size, seed=24)
    policy.set_normalization(observations)
    curve: list[dict[str, float]] = []
    train_steps(policy, observations, actions, args.initial_steps, args.batch_size, args.learning_rate, rng, curve, stage=0)

    dagger_probabilities = np.linspace(0.45, 0.18, max(args.dagger_rounds, 1))
    dagger_metrics: list[dict[str, float]] = []
    for round_index in range(args.dagger_rounds):
        new_observations, new_actions, behavior_rows = collect_teacher_data(env, args.dagger_episodes, first_seed=34000 + 1000 * round_index, policy=policy, teacher_probability=float(dagger_probabilities[round_index]))
        observations = np.concatenate((observations, new_observations), axis=0)
        actions = np.concatenate((actions, new_actions), axis=0)
        dagger_metrics.append(mean_metrics(behavior_rows))
        train_steps(policy, observations, actions, args.dagger_steps, args.batch_size, args.learning_rate * 0.75, rng, curve, stage=round_index + 1)

    evaluation, rows = evaluate(policy, args.evaluation_episodes, first_seed=64000)
    metadata = {
        "name": "R1 dual Revo 2 bimanual grasp-handoff-throw policy",
        "simulation": "MuJoCo",
        "architecture": [env.observation_size, args.hidden_size, args.hidden_size, env.action_size],
        "activation": "tanh",
        "training_method": "domain-randomized behavioral cloning plus two DAgger corrective rounds",
        "control_parameterization": "neural residual corrections around a safe bimanual reference trajectory",
        "training_samples": int(len(observations)),
        "teacher_episodes": args.teacher_episodes,
        "dagger_rounds": args.dagger_rounds,
        "evaluation_episodes": args.evaluation_episodes,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output, metadata=metadata)
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CURVE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    EVALUATION_PATH.write_text(json.dumps({"aggregate": evaluation, "episodes": rows}, indent=2), encoding="utf-8")
    summary = {**metadata, "teacher_metrics": mean_metrics(teacher_rows), "dagger_behavior_metrics": dagger_metrics, "final_imitation_metrics": curve[-1], "elapsed_seconds": time.time() - started, "policy_path": str(args.output)}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
