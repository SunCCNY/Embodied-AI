#!/usr/bin/env python3
"""Evaluate linear and neural Revo2 policies on identical random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from play_grasp_policy import load_policy
from revo2_grasp_env import GraspTaskConfig, Revo2BallGraspEnv
from revo2_sim import ROOT


def summarize(metrics: list[dict[str, float]]) -> dict[str, float | int]:
    successes = [item for item in metrics if item["success"]]
    return {
        "episodes": len(metrics),
        "successes": len(successes),
        "success_rate": float(
            np.mean([item["success"] for item in metrics])
        ),
        "drop_rate": float(
            np.mean([item["dropped"] for item in metrics])
        ),
        "mean_score": float(
            np.mean([item["score"] for item in metrics])
        ),
        "mean_hold_error_m": float(
            np.mean([item["mean_hold_distance"] for item in metrics])
        ),
        "successful_mean_hold_error_m": (
            float(
                np.mean(
                    [item["mean_hold_distance"] for item in successes]
                )
            )
            if successes
            else float("inf")
        ),
        "mean_final_contacts": float(
            np.mean([item["final_contacts"] for item in metrics])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-policy",
        type=Path,
        default=ROOT / "policies/revo2_right_hold.npz",
    )
    parser.add_argument(
        "--neural-policy",
        type=Path,
        default=ROOT / "policies/revo2_right_hold_neural.npz",
    )
    parser.add_argument("--episodes", type=int, default=23)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/linear_vs_neural_hold.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    linear, linear_hand, linear_task, linear_type = load_policy(
        args.linear_policy.resolve()
    )
    neural, neural_hand, neural_task, neural_type = load_policy(
        args.neural_policy.resolve()
    )
    if (linear_hand, linear_task) != (neural_hand, neural_task):
        raise ValueError("Policies must use the same hand and task")

    config = GraspTaskConfig(task=linear_task)
    linear_env = Revo2BallGraspEnv(hand=linear_hand, config=config)
    neural_env = Revo2BallGraspEnv(hand=neural_hand, config=config)
    rows: list[dict[str, float | int]] = []
    linear_metrics: list[dict[str, float]] = []
    neural_metrics: list[dict[str, float]] = []

    for index in range(args.episodes):
        seed = args.seed + index
        linear_result = linear_env.rollout(linear, seed=seed)
        neural_result = neural_env.rollout(neural, seed=seed)
        linear_metrics.append(linear_result)
        neural_metrics.append(neural_result)
        row = {
            "episode": index + 1,
            "seed": seed,
            "linear_success": int(linear_result["success"]),
            "neural_success": int(neural_result["success"]),
            "linear_hold_error_mm": (
                linear_result["mean_hold_distance"] * 1000
            ),
            "neural_hold_error_mm": (
                neural_result["mean_hold_distance"] * 1000
            ),
            "linear_score": linear_result["score"],
            "neural_score": neural_result["score"],
            "linear_contacts": int(linear_result["final_contacts"]),
            "neural_contacts": int(neural_result["final_contacts"]),
        }
        rows.append(row)
        print(
            f"Episode {index + 1:02d}: "
            f"linear={bool(linear_result['success'])} "
            f"({row['linear_hold_error_mm']:.1f} mm), "
            f"neural={bool(neural_result['success'])} "
            f"({row['neural_hold_error_mm']:.1f} mm)"
        )

    summary = {
        "hand": linear_hand,
        "task": linear_task,
        "seed": args.seed,
        "linear_policy": str(args.linear_policy.resolve()),
        "linear_policy_type": linear_type,
        "neural_policy": str(args.neural_policy.resolve()),
        "neural_policy_type": neural_type,
        "linear": summarize(linear_metrics),
        "neural": summarize(neural_metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPaired comparison")
    for label in ("linear", "neural"):
        item = summary[label]
        print(
            f"{label.capitalize():6s}: "
            f"success={item['success_rate']:.0%}, "
            f"successful hold error="
            f"{item['successful_mean_hold_error_m'] * 1000:.1f} mm, "
            f"mean score={item['mean_score']:.4f}"
        )
    print(f"Saved JSON: {args.output.resolve()}")
    print(f"Saved CSV:  {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
