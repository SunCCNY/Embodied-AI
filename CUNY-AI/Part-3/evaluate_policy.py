#!/usr/bin/env python3
"""Evaluate the saved bimanual policy on held-out MuJoCo randomizations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bimanual_env import BimanualHandoffThrowEnv, ROOT
from bimanual_policy import BimanualNeuralPolicy


def aggregate(rows: list[dict[str, float]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in rows[0]:
        finite = [row[key] for row in rows if math.isfinite(row[key])]
        result[key] = float(np.mean(finite)) if finite else None
    return result


def clean(row: dict[str, float]) -> dict[str, float | None]:
    return {key: (float(value) if math.isfinite(value) else None) for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=74000)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "policies/r1_dual_revo2_handoff_throw.npz",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_policy_evaluation.json")
    args = parser.parse_args()
    policy = BimanualNeuralPolicy.load(args.policy)
    env = BimanualHandoffThrowEnv(domain_randomization=not args.deterministic)
    rows: list[dict[str, float]] = []
    for index in range(args.episodes):
        metrics = env.rollout(policy=policy, seed=args.first_seed + index)
        rows.append(metrics)
        print(
            f"Episode {index + 1:03d}: success={bool(metrics['success'])}, "
            f"right_grasp={bool(metrics['right_grasp_success'])}, "
            f"handoff={bool(metrics['left_transfer_success'])}, "
            f"throw={metrics['forward_throw_distance_m']:.3f} m"
        )
    report = {
        "policy": str(args.policy),
        "domain_randomization": not args.deterministic,
        "first_seed": args.first_seed,
        "episodes": args.episodes,
        "aggregate": aggregate(rows),
        "per_episode": [clean(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
