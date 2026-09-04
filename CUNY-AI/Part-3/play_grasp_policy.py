#!/usr/bin/env python3
"""Evaluate or visualize a trained Revo2 ball-grasp policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from neural_grasp_policy import NeuralGraspPolicy
from revo2_grasp_env import GraspTaskConfig, Revo2BallGraspEnv
from train_grasp_policy import LinearGraspPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="trained .npz file produced by train_grasp_policy.py",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run repeatable evaluation without opening a viewer",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1200)
    return parser.parse_args()


def load_policy(
    path: Path,
) -> tuple[LinearGraspPolicy | NeuralGraspPolicy, str, str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Policy not found: {path}\n"
            "Train one first with scripts/train_grasp_policy.py"
        )
    with np.load(path) as saved:
        parameters = np.asarray(saved["parameters"], dtype=np.float64)
        hand = str(saved["hand"].item())
        task = str(saved["task"].item())
        policy_type = (
            str(saved["policy_type"].item())
            if "policy_type" in saved.files
            else "linear_sigmoid"
        )
        if policy_type == NeuralGraspPolicy.policy_type:
            hidden_size = int(saved["hidden_size"].item())
            policy = NeuralGraspPolicy(
                parameters,
                observation_size=int(saved["observation_size"].item()),
                hidden_size=hidden_size,
            )
        elif policy_type == "linear_sigmoid":
            policy = LinearGraspPolicy(parameters)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
    return policy, hand, task, policy_type


def run_headless(
    policy: LinearGraspPolicy | NeuralGraspPolicy,
    env: Revo2BallGraspEnv,
    episodes: int,
    seed: int,
) -> None:
    metrics = []
    for index in range(episodes):
        result = env.rollout(policy, seed=seed + index)
        metrics.append(result)
        print(
            f"Episode {index + 1}: "
            f"success={bool(result['success'])}, "
            f"score={result['score']:.4f}, "
            f"hold_error={result['mean_hold_distance'] * 1000:.1f} mm, "
            f"contacts={int(result['final_contacts'])}, "
            f"move_range={result['move_ball_range'] * 1000:.1f} mm"
        )
    print(
        "\nSummary: "
        f"success={np.mean([item['success'] for item in metrics]):.0%}, "
        f"drops={np.mean([item['dropped'] for item in metrics]):.0%}, "
        "mean hold error="
        f"{np.mean([item['mean_hold_distance'] for item in metrics]) * 1000:.1f} mm, "
        "move range="
        f"{np.mean([item['move_ball_range'] for item in metrics]) * 1000:.1f} mm"
    )


def run_interactive(
    policy: LinearGraspPolicy | NeuralGraspPolicy,
    env: Revo2BallGraspEnv,
    seed: int,
) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("The MuJoCo viewer is unavailable") from exc

    observation = env.reset(seed=seed)
    episode = 1
    print(
        "\nTrained BrainCo Revo2 policy\n"
        "Orange sphere: grasp object | green sphere: current target | Esc: exit\n"
    )
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, -0.055, 0.075])
        viewer.cam.distance = 0.34
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -12.0

        while viewer.is_running():
            frame_start = time.monotonic()
            action = policy(observation)
            observation, _, terminated, _ = env.step(action)
            viewer.sync()
            if terminated:
                result = env.metrics()
                print(
                    f"Episode {episode}: "
                    f"success={bool(result['success'])}, "
                    f"hold error={result['mean_hold_distance'] * 1000:.1f} mm"
                )
                episode += 1
                observation = env.reset(seed=seed + episode - 1)

            remaining = env.control_dt - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    policy_path = args.policy.resolve()
    policy, hand, task, policy_type = load_policy(policy_path)
    env = Revo2BallGraspEnv(
        hand=hand, config=GraspTaskConfig(task=task)
    )
    print(
        f"Loaded {hand}-hand {task} {policy_type} policy: {policy_path}"
    )
    if args.headless:
        run_headless(policy, env, args.episodes, args.seed)
    else:
        run_interactive(policy, env, args.seed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
