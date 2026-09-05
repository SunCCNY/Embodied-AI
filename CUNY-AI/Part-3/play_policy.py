#!/usr/bin/env python3
"""Open the final dual-Revo neural policy in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco.viewer

from bimanual_env import BimanualHandoffThrowEnv, ROOT
from bimanual_policy import BimanualNeuralPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=64002)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "policies/r1_dual_revo2_handoff_throw.npz",
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    env = BimanualHandoffThrowEnv(domain_randomization=not args.deterministic)
    policy = BimanualNeuralPolicy.load(args.policy)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.25, 0.0, 0.79]
        viewer.cam.distance = 1.52
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -12
        for episode in range(args.episodes):
            observation = env.reset(seed=args.seed + episode)
            terminated = False
            while not terminated and viewer.is_running():
                started = time.time()
                observation, _, terminated, _ = env.step(policy(observation))
                viewer.sync()
                time.sleep(max(0.0, env.control_dt - (time.time() - started)))
            metrics = env.metrics()
            print(
                f"Episode {episode + 1}: success={bool(metrics['success'])}, "
                f"handoff={bool(metrics['left_transfer_success'])}, "
                f"throw={metrics['forward_throw_distance_m']:.3f} m"
            )
            if not viewer.is_running():
                break


if __name__ == "__main__":
    main()
