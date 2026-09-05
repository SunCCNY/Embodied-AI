#!/usr/bin/env python3
"""Render a trained bimanual grasp-handoff-throw rollout to MP4 and PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from bimanual_env import BimanualHandoffThrowEnv, ROOT
from bimanual_policy import BimanualNeuralPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=64002)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mode", choices=("trained", "untrained", "teacher"), default="trained")
    parser.add_argument("--network-seed", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    env = BimanualHandoffThrowEnv(domain_randomization=not args.deterministic)
    if args.mode == "trained":
        policy = BimanualNeuralPolicy.load(ROOT / "policies/r1_dual_revo2_handoff_throw.npz")
    elif args.mode == "untrained":
        nominal_env = BimanualHandoffThrowEnv(domain_randomization=False)
        policy = BimanualNeuralPolicy.from_environment(
            nominal_env, hidden_size=160, seed=args.network_seed
        )
    else:
        policy = None
    output = args.output or ROOT / "media" / f"{args.mode}_bimanual_handoff_throw.mp4"
    observation = env.reset(seed=args.seed)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.array([0.25, 0.00, 0.79])
    camera.distance = 1.52
    camera.azimuth = 150
    camera.elevation = -12
    renderer = mujoco.Renderer(env.model, height=720, width=960)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=round(1.0 / env.control_dt), codec="libx264", quality=8)
    snapshots = {
        "right_grasp": 2.22,
        "transfer": 3.50,
        "handoff": 4.55,
        "left_hold": 5.30,
        "throw": 5.98,
    }
    saved: set[str] = set()
    terminated = False
    final_frame = None
    try:
        while not terminated:
            action = env.teacher_action() if policy is None else policy(observation)
            observation, _, terminated, _ = env.step(action)
            renderer.update_scene(env.data, camera=camera)
            frame = renderer.render()
            final_frame = frame
            writer.append_data(frame)
            for name, timestamp in snapshots.items():
                if name not in saved and env.elapsed >= timestamp:
                    imageio.imwrite(ROOT / f"media/{args.mode}_{name}.png", frame)
                    saved.add(name)
    finally:
        writer.close()
        renderer.close()
    if final_frame is not None:
        imageio.imwrite(ROOT / f"media/{args.mode}_final.png", final_frame)
    metrics = env.metrics()
    metrics.update({
        "controller": args.mode,
        "episode_seed": args.seed,
        "domain_randomization": not args.deterministic,
        "network_seed": args.network_seed if args.mode == "untrained" else None,
    })
    serializable = {key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value) for key, value in metrics.items()}
    metrics_path = ROOT / "results" / f"{args.mode}_demo_metrics.json"
    metrics_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(metrics)
    print(f"Rendered: {output}")


if __name__ == "__main__":
    main()
