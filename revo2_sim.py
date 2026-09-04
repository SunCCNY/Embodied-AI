#!/usr/bin/env python3
"""Interactive MuJoCo demo for the BrainCo Revo2 Basic hand."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "right": ROOT / "vendor/Revo2_xml/xml_right/scene_grasp.xml",
    "left": ROOT / "vendor/Revo2_xml/xml_left/scene_grasp.xml",
}

ACTUATOR_SUFFIXES = (
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)

ACTUATOR_LABELS = (
    "thumb opposition",
    "thumb flexion",
    "index flexion",
    "middle flexion",
    "ring flexion",
    "little-finger flexion",
)

# Each pose is expressed as a fraction of the actuator's official control range.
POSES = {
    "open": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "power": np.array([0.82, 0.88, 0.84, 0.88, 0.90, 0.92]),
    "pinch": np.array([0.78, 0.82, 0.88, 0.18, 0.08, 0.05]),
}

# STUDENT PARAMETERS: change one value at a time and record the result.
GRAVITY_Z = 0.0
KEY_INCREMENT = 0.05
AUTO_CYCLE_SECONDS = 5.0
CLOSE_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a viewer (useful for validation and training integration)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="number of simulation steps in headless mode",
    )
    return parser.parse_args()


class Revo2Simulation:
    def __init__(self, hand: str) -> None:
        self.hand = hand
        self.model = mujoco.MjModel.from_xml_path(str(SCENES[hand]))
        self.data = mujoco.MjData(self.model)

        # The free object begins in the palm; zero gravity lets the fixed wrist
        # demonstrate contact without requiring an arm controller.
        self.model.opt.gravity[:] = (0.0, 0.0, GRAVITY_Z)

        names = [f"{hand}_{suffix}" for suffix in ACTUATOR_SUFFIXES]
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                for name in names
            ],
            dtype=int,
        )
        if np.any(self.actuator_ids < 0):
            raise RuntimeError(f"Missing expected {hand}-hand actuators: {names}")

        ranges = self.model.actuator_ctrlrange[self.actuator_ids]
        self.ctrl_low = ranges[:, 0].copy()
        self.ctrl_high = ranges[:, 1].copy()
        self.target_ratio = POSES["open"].copy()
        self.selected = 0
        self.auto_demo = False
        self.demo_start = time.monotonic()
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.target_ratio[:] = POSES["open"]
        self.data.ctrl[self.actuator_ids] = self.ratio_to_ctrl(self.target_ratio)
        self.auto_demo = False
        self.demo_start = time.monotonic()
        mujoco.mj_forward(self.model, self.data)

    def ratio_to_ctrl(self, ratio: np.ndarray) -> np.ndarray:
        return self.ctrl_low + np.clip(ratio, 0.0, 1.0) * (
            self.ctrl_high - self.ctrl_low
        )

    def set_pose(self, pose: str) -> None:
        self.auto_demo = False
        self.target_ratio[:] = POSES[pose]
        print(f"Pose: {pose}", flush=True)

    def toggle_demo(self) -> None:
        self.auto_demo = not self.auto_demo
        self.demo_start = time.monotonic()
        print(
            f"Automatic grasp demo: {'on' if self.auto_demo else 'off'}",
            flush=True,
        )

    def key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except (ValueError, OverflowError):
            return

        if key == "O":
            self.set_pose("open")
        elif key == "C":
            self.set_pose("power")
        elif key == "P":
            self.set_pose("pinch")
        elif key in ("A", " "):
            self.toggle_demo()
        elif key == "R":
            self.reset()
            print("Simulation reset", flush=True)
        elif key in "123456":
            self.selected = int(key) - 1
            print(
                f"Selected actuator {key}: {ACTUATOR_LABELS[self.selected]}",
                flush=True,
            )
        elif key in ("[", "]"):
            self.auto_demo = False
            delta = KEY_INCREMENT if key == "]" else -KEY_INCREMENT
            self.target_ratio[self.selected] = np.clip(
                self.target_ratio[self.selected] + delta, 0.0, 1.0
            )
            print(
                f"{ACTUATOR_LABELS[self.selected]}: "
                f"{self.target_ratio[self.selected]:.0%}",
                flush=True,
            )

    def update_controls(self, wall_time: float) -> None:
        if self.auto_demo:
            phase = (
                (wall_time - self.demo_start)
                * (2.0 * math.pi / AUTO_CYCLE_SECONDS)
            )
            blend = 0.5 - 0.5 * math.cos(phase)
            self.target_ratio[:] = blend * POSES["power"]

        target_ctrl = self.ratio_to_ctrl(self.target_ratio)
        # Slew-limit commands so the fingers close over roughly five seconds.
        # The official collision meshes can impart unrealistic impulses if a
        # full grasp command is applied in a single physics step.
        current = self.data.ctrl[self.actuator_ids]
        max_delta = (
            (self.ctrl_high - self.ctrl_low)
            * self.model.opt.timestep
            / CLOSE_SECONDS
        )
        self.data.ctrl[self.actuator_ids] = current + np.clip(
            target_ctrl - current, -max_delta, max_delta
        )

    def step(self, wall_time: float) -> None:
        self.update_controls(wall_time)
        mujoco.mj_step(self.model, self.data)


def run_headless(sim: Revo2Simulation, steps: int) -> None:
    sim.auto_demo = True
    sim.demo_start = 0.0
    for step in range(steps):
        sim.step(step * sim.model.opt.timestep)

    if not np.all(np.isfinite(sim.data.qpos)):
        raise RuntimeError("Simulation produced non-finite joint positions")

    print(
        "Validation passed: "
        f"hand={sim.hand}, actuators={sim.model.nu}, "
        f"joints={sim.model.njnt}, bodies={sim.model.nbody}, steps={steps}"
    )


def run_interactive(sim: Revo2Simulation) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("The MuJoCo viewer is unavailable") from exc

    print("\nBrainCo Revo2 Basic interactive simulation")
    print("O open | C power grasp | P pinch | A/Space auto | R reset")
    print("1-6 select actuator | [ and ] adjust | Esc exit\n")

    with mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=sim.key_callback
    ) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, -0.055, 0.075])
        viewer.cam.distance = 0.34
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -12.0

        while viewer.is_running():
            frame_start = time.monotonic()
            sim.step(frame_start)
            viewer.sync()

            remaining = sim.model.opt.timestep - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    sim = Revo2Simulation(args.hand)
    if args.headless:
        run_headless(sim, args.steps)
    else:
        run_interactive(sim)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
