"""Headless evaluation of an R1 velocity checkpoint: fixed-command metrics + scripted-command video.

Metrics pass: envs are split evenly across the requested forward speeds; each env holds its
command for the whole run. Reports achieved forward velocity (body frame, envs that never fell),
fall rate, lateral/yaw drift, and for v=0 the base xy drift (standing check).

Video pass (optional): one env follows a command schedule, rendered offscreen.
WSL: run with MUJOCO_GL=osmesa + LD_LIBRARY_PATH to local osmesa (EGL wedges the GPU here).
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Unitree-R1-Flat"


def make_env(num_envs: int, render: bool, device: str):
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = num_envs
  cmd = env_cfg.commands["twist"]
  # Commands are written directly every step; disable everything that would overwrite them.
  cmd.heading_command = False
  cmd.ranges.heading = None
  cmd.rel_standing_envs = 0.0
  cmd.init_velocity_prob = 0.0
  cmd.resampling_time_range = (1e9, 1e9)
  cmd.debug_vis = render
  if render:
    env_cfg.viewer.max_extra_envs = 0
    env_cfg.viewer.width = 960
    env_cfg.viewer.height = 540
  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if render else None
  )
  agent_cfg = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return wrapped, agent_cfg


def load_policy(wrapped, agent_cfg, ckpt: str, device: str):
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  return runner, runner.get_inference_policy(device=device)


def set_cmd(wrapped, vx: torch.Tensor):
  term = wrapped.unwrapped.command_manager.get_term("twist")
  term.vel_command_b[:, 0] = vx
  term.vel_command_b[:, 1:] = 0.0


def metrics_pass(ckpt, speeds, envs_per_speed, seconds, settle_s, device):
  n = envs_per_speed * len(speeds)
  wrapped, agent_cfg = make_env(n, False, device)
  _, policy = load_policy(wrapped, agent_cfg, ckpt, device)
  vx = torch.tensor(speeds, device=device).repeat_interleave(envs_per_speed)
  robot = wrapped.unwrapped.scene["robot"]
  dt = wrapped.unwrapped.step_dt
  steps, settle = int(seconds / dt), int(settle_s / dt)

  obs, _ = wrapped.reset()
  set_cmd(wrapped, vx)
  obs = wrapped.get_observations()
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  v_sum = torch.zeros(n, 3, device=device)
  wz_sum = torch.zeros(n, device=device)
  xy0 = None
  with torch.inference_mode():
    for t in range(steps):
      set_cmd(wrapped, vx)
      obs, _, dones, _ = wrapped.step(policy(obs))
      fell |= dones.bool()
      if t == settle:
        xy0 = robot.data.root_link_pos_w[:, :2].clone()
      if t >= settle:
        v_sum += robot.data.root_link_lin_vel_b
        wz_sum += robot.data.root_link_ang_vel_b[:, 2]
    xy1 = robot.data.root_link_pos_w[:, :2]
  k = steps - settle
  v_mean, wz_mean = v_sum / k, wz_sum / k
  drift = torch.linalg.norm(xy1 - xy0, dim=-1)

  rows = []
  for i, s in enumerate(speeds):
    sl = slice(i * envs_per_speed, (i + 1) * envs_per_speed)
    ok = ~fell[sl]
    row = {
      "cmd_vx": s,
      "envs": envs_per_speed,
      "fell": int(fell[sl].sum()),
      "vx_mean": float(v_mean[sl, 0][ok].mean()) if ok.any() else None,
      "vx_std": float(v_mean[sl, 0][ok].std()) if ok.sum() > 1 else None,
      "vy_mean": float(v_mean[sl, 1][ok].mean()) if ok.any() else None,
      "wz_mean": float(wz_mean[sl][ok].mean()) if ok.any() else None,
    }
    if s == 0.0:
      row["xy_drift_m_max"] = float(drift[sl][ok].max()) if ok.any() else None
      row["xy_drift_m_mean"] = float(drift[sl][ok].mean()) if ok.any() else None
    rows.append(row)
  wrapped.close()
  return rows


def video_pass(ckpt, schedule, out_path, device):
  import imageio.v2 as imageio

  wrapped, agent_cfg = make_env(1, True, device)
  _, policy = load_policy(wrapped, agent_cfg, ckpt, device)
  dt = wrapped.unwrapped.step_dt
  obs, _ = wrapped.reset()
  frames, falls = [], 0
  with torch.inference_mode():
    total = schedule[-1][0]
    seg = 0
    for t in range(int(total / dt)):
      while t * dt >= schedule[seg][0]:
        seg += 1
      set_cmd(wrapped, torch.tensor([schedule[seg][1]], device=device))
      obs, _, dones, _ = wrapped.step(policy(obs))
      falls += int(dones.sum())
      frames.append(wrapped.unwrapped.render())
  wrapped.close()
  writer = imageio.get_writer(out_path, fps=round(1 / dt), codec="libx264", quality=7)
  for f in frames:
    writer.append_data(f)
  writer.close()
  return falls


def parse_schedule(s: str):
  """'3:0,6:0.5,9:1.0' -> [(3.0, 0.0), (6.0, 0.5), (9.0, 1.0)] (segment end time, vx)."""
  return [tuple(float(x) for x in p.split(":")) for p in s.split(",")]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--tag", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--speeds", default="0,0.5,1.0")
  ap.add_argument("--envs-per-speed", type=int, default=64)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--settle", type=float, default=4.0)
  ap.add_argument("--video-schedule", default=None)
  ap.add_argument("--export-onnx", action="store_true")
  args = ap.parse_args()

  device = "cuda:0"
  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  speeds = [float(s) for s in args.speeds.split(",")]

  result = {"ckpt": os.path.abspath(args.ckpt), "tag": args.tag, "seconds": args.seconds,
            "settle_s": args.settle}
  result["metrics"] = metrics_pass(
    args.ckpt, speeds, args.envs_per_speed, args.seconds, args.settle, device
  )
  for r in result["metrics"]:
    print(json.dumps(r))

  if args.export_onnx:
    wrapped, agent_cfg = make_env(1, False, device)
    runner, _ = load_policy(wrapped, agent_cfg, args.ckpt, device)
    runner.export_policy_to_onnx(str(out), filename=f"{args.tag}.onnx")
    wrapped.close()
    result["onnx"] = str(out / f"{args.tag}.onnx")

  if args.video_schedule:
    sched = parse_schedule(args.video_schedule)
    vpath = out / f"{args.tag}.mp4"
    result["video"] = str(vpath)
    result["video_schedule"] = sched
    result["video_falls"] = video_pass(args.ckpt, sched, str(vpath), device)
    print(f"video -> {vpath} (falls: {result['video_falls']})")

  (out / f"{args.tag}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
