"""Stop-transition yaw turn + flight-phase fraction for R1 velocity checkpoints (headless, no render).

Each env runs at vx for --run-s seconds, then the command steps to zero for --stop-s seconds.
Reports: flight fraction (both feet off ground) while running, base yaw change and xy travel
after the stop, falls.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_walk_run import load_policy, make_env, set_cmd  # noqa: E402


def yaw(q):  # wxyz
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--speeds", default="0.5,1.0,1.5,2.0")
  ap.add_argument("--envs-per-speed", type=int, default=64)
  ap.add_argument("--run-s", type=float, default=8.0)
  ap.add_argument("--stop-s", type=float, default=5.0)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  device = "cuda:0"
  speeds = [float(s) for s in args.speeds.split(",")]
  n = args.envs_per_speed * len(speeds)
  wrapped, agent_cfg = make_env(n, False, device)
  _, policy = load_policy(wrapped, agent_cfg, args.ckpt, device)
  env = wrapped.unwrapped
  robot, feet = env.scene["robot"], env.scene["feet_ground_contact"]
  dt = env.step_dt
  vx = torch.tensor(speeds, device=device).repeat_interleave(args.envs_per_speed)
  run_steps, stop_steps = int(args.run_s / dt), int(args.stop_s / dt)

  obs, _ = wrapped.reset()
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  flight = torch.zeros(n, device=device)
  counted = 0
  with torch.inference_mode():
    for t in range(run_steps):
      set_cmd(wrapped, vx)
      obs, _, d, _ = wrapped.step(policy(obs))
      fell |= d.bool()
      if t >= run_steps // 2:
        flight += (feet.data.found.sum(-1) == 0).float()
        counted += 1
    yaw0 = yaw(robot.data.root_link_quat_w).clone()
    xy0 = robot.data.root_link_pos_w[:, :2].clone()
    max_dyaw = torch.zeros(n, device=device)
    for t in range(stop_steps):
      set_cmd(wrapped, torch.zeros_like(vx))
      obs, _, d, _ = wrapped.step(policy(obs))
      fell |= d.bool()
      dyaw = torch.atan2(torch.sin(yaw(robot.data.root_link_quat_w) - yaw0),
                         torch.cos(yaw(robot.data.root_link_quat_w) - yaw0)).abs()
      max_dyaw = torch.maximum(max_dyaw, dyaw)
    final_dyaw = dyaw
    travel = torch.linalg.norm(robot.data.root_link_pos_w[:, :2] - xy0, dim=-1)

  rows = []
  for i, s in enumerate(speeds):
    sl = slice(i * args.envs_per_speed, (i + 1) * args.envs_per_speed)
    ok = ~fell[sl]
    deg = lambda x: float(torch.rad2deg(x[sl][ok]).mean())
    rows.append({
      "cmd_vx": s, "fell": int(fell[sl].sum()),
      "flight_frac": float((flight[sl][ok] / counted).mean()),
      "stop_yaw_final_deg_mean": deg(final_dyaw),
      "stop_yaw_final_deg_max": float(torch.rad2deg(final_dyaw[sl][ok]).max()),
      "stop_yaw_peak_deg_mean": deg(max_dyaw),
      "stop_travel_m_mean": float(travel[sl][ok].mean()),
    })
    print(json.dumps(rows[-1]))
  Path(args.out).write_text(json.dumps({"ckpt": args.ckpt, "run_s": args.run_s,
                                        "stop_s": args.stop_s, "rows": rows}, indent=2))
  wrapped.close()


if __name__ == "__main__":
  main()
