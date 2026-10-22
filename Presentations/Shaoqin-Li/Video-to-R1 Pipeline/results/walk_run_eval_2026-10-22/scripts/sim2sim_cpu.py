"""Sim2sim for the R1 velocity policy: exported ONNX in plain CPU MuJoCo (no mjlab at rollout time).

The model is the exact MjModel mjlab compiles for training (floor, actuators, collisions, solver
settings), saved as .mjb; the rollout re-implements the observation/action wiring the way Unitree's
C++ deploy does (deploy/include/isaaclab/envs/mdp/observations/observations.h): 83-dim obs =
gyro(3) | projected gravity(3) | command(3) | gait phase(2, zeroed if |cmd|<0.1) |
joint_pos - default(24) | joint_vel(24) | last action(24); target = default + scale * action.

  parity  : checks CPU obs/ctrl/ONNX against the mjlab env on the same states (must pass first)
  sweep   : speeds x action delays x action-scale variant, N seeds each, multiprocessing
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PHYS_DT, DECIM, PERIOD = 0.005, 4, 0.6
SPEC_CACHE = HERE.parent / "logs" / "sim2sim_cpu"


def export_spec(ckpt: str):
  """Compile the training env once; save the MjModel and the policy-side wiring."""
  import torch

  sys.path.insert(0, str(HERE))
  from eval_walk_run import load_policy, make_env

  w, agent_cfg = make_env(1, False, "cuda:0")
  env = w.unwrapped
  m = env.sim.mj_model
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  names = list(term.target_names)  # policy action order
  jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + n) for n in names]
  act_of_joint = []
  for j in jnt:
    (a,) = np.where(m.actuator_trnid[:, 0] == j)[0]
    act_of_joint.append(int(a))
  scale = term._scale[0].cpu().numpy() if hasattr(term, "_scale") else None
  default = robot.data.default_joint_pos[0].cpu().numpy()
  rnames = list(robot.joint_names)
  default = np.array([default[rnames.index(n)] for n in names])
  SPEC_CACHE.mkdir(parents=True, exist_ok=True)
  mujoco.mj_saveModel(m, str(SPEC_CACHE / "r1_flat.mjb"))
  spec = {
    "joint_names": names,
    "qpos_adr": [int(m.jnt_qposadr[j]) for j in jnt],
    "qvel_adr": [int(m.jnt_dofadr[j]) for j in jnt],
    "act_of_joint": act_of_joint,
    "scale": [float(s) for s in scale],
    "default": default.tolist(),
    "root_z": float(robot.data.default_root_state[0, 2]),
  }
  (SPEC_CACHE / "spec.json").write_text(json.dumps(spec, indent=1))
  return w, agent_cfg, env, spec, load_policy


class Rollout:
  def __init__(self, onnx_path, spec, scale_override=None):
    import onnxruntime as ort

    self.m = mujoco.MjModel.from_binary_path(str(SPEC_CACHE / "r1_flat.mjb"))
    self.d = mujoco.MjData(self.m)
    self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    self.qa = np.array(spec["qpos_adr"])
    self.va = np.array(spec["qvel_adr"])
    self.aj = np.array(spec["act_of_joint"])
    self.default = np.array(spec["default"])
    self.scale = np.array(spec["scale"]) if scale_override is None else np.full(24, scale_override)
    self.root_z = spec["root_z"]
    self.gyro = self.m.sensor("robot/imu_ang_vel")

  def obs(self, cmd, t, last_action):
    d = self.d
    gyro = d.sensordata[self.gyro.adr[0]:self.gyro.adr[0] + 3]
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    grav = R.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    ph = (t % PERIOD) / PERIOD
    phase = np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
    if np.linalg.norm(cmd) < 0.1:
      phase[:] = 0.0
    return np.concatenate([gyro, grav, cmd, phase, d.qpos[self.qa] - self.default,
                           d.qvel[self.va], last_action]).astype(np.float32)

  def act(self, o):
    return self.sess.run(None, {"obs": o[None]})[0][0]

  def run(self, vx, seconds, delay_ms, seed, settle_s=4.0):
    rng = np.random.default_rng(seed)
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = [0, 0, self.root_z]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[self.qa] = self.default + rng.uniform(-0.05, 0.05, 24)
    mujoco.mj_forward(m, d)
    cmd = np.array([vx, 0.0, 0.0])
    last = np.zeros(24, np.float32)
    target = self.default.copy()
    delay_ticks = int(round(delay_ms / 1000 / PHYS_DT))
    pending = []  # (apply_at_tick, target)
    n_pol = int(seconds / (PHYS_DT * DECIM))
    settle = int(settle_s / (PHYS_DT * DECIM))
    tick, vsum, k, fell_t = 0, 0.0, 0, None
    for i in range(n_pol):
      t = i * PHYS_DT * DECIM
      a = self.act(self.obs(cmd, t, last))
      last = a.astype(np.float32)
      pending.append((tick + delay_ticks, self.default + self.scale * a))
      for _ in range(DECIM):
        while pending and pending[0][0] <= tick:
          target = pending.pop(0)[1]
        d.ctrl[self.aj] = target
        mujoco.mj_step(m, d)
        tick += 1
      R = np.empty(9)
      mujoco.mju_quat2Mat(R, d.qpos[3:7])
      if R[8] < 0.5 or d.qpos[2] < 0.35:  # tilt > 60 deg or pelvis on the ground
        fell_t = t
        break
      if i >= settle:
        vsum += (R.reshape(3, 3).T @ d.qvel[:3])[0]
        k += 1
    return {"fell": fell_t is not None, "fell_t": fell_t, "vx": vsum / k if k else None,
            "xy_final": d.qpos[:2].tolist()}


def parity(ckpt, onnx_path):
  import torch

  w, agent_cfg, env, spec, load_policy = export_spec(ckpt)
  _, policy = load_policy(w, agent_cfg, ckpt, "cuda:0")
  ro = Rollout(onnx_path, spec)
  print("action order:", spec["joint_names"][:6], "... scale:", np.round(spec["scale"], 4).tolist())
  sys.path.insert(0, str(HERE))
  from eval_walk_run import set_cmd

  robot = env.scene["robot"]
  obs, _ = w.reset()
  worst = {"obs": 0.0, "onnx_vs_torch": 0.0, "ctrl": 0.0}
  with torch.inference_mode():
    for i in range(150):
      vx = 0.0 if i < 50 else 1.0
      set_cmd(w, torch.tensor([vx], device="cuda:0"))
      obs = w.get_observations()
      o_gpu = obs["actor"][0].cpu().numpy()
      # copy GPU state into CPU MuJoCo and rebuild the observation there
      ro.d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
      ro.d.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
      mujoco.mj_forward(ro.m, ro.d)
      t = float(env.episode_length_buf[0]) * env.step_dt
      last = env.action_manager.action[0].cpu().numpy()
      o_cpu = ro.obs(np.array([vx, 0, 0]), t, last)
      diff = np.abs(o_cpu - o_gpu)
      worst["obs"] = max(worst["obs"], float(diff.max()))
      if os.environ.get("PARITY_DEBUG") and diff.max() > 1e-3:
        segs = {"gyro": (0, 3), "grav": (3, 6), "cmd": (6, 9), "phase": (9, 11), "qpos": (11, 35),
                "qvel": (35, 59), "last": (59, 83)}
        print(i, {k: round(float(diff[a:b].max()), 4) for k, (a, b) in segs.items()}, "t", t,
              "phase gpu", o_gpu[9:11], "cpu", o_cpu[9:11])
      a_t = policy(obs)
      a_o = ro.act(o_gpu)
      worst["onnx_vs_torch"] = max(worst["onnx_vs_torch"],
                                   float(np.abs(a_o - a_t[0].cpu().numpy()).max()))
      obs, _, _, _ = w.step(a_t)
      ctrl_gpu = env.sim.data.ctrl[0].cpu().numpy()
      ctrl_cpu = np.empty(24)
      ctrl_cpu[ro.aj] = ro.default + ro.scale * a_t[0].cpu().numpy()
      cd = np.abs(ctrl_cpu - ctrl_gpu)
      worst["ctrl"] = max(worst["ctrl"], float(cd.max()))
      if os.environ.get("PARITY_DEBUG") and cd.max() > 1e-3:
        print(i, "ctrl diff act", int(cd.argmax()), float(cd.max()), "cpu", float(ctrl_cpu[cd.argmax()]),
              "gpu", float(ctrl_gpu[cd.argmax()]), "range", ro.m.actuator_ctrlrange[cd.argmax()].tolist())
  print("PARITY worst abs diff:", json.dumps(worst))
  return worst


def _job(args):
  onnx_path, scale_override, vx, delay, seed, seconds = args
  spec = json.loads((SPEC_CACHE / "spec.json").read_text())
  r = Rollout(onnx_path, spec, scale_override).run(vx, seconds, delay, seed)
  r.update(vx_cmd=vx, delay_ms=delay, seed=seed, scale="deploy_0.25" if scale_override else "trained")
  return r


def sweep(onnx_path, speeds, delays, seeds, seconds, scale_variants, out):
  jobs = [(onnx_path, sv, v, dl, s, seconds)
          for sv in scale_variants for v in speeds for dl in delays for s in range(seeds)]
  with Pool(min(os.cpu_count() or 4, 16)) as p:
    res = p.map(_job, jobs)
  Path(out).write_text(json.dumps(res, indent=1))
  rows = {}
  for r in res:
    key = (r["scale"], r["vx_cmd"], r["delay_ms"])
    rows.setdefault(key, []).append(r)
  print(f"{'scale':12s} {'vx_cmd':>6s} {'delay':>5s} {'survive':>8s} {'vx_mean':>8s}")
  for (sc, v, dl), rs in sorted(rows.items()):
    ok = [r for r in rs if not r["fell"]]
    vm = np.mean([r["vx"] for r in ok]) if ok else float("nan")
    print(f"{sc:12s} {v:6.2f} {dl:5.0f} {len(ok):>4d}/{len(rs):<3d} {vm:8.3f}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["parity", "sweep"])
  ap.add_argument("--ckpt")
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--speeds", default="0,0.5,1.0,1.5,2.0")
  ap.add_argument("--delays", default="0,20,40")
  ap.add_argument("--seeds", type=int, default=10)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--deploy-scale", action="store_true", help="also run the shipped deploy.yaml flat 0.25 scale")
  ap.add_argument("--out")
  a = ap.parse_args()
  if a.mode == "parity":
    parity(a.ckpt, a.onnx)
  else:
    sv = [None, 0.25] if a.deploy_scale else [None]
    sweep(a.onnx, [float(x) for x in a.speeds.split(",")], [float(x) for x in a.delays.split(",")],
          a.seeds, a.seconds, sv, a.out)


if __name__ == "__main__":
  main()
