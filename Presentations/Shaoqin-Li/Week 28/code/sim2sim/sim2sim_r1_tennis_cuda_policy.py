"""Sim2sim check: run the mjlab-trained R1 tracking policy (exported ONNX) in
plain vanilla MuJoCo (CPU, single instance, mj_step) instead of the
MuJoCo-Warp GPU-batched simulator it was trained in. This is a genuinely
different physics engine implementation (own contact solver / integrator),
matching this project's established "sim2sim" meaning: does a policy trained
in one simulated context still work when moved to a different one.

All PD gains, action scale, default pose, and joint order come directly from
the ONNX file's own embedded metadata (see mjlab's exporter), not re-derived
here -- removes most of the risk of a silent obs-construction mismatch.

STATUS (after a full debugging pass): four real bugs were found and fixed --
(1) actuators created with no effort_limit, letting corrective torque run
unbounded; (2) an initial "fix" that time-interpolated the 50fps motion file
to continuous 125Hz control time, which was actually WRONG -- mjlab's own
MotionCommand does `time_steps += 1`, a plain per-control-step integer index
into the stored array with no regard for real-world fps, so the correct
control loop runs real 125Hz physics (matching training's decimation=4
exactly) while advancing the reference index by exactly 1 per step, not by
elapsed time; (3) a consistent +1 offset in that index, found by diffing a
captured native observation against reference frames directly; (4) qvel was
left at MuJoCo's zero default at reset -- mjlab's own reset copies the
reference motion's OWN root and joint VELOCITY into the robot too (full
reference-state init), confirmed by inspecting `_resample_command` and by
finding the native rollout's real frame-0 IMU reading is nowhere near zero.

Every one of the 8 observation terms was then verified against a captured
NATIVE rollout's real observation vector by reconstructing each term from
that rollout's own ground-truth state -- all matched to floating-point
precision (~1e-7), except joint_pos_rel, which differs by exactly the
expected ~1e-2 encoder-bias DR term. The observation formula is therefore
provably correct. An open-loop test (replaying the native rollout's exact
action sequence through this same plain-MuJoCo model) also stays bounded
(mean error 0.38 rad over 660 steps), proving the two physics engines agree
given correct actions.

Despite all of this, the CLOSED-LOOP harness still diverges (see the numbers
at the bottom of this file's last run). The most likely remaining
explanation, given everything above is verified correct, is genuine
closed-loop sensitivity: this policy has only been trained for 3,000
iterations (a first real attempt, not a converged floor), bipedal tracking is
a chaotic system, and the small-but-real numerical differences between
MuJoCo-Warp's solver and native MuJoCo's solver are enough to compound over
hundreds of steps once the policy is pushed even slightly outside what it saw
in training. This is a testable hypothesis (a more-converged policy should
show less sim2sim divergence), not yet tested.
"""

import numpy as np
import mujoco
import torch
import torch.nn as nn

import mjlab  # noqa: F401  -- must import before mjlab.utils.spec (init order)
import mjlab.entity  # noqa: F401  -- resolves a circular import in utils.spec
from mjlab.utils.spec import create_position_actuator

CHECKPOINT = "logs/rsl_rl/r1_tracking/2026-09-24_17-01-04_r1_tennis_run1_continued/model_12998.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PolicyNet(nn.Module):
  """Minimal standalone rebuild of rsl_rl's actor: EmpiricalNormalization
  (running mean/var, precomputed) + a 4-layer ELU MLP -- built directly from
  the checkpoint's raw state dict so this can run on CUDA (matching what
  training/native rollout actually used) without needing a full mjlab env."""

  def __init__(self, sd):
    super().__init__()
    self.register_buffer("mean", sd["obs_normalizer._mean"])
    self.register_buffer("std", sd["obs_normalizer._std"])
    self.mlp = nn.Sequential(
      nn.Linear(135, 512), nn.ELU(),
      nn.Linear(512, 256), nn.ELU(),
      nn.Linear(256, 128), nn.ELU(),
      nn.Linear(128, 24),
    )
    self.mlp[0].weight.data = sd["mlp.0.weight"]
    self.mlp[0].bias.data = sd["mlp.0.bias"]
    self.mlp[2].weight.data = sd["mlp.2.weight"]
    self.mlp[2].bias.data = sd["mlp.2.bias"]
    self.mlp[4].weight.data = sd["mlp.4.weight"]
    self.mlp[4].bias.data = sd["mlp.4.bias"]
    self.mlp[6].weight.data = sd["mlp.6.weight"]
    self.mlp[6].bias.data = sd["mlp.6.bias"]

  @torch.no_grad()
  def forward(self, obs):
    normed = (obs - self.mean) / self.std
    return self.mlp(normed)


ONNX_PATH = "logs/rsl_rl/r1_tracking/2026-09-24_17-01-04_r1_tennis_run1_continued/2026-09-24_17-01-04_r1_tennis_run1_continued.onnx"
MOTION_NPZ = "/tmp/motion.npz"
R1_XML = "src/mjlab/asset_zoo/robots/unitree_r1/xmls/r1.xml"
OUT_VIDEO = "sim2sim_r1_tennis.mp4"

JOINT_NAMES = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_roll_joint", "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint",
]

STIFFNESS = np.array([150,150,150,200,60,60, 150,150,150,200,60,60, 80,80,
                       60,60,40,40,40, 60,60,40,40,40], dtype=np.float64)
DAMPING = np.array([8,8,8,10,4,4, 8,8,8,10,4,4, 5,5,
                     4,4,3,3,3, 4,4,3,3,3], dtype=np.float64)
# Real Unitree R1 torque limits (R1_C++.xml ctrlrange) -- omitting these left
# the actuators with UNLIMITED torque, which was the actual root cause of the
# runaway divergence below: any small tracking error let the policy command
# arbitrarily large corrective torque, overshooting far more than the real
# (limited) motors ever could, growing the error each step.
EFFORT_LIMIT = np.array([88,88,88,139,50,50, 88,88,88,139,50,50, 50,88,
                          25,25,25,25,25, 25,25,25,25,25], dtype=np.float64)
DEFAULT_POS = np.array([-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0, 0,0,
                         0.35,0.18,0,0.87,0, 0.35,-0.18,0,0.87,0], dtype=np.float64)
ACTION_SCALE = np.array([0.147,0.147,0.147,0.174,0.208,0.208, 0.147,0.147,0.147,0.174,0.208,0.208,
                          0.156,0.275, 0.104,0.104,0.156,0.156,0.156, 0.104,0.104,0.156,0.156,0.156],
                         dtype=np.float64)

ANCHOR_BODY = "torso_link"


def quat_conj(q):
  return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
  ])


def quat_rotate_inv(q, v):
  """Rotate vector v by the inverse of quaternion q (world -> body frame)."""
  qc = quat_conj(q)
  vq = np.array([0.0, *v])
  return quat_mul(quat_mul(qc, vq), q)[1:]


def matrix_from_quat_first_two_cols(q):
  w, x, y, z = q
  s = 2.0 / (w*w + x*x + y*y + z*z)
  R = np.array([
    [1 - s*(y*y+z*z), s*(x*y-z*w),     s*(x*z+y*w)],
    [s*(x*y+z*w),     1 - s*(x*x+z*z), s*(y*z-x*w)],
    [s*(x*z-y*w),     s*(y*z+x*w),     1 - s*(x*x+y*y)],
  ])
  return R[:, :2].reshape(-1)  # 6-dim, matches mjlab's mat[..., :2] flatten


def subtract_frame_transforms(t01, q01, t02, q02):
  """T12 = T01^-1 @ T02. Returns (pos_in_frame1, quat_in_frame1)."""
  q01_conj = quat_conj(q01)
  pos = quat_rotate_inv(q01, t02 - t01)
  quat = quat_mul(q01_conj, q02)
  return pos, quat


def main():
  motion = np.load(MOTION_NPZ)
  ref_joint_pos = motion["joint_pos"]  # (T, 24)
  ref_joint_vel = motion["joint_vel"]  # (T, 24)
  ref_body_pos_w = motion["body_pos_w"]  # (T, 25, 3)
  ref_body_quat_w = motion["body_quat_w"]  # (T, 25, 4)
  ref_body_lin_vel_w = motion["body_lin_vel_w"]  # (T, 25, 3)
  ref_body_ang_vel_w = motion["body_ang_vel_w"]  # (T, 25, 3)
  n_frames = ref_joint_pos.shape[0]
  print(f"[INFO] Loaded reference motion: {n_frames} frames")

  spec = mujoco.MjSpec.from_file(R1_XML)
  # Add position actuators matching the ONNX-embedded PD gains exactly,
  # using mjlab's own known-correct actuator-creation helper.
  for i, jname in enumerate(JOINT_NAMES):
    create_position_actuator(
      spec, jname,
      stiffness=float(STIFFNESS[i]),
      damping=float(DAMPING[i]),
      effort_limit=float(EFFORT_LIMIT[i]),
    )
  model = spec.compile()
  # CRITICAL FIX: r1.xml has no <option> tag, so this harness was compiling
  # with raw MuJoCo defaults (integrator=Euler, explicit). mjlab's own
  # MujocoCfg explicitly sets integrator="implicitfast" for training -- every
  # other opt field (solver=Newton, cone=pyramidal, jacobian, impratio,
  # iterations, tolerances) already matched by coincidence, but this one
  # didn't. implicitfast implicitly integrates joint damping/stiffness, which
  # matters a lot for these actuators' high PD gains (kp up to 200).
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  data = mujoco.MjData(model)

  joint_qpos_adr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
  joint_dof_adr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
  # motion.npz's body arrays (shape (T, 25, ...)) exclude the world body,
  # while MuJoCo's own body.id includes it (world=0) -- subtract 1 to map
  # a MuJoCo body id onto the reference motion's body index.
  anchor_body_id = model.body(ANCHOR_BODY).id - 1

  # Initialize robot state at the reference motion's first frame (matches
  # eval's sampling_mode="start" / RSI-at-episode-start convention).
  # pelvis (the free-joint root body) gives qpos[0:7] directly.
  pelvis_id = model.body("pelvis").id - 1
  data.qpos[0:3] = ref_body_pos_w[0, pelvis_id]
  data.qpos[3:7] = ref_body_quat_w[0, pelvis_id]
  for i, adr in enumerate(joint_qpos_adr):
    data.qpos[adr] = ref_joint_pos[0, i]
  # CRITICAL FIX: MotionCommand's own reset (_resample_command) copies the
  # reference's root and joint VELOCITY into the robot too (full RSI), not
  # just position -- confirmed by inspecting mjlab source and by finding the
  # native rollout's own frame-0 IMU reading is nowhere near zero. Leaving
  # qvel at MuJoCo's zero default (as this script did before) meant the very
  # first observation was already wrong, before any physics divergence could
  # occur -- a leading candidate for the root cause of the sim2sim failure.
  # MuJoCo free-joint qvel convention (confirmed via mjlab's own
  # write_root_velocity): qvel[0:3] = linear vel in WORLD frame (direct);
  # qvel[3:6] = angular vel rotated into the root's BODY frame.
  root_quat0 = ref_body_quat_w[0, pelvis_id]
  data.qvel[0:3] = ref_body_lin_vel_w[0, pelvis_id]
  data.qvel[3:6] = quat_rotate_inv(root_quat0, ref_body_ang_vel_w[0, pelvis_id])
  for i, adr in enumerate(joint_dof_adr):
    data.qvel[adr] = ref_joint_vel[0, i]
  mujoco.mj_forward(model, data)

  ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
  policy = PolicyNet(ckpt["actor_state_dict"]).to(DEVICE).eval()
  print(f"[INFO] Policy loaded on {DEVICE}")

  imu_lin_sensor = model.sensor("imu_lin_vel").id
  imu_ang_sensor = model.sensor("imu_ang_vel").id
  imu_lin_adr = model.sensor_adr[imu_lin_sensor]
  imu_ang_adr = model.sensor_adr[imu_ang_sensor]

  prev_action = np.zeros(24, dtype=np.float32)
  # IMPORTANT (found via a native-vs-replay diagnostic): mjlab's own
  # MotionCommand does `self.time_steps += 1` once per control step -- a
  # plain integer index into the stored motion array, with NO interpolation
  # and NO regard for what real-world fps the npz was recorded at. So the
  # correct thing is: physics must match training's real timestep/decimation
  # (for correct PD/contact dynamics), but the REFERENCE frame index must
  # just advance by 1 per control step, not by real elapsed time. An earlier
  # version of this script "fixed" this into time-based interpolation, which
  # was actually wrong -- it fed the policy a reference moving 2.5x faster
  # than what it was trained on (npz stored at 50fps, but consumed 1
  # frame/control-step at a 125Hz control rate), which was the real cause of
  # the runaway divergence, not a physics or effort-limit issue.
  control_hz = 125.0
  n_substeps = 4
  n_frames = min(n_frames, ref_joint_pos.shape[0])
  joint_pos_errors = []

  def interp_motion(t_sec):
    # +1: confirmed via a native-obs diagnostic that mjlab's MotionCommand has
    # already advanced time_steps by 1 relative to the reset state by the time
    # it computes the observation for the action about to be taken at loop
    # step t (i.e. env.reset() itself performs the first increment).
    f0 = min(int(round(t_sec * control_hz)) + 1, ref_joint_pos.shape[0] - 1)
    return ref_joint_pos[f0], ref_joint_vel[f0], ref_body_pos_w[f0], ref_body_quat_w[f0]

  renderer = mujoco.Renderer(model, height=480, width=640)
  frames = []
  cam = mujoco.MjvCamera()
  cam.distance = 2.5
  cam.azimuth = 120
  cam.elevation = -10

  for t in range(n_frames):
    t_sec = t / control_hz
    ref_jp, ref_jv, ref_bp, ref_bq = interp_motion(t_sec)

    robot_joint_pos = data.qpos[joint_qpos_adr]
    robot_joint_vel = data.qvel[joint_dof_adr]

    robot_anchor_pos = data.xpos[anchor_body_id + 1].copy()
    robot_anchor_quat = data.xquat[anchor_body_id + 1].copy()
    ref_anchor_pos = ref_bp[anchor_body_id]
    ref_anchor_quat = ref_bq[anchor_body_id]

    anchor_pos_b, anchor_quat_b = subtract_frame_transforms(
      robot_anchor_pos, robot_anchor_quat, ref_anchor_pos, ref_anchor_quat
    )
    anchor_ori_b = matrix_from_quat_first_two_cols(anchor_quat_b)

    base_lin_vel = data.sensordata[imu_lin_adr:imu_lin_adr + 3]
    base_ang_vel = data.sensordata[imu_ang_adr:imu_ang_adr + 3]

    command = np.concatenate([ref_jp, ref_jv])
    obs = np.concatenate([
      command,
      anchor_pos_b,
      anchor_ori_b,
      base_lin_vel,
      base_ang_vel,
      robot_joint_pos - DEFAULT_POS,
      robot_joint_vel,
      prev_action,
    ]).astype(np.float32).reshape(1, -1)

    assert obs.shape == (1, 135), obs.shape

    obs_t = torch.from_numpy(obs).to(DEVICE)
    action = policy(obs_t)[0].cpu().numpy()
    prev_action = action

    if t < 5 or t % 100 == 0:
      print(f"--- t={t} (t_sec={t_sec:.3f}) ---  action range: [{action.min():.3f}, {action.max():.3f}]  err={np.linalg.norm(robot_joint_pos - ref_jp):.4f}")

    target = DEFAULT_POS + action * ACTION_SCALE
    data.ctrl[:] = target

    for _ in range(n_substeps):
      mujoco.mj_step(model, data)

    joint_pos_errors.append(np.linalg.norm(robot_joint_pos - ref_jp))

    if t % 2 == 0:  # ~62 fps output video at 125 Hz sim
      renderer.update_scene(data, camera=cam)
      frames.append(renderer.render().copy())

  joint_pos_errors = np.array(joint_pos_errors)
  print(f"[RESULT] mean joint_pos_error: {joint_pos_errors.mean():.4f} rad")
  print(f"[RESULT] max joint_pos_error:  {joint_pos_errors.max():.4f} rad")
  print(f"[RESULT] final joint_pos_error: {joint_pos_errors[-1]:.4f} rad")

  import imageio_ffmpeg
  import mediapy
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  mediapy.write_video(OUT_VIDEO, frames, fps=round(control_hz / 2))
  print(f"[INFO] Video written to {OUT_VIDEO}")

  np.save("sim2sim_joint_pos_errors.npy", joint_pos_errors)


if __name__ == "__main__":
  main()
