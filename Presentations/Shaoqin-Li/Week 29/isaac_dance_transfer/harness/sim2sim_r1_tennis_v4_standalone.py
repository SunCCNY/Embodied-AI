"""Sim2sim: run the mjlab-trained R1 tracking policy (exported ONNX) in plain MuJoCo (CPU, mj_step).

CORRECTED 2026-10-01. The earlier version of this docstring blamed a "Warp-vs-native solver discrepancy"; that was
wrong. The failure was the harness model: raw r1.xml has NO ground plane and the harness ran 0.002 s x 4 (125 Hz)
physics while training uses 0.005 s x 4 (50 Hz). With a floor + mjlab's contact overrides + 50 Hz physics the policy
tracks the full 10.4 s clip (mean joint err ~0.58 rad). See standalone_r1_model.py for the exact model settings and
sim2sim_v5_dr.py for randomized / stress evaluation.
"""

import numpy as np
import mujoco
import onnxruntime as ort

import mjlab  # noqa: F401  -- must import before mjlab.utils.spec (init order)
import mjlab.entity  # noqa: F401  -- resolves a circular import in utils.spec
from mjlab.utils.spec import create_position_actuator


ONNX_PATH = "logs/rsl_rl/r1_tracking/2026-09-24_17-01-04_r1_tennis_run1_continued/2026-09-24_17-01-04_r1_tennis_run1_continued.onnx"
MOTION_NPZ = __import__("os").environ.get("MOTION", "artifacts/r1_tennis:v0/motion.npz")
R1_XML = "src/mjlab/asset_zoo/robots/unitree_r1/xmls/r1.xml"
OUT_VIDEO = "sim2sim_r1_tennis_v2_timing.mp4"

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

  import os
  from standalone_r1_model import build_model
  KEEP = [k for k in os.environ.get("KEEP_RAW", "").split(",") if k]
  model = build_model(JOINT_NAMES, STIFFNESS, DAMPING, EFFORT_LIMIT, keep_raw=KEEP)
  print(f"[INFO] standalone model, keep_raw={KEEP}, timestep={model.opt.timestep}, ngeom={model.ngeom}")
  data = mujoco.MjData(model)

  joint_qpos_adr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
  joint_dof_adr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
  act_of_joint = np.array([[i for i in range(model.nu) if model.actuator_trnid[i][0] == model.joint(n).id][0] for n in JOINT_NAMES])
  # motion.npz's body arrays (shape (T, 25, ...)) exclude the world body,
  # while MuJoCo's own body.id includes it (world=0) -- subtract 1 to map
  # a MuJoCo body id onto the reference motion's body index.
  BODY0 = model.body("pelvis").id
  anchor_body_id = model.body(ANCHOR_BODY).id - BODY0

  # Initialize robot state at the reference motion's first frame (matches
  # eval's sampling_mode="start" / RSI-at-episode-start convention).
  # pelvis (the free-joint root body) gives qpos[0:7] directly.
  pelvis_id = 0
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

  sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
  input_names = [i.name for i in sess.get_inputs()]
  print(f"[INFO] ONNX inputs: {input_names}")

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
  control_hz = 50.0
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

  frames = []
  for t in range(n_frames):
    t_sec = t / control_hz
    ref_jp, ref_jv, ref_bp, ref_bq = interp_motion(t_sec)

    mujoco.mj_forward(model, data)  # refresh xpos/xquat/sensors to the CURRENT state
    robot_joint_pos = data.qpos[joint_qpos_adr].copy()
    robot_joint_vel = data.qvel[joint_dof_adr]

    robot_anchor_pos = data.xpos[anchor_body_id + BODY0].copy()
    robot_anchor_quat = data.xquat[anchor_body_id + BODY0].copy()
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
    time_step = np.array([[t + 1]], dtype=np.float32)

    feeds = {"obs": obs}
    if "time_step" in input_names:
      feeds["time_step"] = time_step
    outputs = sess.run(None, feeds)
    action = outputs[0][0]
    prev_action = action

    if t < 5 or t % 50 == 0:
      print(f"--- t={t} (t_sec={t_sec:.3f}) ---  action range: [{action.min():.3f}, {action.max():.3f}]  err={np.linalg.norm(robot_joint_pos - ref_jp):.4f} pelvis_z={data.qpos[2]:.3f} ref_z={ref_bp[0][2]:.3f}")

    target = DEFAULT_POS + action * ACTION_SCALE
    data.ctrl[act_of_joint] = target

    for _ in range(n_substeps):
      mujoco.mj_step(model, data)

    joint_pos_errors.append(np.linalg.norm(robot_joint_pos - ref_jp))


  joint_pos_errors = np.array(joint_pos_errors)
  print(f"[RESULT] mean joint_pos_error: {joint_pos_errors.mean():.4f} rad")
  print(f"[RESULT] max joint_pos_error:  {joint_pos_errors.max():.4f} rad")
  print(f"[RESULT] final joint_pos_error: {joint_pos_errors[-1]:.4f} rad")

  np.save(os.environ.get("OUT", "/tmp/v4_full.npy"), joint_pos_errors)


if __name__ == "__main__":
  main()
