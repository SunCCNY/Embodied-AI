"""Open-loop replay: feed the EXACT action sequence captured from mjlab's native
MuJoCo-Warp rollout through plain MuJoCo (CPU, mj_step), starting from the same
initial state. If plain-MuJoCo physics reproduces the same joint trajectory,
the two physics engines agree and the sim2sim bug is entirely in observation
reconstruction. If it diverges even here, it's a physics/model config mismatch.
"""
import numpy as np
import mujoco

import mjlab  # noqa: F401
import mjlab.entity  # noqa: F401
from mjlab.utils.spec import create_position_actuator

R1_XML = "src/mjlab/asset_zoo/robots/unitree_r1/xmls/r1.xml"

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
EFFORT_LIMIT = np.array([88,88,88,139,50,50, 88,88,88,139,50,50, 50,88,
                          25,25,25,25,25, 25,25,25,25,25], dtype=np.float64)
DEFAULT_POS = np.array([-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0, 0,0,
                         0.35,0.18,0,0.87,0, 0.35,-0.18,0,0.87,0], dtype=np.float64)
ACTION_SCALE = np.array([0.147,0.147,0.147,0.174,0.208,0.208, 0.147,0.147,0.147,0.174,0.208,0.208,
                          0.156,0.275, 0.104,0.104,0.156,0.156,0.156, 0.104,0.104,0.156,0.156,0.156],
                         dtype=np.float64)

data_log = np.load("/tmp/native_rollout.npz")
actions = data_log["actions"]  # (T, 24)
native_joint_pos = data_log["joint_pos"]  # (T, 24)
native_root_pos = data_log["root_pos"]  # (T, 3)
native_root_quat = data_log["root_quat"]  # (T, 4)
T = actions.shape[0]
print(f"Loaded {T} native steps")

spec = mujoco.MjSpec.from_file(R1_XML)
for i, jname in enumerate(JOINT_NAMES):
  create_position_actuator(
    spec, jname, stiffness=float(STIFFNESS[i]), damping=float(DAMPING[i]),
    effort_limit=float(EFFORT_LIMIT[i]),
  )
model = spec.compile()
data = mujoco.MjData(model)

joint_qpos_adr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
joint_dof_adr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]

# Initialize to the native rollout's own first-frame state (same starting point).
data.qpos[0:3] = native_root_pos[0]
data.qpos[3:7] = native_root_quat[0]
for i, adr in enumerate(joint_qpos_adr):
  data.qpos[adr] = native_joint_pos[0, i]
mujoco.mj_forward(model, data)

n_substeps = 4  # 125 Hz control, matching training exactly (no interpolation needed
                 # here since we're replaying the exact 125Hz-captured action sequence)

errors = []
for t in range(T):
  target = DEFAULT_POS + actions[t] * ACTION_SCALE
  data.ctrl[:] = target
  for _ in range(n_substeps):
    mujoco.mj_step(model, data)
  replay_joint_pos = data.qpos[joint_qpos_adr]
  err = np.linalg.norm(replay_joint_pos - native_joint_pos[t])
  errors.append(err)
  if t < 5 or t % 100 == 0:
    print(f"t={t}  replay_vs_native joint_pos_err={err:.4f}")

errors = np.array(errors)
print(f"\n[RESULT] mean error vs native: {errors.mean():.4f}")
print(f"[RESULT] max error vs native:  {errors.max():.4f}")
print(f"[RESULT] final error vs native: {errors[-1]:.4f}")
