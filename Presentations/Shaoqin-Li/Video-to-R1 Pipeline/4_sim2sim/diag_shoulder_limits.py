"""Diagnostic: is the dance shoulder error at joint-limit frames, and is the hand still placed correctly?"""
import os
import numpy as np, mujoco
import sim2sim_v5_dr as V, sim2sim_r1_tennis_v4_standalone as H
env = V.Env(); b = env.base
hand = b.body("left_wrist_roll_link").id
q, hp = [], []
orig_fwd = mujoco.mj_forward
def fwd(m, d):
    orig_fwd(m, d)
    if m.nu == b.nu: q.append(d.qpos[env.qadr].copy()); hp.append(d.xpos[hand].copy())
mujoco.mj_forward = fwd
r = env.rollout(V.SCENARIOS["nominal"], 0)
mujoco.mj_forward = orig_fwd
q = np.array(q[1:len(env.jp)]); hp = np.array(hp[1:len(env.jp)]); n = min(len(q), env.T - 1)
q, hp = q[:n], hp[:n]; ref = env.jp[1:n+1]; refhand = env.bp[1:n+1, hand - env.BODY0]
rng = np.array([b.jnt_range[b.joint(j).id] for j in H.JOINT_NAMES])
L = [H.JOINT_NAMES.index(j) for j in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint")]
pinned = np.any((ref[:, L] <= rng[L, 0] + 0.02) | (ref[:, L] >= rng[L, 1] - 0.02), axis=1)
e = np.sqrt(((q[:, L] - ref[:, L])**2).mean(1))
print(f"frames {n}; left shoulder pinned at a limit in reference: {pinned.mean()*100:.0f}%")
print(f"left-shoulder RMS error  pinned {e[pinned].mean():.3f} rad   free {e[~pinned].mean():.3f} rad")
he = np.linalg.norm(hp - refhand, axis=1)
if os.environ.get("SAVE"):                                 # raw series for the paper figure
    np.savez(os.environ["SAVE"], q=q[:, L], ref=ref[:, L], rng=rng[L], pinned=pinned, hand_err=he, names=np.array([H.JOINT_NAMES[i] for i in L]))
print(f"left-hand position error (world)  pinned {he[pinned].mean()*100:.1f} cm   free {he[~pinned].mean()*100:.1f} cm   overall {he.mean()*100:.1f} cm")
