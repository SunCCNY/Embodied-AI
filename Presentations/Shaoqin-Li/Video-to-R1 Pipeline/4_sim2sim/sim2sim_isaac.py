"""PhysX (Isaac Lab) -> plain MuJoCo sim2sim for the R1 dance tracker.

The Isaac policy sees/outputs joints in PhysX order (breadth-first, L/R interleaved); the harness model is in MJCF order.
Everything is mapped by NAME using isaac_info.json (dumped from the live Isaac env).

  python sim2sim_isaac.py validate                 # harness obs vs Isaac's own obs0 at frame 0
  python sim2sim_isaac.py sweep [scenarios,...]    # survival / joint error (OFF env var = reference frame offset)
  python sim2sim_isaac.py video <scenario> <seed> out.mp4
Env: ONNX (Isaac-order policy), MOTION (mjlab-order npz), INFO (isaac_info.json), OFF (frame offset, default 0).
"""
import copy
import json
import os
import sys

import numpy as np
import mujoco

os.environ.setdefault("MOTION", "motions/dancer_r1.npz")
import sim2sim_v5_dr as V  # noqa: E402
import sim2sim_r1_tennis_v4_standalone as H  # noqa: E402

H.MOTION_NPZ = os.environ["MOTION"]
INFO = json.load(open(os.environ.get("INFO", "isaac_info.json")))
OFF = int(os.environ.get("OFF", "0"))

ISAAC = INFO["joint_names"]
P = np.array([H.JOINT_NAMES.index(n) for n in ISAAC])  # isaac index i -> mjcf index P[i]
Q = np.argsort(P)                                      # mjcf index j -> isaac index Q[j]
assert np.allclose(np.array(INFO["default_joint_pos"]), H.DEFAULT_POS[P], atol=0.02), "default pose mismatch (Isaac adds +-0.01 encoder-bias noise at startup)"
assert np.allclose(np.array(INFO["action_scale"]), H.ACTION_SCALE[P], atol=2e-3), "action scale mismatch"


class IsaacEnv(V.Env):
    def obs_from(self, data, f, prev_action_isaac, bias=0.0):
        jpos = data.qpos[self.qadr].copy()
        jvel = data.qvel[self.dadr].copy()
        apos, aquat = data.xpos[self.anchor + self.BODY0].copy(), data.xquat[self.anchor + self.BODY0].copy()
        rel_p, rel_q = H.subtract_frame_transforms(apos, aquat, self.bp[f, self.anchor], self.bq[f, self.anchor])
        ori6 = H.matrix_from_quat_first_two_cols(rel_q)
        lin = data.sensordata[self.lin_adr:self.lin_adr + 3].copy()
        ang = data.sensordata[self.ang_adr:self.ang_adr + 3].copy()
        return jpos, jvel, rel_p, ori6, lin, ang

    def assemble(self, f, rel_p, ori6, lin, ang, jpos_rel, jvel, prev_action_isaac):
        return np.concatenate([self.jp[f][P], self.jv[f][P], rel_p, ori6, lin, ang,
                               jpos_rel[P], jvel[P], prev_action_isaac]).astype(np.float32).reshape(1, -1)

    def rollout(self, cfg, seed, hook=None, realtime=False):
        import time
        rng = np.random.default_rng(seed)
        U = lambda lo, hi, n=None: rng.uniform(lo, hi, n)
        model = copy.copy(self.base)
        if cfg["com"]:
            model.body_ipos[self.torso] += [U(-0.025, 0.025), U(-0.05, 0.05), U(-0.05, 0.05)]
        if cfg["fric"] is not None:
            mu = U(0.3, 1.2) if cfg["fric"] == "train" else float(cfg["fric"])
            for g in self.foot_geoms:
                model.geom_friction[g, 0] = mu
        if cfg["payload"]:
            model.body_mass[self.torso] += cfg["payload"]
        if cfg["kp_scale"] != 1.0:
            model.actuator_gainprm[:, 0] *= cfg["kp_scale"]
            model.actuator_biasprm[:, 1] *= cfg["kp_scale"]
        bias = U(-0.01, 0.01, 24) if cfg["enc_bias"] else np.zeros(24)
        data = mujoco.MjData(model)

        q0 = self.bq[0, 0].copy(); p0 = self.bp[0, 0].copy()
        lin0 = self.blv[0, 0].copy(); ang0 = self.bav[0, 0].copy(); jp0 = self.jp[0].copy()
        if cfg["reset_dr"]:
            PR, VR = V.POSE_RANGE, V.VEL_RANGE
            p0 += [U(*PR["x"]), U(*PR["y"]), U(*PR["z"])]
            q0 = H.quat_mul(V.rpy_to_quat(U(*PR["roll"]), U(*PR["pitch"]), U(*PR["yaw"])), q0)
            lin0 += [U(*VR[k]) for k in "xyz"]
            ang0 += [U(*VR[k]) for k in ("roll", "pitch", "yaw")]
            jp0 = np.clip(jp0 + U(-0.1, 0.1, 24), self.jrange[:, 0], self.jrange[:, 1])
        data.qpos[0:3] = p0; data.qpos[3:7] = q0
        data.qvel[0:3] = lin0; data.qvel[3:6] = H.quat_rotate_inv(q0, ang0)
        for i in range(24):
            data.qpos[self.qadr[i]] = jp0[i]; data.qvel[self.dadr[i]] = self.jv[0, i]
        mujoco.mj_forward(model, data)

        next_push = U(1.0, 3.0) if cfg["push"] else 1e9
        queue = [np.zeros(24)] * cfg["latency"]
        prev_action = np.zeros(24, np.float32)  # isaac order
        errs, aerrs = [], []
        reason, steps = None, 0
        for t in range(self.T - 1 - OFF):
            f = min(t + OFF, self.T - 1)
            mujoco.mj_forward(model, data)
            jpos, jvel, rel_p, ori6, lin, ang = self.obs_from(data, f, prev_action)
            jpos_obs = jpos + bias - H.DEFAULT_POS
            if cfg.get("zero_pos"):
                rel_p = np.zeros(3)
            if cfg.get("zero_lin"):
                lin = np.zeros(3)
            if cfg["obs_noise"]:
                n = V.OBS_NOISE
                rel_p = rel_p + U(-n["anchor_pos"], n["anchor_pos"], 3); ori6 = ori6 + U(-n["anchor_ori"], n["anchor_ori"], 6)
                lin = lin + U(-n["lin_vel"], n["lin_vel"], 3); ang = ang + U(-n["ang_vel"], n["ang_vel"], 3)
                jpos_obs = jpos_obs + U(-n["joint_pos"], n["joint_pos"], 24); jvel = jvel + U(-n["joint_vel"], n["joint_vel"], 24)
            obs = self.assemble(f, rel_p, ori6, lin, ang, jpos_obs, jvel, prev_action)
            action_i = self.sess.run(None, {"obs": obs})[0][0]
            prev_action = action_i
            action = action_i[Q]  # -> mjcf order
            queue.append(action); applied = queue.pop(0)
            data.ctrl[self.act] = H.DEFAULT_POS + applied * H.ACTION_SCALE - bias
            errs.append(np.linalg.norm(jpos - self.jp[f]))

            for _ in range(4):
                mujoco.mj_step(model, data)
            steps = t + 1

            if (t + 1) * 0.02 >= next_push:
                next_push += U(1.0, 3.0)
                s = cfg["push"]
                data.qvel[0:3] += s * np.array([U(*V.VEL_RANGE[k]) for k in "xyz"])
                dw = s * np.array([U(*V.VEL_RANGE[k]) for k in ("roll", "pitch", "yaw")])
                data.qvel[3:6] += H.quat_rotate_inv(data.qpos[3:7], dw)

            mujoco.mj_kinematics(model, data)
            fn = min(f + 1, self.T - 1)
            apos, aquat = data.xpos[self.anchor + self.BODY0], data.xquat[self.anchor + self.BODY0]
            aerrs.append(abs(self.bp[fn, self.anchor][2] - apos[2]))
            g = np.array([0, 0, -1.0])
            if abs(H.quat_rotate_inv(self.bq[fn, self.anchor], g)[2] - H.quat_rotate_inv(aquat, g)[2]) > 0.8:
                reason = "anchor_ori"
            elif aerrs[-1] > 0.25:
                reason = "anchor_z"
            elif any(abs(self.bp[fn, i][2] - data.xpos[i + self.BODY0][2]) > 0.25 for i in self.ee):
                reason = "ee_z"
            if hook:
                hook(t, model, data, fn)
            if realtime:
                time.sleep(0.02)
            if reason:
                break
        return dict(survived=reason is None, steps=steps, reason=reason, mean_err=float(np.mean(errs)),
                    max_anchor_z=float(np.max(aerrs)), errs=errs)


def validate():
    """Put the harness model in Isaac's frame-0 state and compare the observation to Isaac's own obs0."""
    env = IsaacEnv.__new__(IsaacEnv)
    from standalone_r1_model import build_model
    env.base = build_model(H.JOINT_NAMES, H.STIFFNESS, H.DAMPING, H.EFFORT_LIMIT)
    m = np.load(H.MOTION_NPZ)
    env.jp, env.jv, env.bp, env.bq = m["joint_pos"], m["joint_vel"], m["body_pos_w"], m["body_quat_w"]
    b = env.base
    env.BODY0 = b.body("pelvis").id
    env.anchor = b.body(H.ANCHOR_BODY).id - env.BODY0
    env.qadr = [b.joint(n).qposadr[0] for n in H.JOINT_NAMES]
    env.dadr = [b.joint(n).dofadr[0] for n in H.JOINT_NAMES]
    env.lin_adr = b.sensor_adr[b.sensor("imu_lin_vel").id]
    env.ang_adr = b.sensor_adr[b.sensor("imu_ang_vel").id]
    s = INFO["state0"]
    d = mujoco.MjData(b)
    d.qpos[0:3] = s["root_pos_w"]; d.qpos[3:7] = s["root_quat_w"]
    d.qvel[0:3] = s["root_lin_vel_w"]; d.qvel[3:6] = H.quat_rotate_inv(np.array(s["root_quat_w"]), np.array(s["root_ang_vel_w"]))
    jp_i, jv_i = np.array(s["joint_pos"]), np.array(s["joint_vel"])
    for k, n in enumerate(ISAAC):
        j = H.JOINT_NAMES.index(n)
        d.qpos[env.qadr[j]] = jp_i[k]
        d.qvel[env.dadr[j]] = jv_i[k]
    mujoco.mj_forward(b, d)
    jpos, jvel, rel_p, ori6, lin, ang = env.obs_from(d, s["time_step"], np.zeros(24))
    obs = env.assemble(s["time_step"], rel_p, ori6, lin, ang, jpos - np.array(INFO["default_joint_pos"])[Q], jvel, np.zeros(24))[0]
    ref = np.array(INFO["obs0"])
    names = ["cmd_jp", "cmd_jv", "anchor_pos_b", "anchor_ori_b", "lin_vel", "ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action"]
    sizes = [24, 24, 3, 6, 3, 3, 24, 24, 24]
    i = 0
    print(f"{'term':16s} {'max|harness-isaac|':>20s} {'max|isaac|':>12s}")
    for n, k in zip(names, sizes):
        print(f"{n:16s} {np.abs(obs[i:i+k]-ref[i:i+k]).max():20.5f} {np.abs(ref[i:i+k]).max():12.4f}")
        i += k


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "validate":
        validate()
    elif cmd == "sweep":
        V.Env = IsaacEnv
        V.sweep(only=sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    elif cmd == "video":
        V.Env = IsaacEnv
        V.video(sys.argv[2], int(sys.argv[3]), sys.argv[4])
