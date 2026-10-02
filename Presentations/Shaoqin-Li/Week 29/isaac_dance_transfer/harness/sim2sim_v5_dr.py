"""R1 tennis tracking policy in standalone plain MuJoCo, with the SAME randomization it was trained under
(plus optional out-of-distribution stress), termination rules copied from the training config, video, and live viewer.

  python sim2sim_v5_dr.py sweep                      # table of scenarios x seeds
  python sim2sim_v5_dr.py video  <scenario> <seed> out.mp4
  python sim2sim_v5_dr.py viewer <scenario> <seed>   # live MuJoCo window (needs a display, e.g. WSLg)
"""
import os
import copy, json, re, sys, time
import numpy as np
import mujoco
import onnxruntime as ort
import sim2sim_r1_tennis_v4_standalone as H            # constants + quaternion helpers (import order matters)
from standalone_r1_model import build_model

VEL_RANGE = dict(x=(-0.5, 0.5), y=(-0.5, 0.5), z=(-0.2, 0.2), roll=(-0.52, 0.52), pitch=(-0.52, 0.52), yaw=(-0.78, 0.78))
POSE_RANGE = dict(x=(-0.05, 0.05), y=(-0.05, 0.05), z=(-0.01, 0.01), roll=(-0.1, 0.1), pitch=(-0.1, 0.1), yaw=(-0.2, 0.2))
OBS_NOISE = dict(anchor_pos=0.25, anchor_ori=0.05, lin_vel=0.5, ang_vel=0.2, joint_pos=0.01, joint_vel=0.5)
FOOT = re.compile(r"^(left|right)_foot[1-7]_collision$")
EE = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")

NONE = dict(obs_noise=False, reset_dr=False, push=0.0, com=False, enc_bias=False, fric=None, latency=0, kp_scale=1.0, payload=0.0)
TRAIN = dict(NONE, obs_noise=True, reset_dr=True, push=1.0, com=True, enc_bias=True, fric="train")
SCENARIOS = {
    "nominal": NONE,
    "train_DR": TRAIN,
    "obs_noise_only": dict(NONE, obs_noise=True),
    "push_only": dict(NONE, push=1.0),
    "friction_only": dict(NONE, fric="train"),
    "train_DR+push2x": dict(TRAIN, push=2.0),
    "train_DR+push3x": dict(TRAIN, push=3.0),
    "train_DR+latency1": dict(TRAIN, latency=1),
    "train_DR+latency2": dict(TRAIN, latency=2),
    "train_DR+latency3": dict(TRAIN, latency=3),
    "train_DR+kp-20%": dict(TRAIN, kp_scale=0.8),
    "train_DR+kp+20%": dict(TRAIN, kp_scale=1.2),
    "train_DR+payload2kg": dict(TRAIN, payload=2.0),
    "train_DR+fric0.15": dict(TRAIN, fric=0.15),
    "train_DR+no_anchor_pos": dict(TRAIN, zero_pos=True),
    "train_DR+no_lin_vel": dict(TRAIN, zero_lin=True),
    "train_DR+no_pos_no_linvel": dict(TRAIN, zero_pos=True, zero_lin=True),
}


def rpy_to_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2), np.cos(y/2), np.sin(y/2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


class Env:
    def __init__(self):
        self.base = build_model(H.JOINT_NAMES, H.STIFFNESS, H.DAMPING, H.EFFORT_LIMIT)
        self.sess = ort.InferenceSession(os.environ.get("ONNX", H.ONNX_PATH), providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        m = np.load(H.MOTION_NPZ)
        self.jp, self.jv = m["joint_pos"], m["joint_vel"]
        self.bp, self.bq, self.blv, self.bav = m["body_pos_w"], m["body_quat_w"], m["body_lin_vel_w"], m["body_ang_vel_w"]
        self.T = self.jp.shape[0]
        b = self.base
        self.BODY0 = b.body("pelvis").id
        self.anchor = b.body(H.ANCHOR_BODY).id - self.BODY0
        self.ee = [b.body(n).id - self.BODY0 for n in EE]
        self.qadr = [b.joint(n).qposadr[0] for n in H.JOINT_NAMES]
        self.dadr = [b.joint(n).dofadr[0] for n in H.JOINT_NAMES]
        self.act = np.array([[i for i in range(b.nu) if b.actuator_trnid[i][0] == b.joint(n).id][0] for n in H.JOINT_NAMES])
        self.jrange = np.array([b.jnt_range[b.joint(n).id] for n in H.JOINT_NAMES])
        self.lin_adr = b.sensor_adr[b.sensor("imu_lin_vel").id]
        self.ang_adr = b.sensor_adr[b.sensor("imu_ang_vel").id]
        self.foot_geoms = [i for i in range(b.ngeom) if FOOT.match(mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
        self.torso = b.body("torso_link").id

    def set_ref_pose(self, model, data, f):
        """Put `data` at reference frame f (used for the ghost)."""
        data.qpos[0:3] = self.bp[f, 0]; data.qpos[3:7] = self.bq[f, 0]
        for i, a in enumerate(self.qadr):
            data.qpos[a] = self.jp[f, i]
        mujoco.mj_kinematics(model, data)

    def rollout(self, cfg, seed, hook=None, realtime=False):
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
            p0 += [U(*POSE_RANGE["x"]), U(*POSE_RANGE["y"]), U(*POSE_RANGE["z"])]
            q0 = H.quat_mul(rpy_to_quat(U(*POSE_RANGE["roll"]), U(*POSE_RANGE["pitch"]), U(*POSE_RANGE["yaw"])), q0)
            lin0 += [U(*VEL_RANGE[k]) for k in "xyz"]
            ang0 += [U(*VEL_RANGE[k]) for k in ("roll", "pitch", "yaw")]
            jp0 = np.clip(jp0 + U(-0.1, 0.1, 24), self.jrange[:, 0], self.jrange[:, 1])
        data.qpos[0:3] = p0; data.qpos[3:7] = q0
        data.qvel[0:3] = lin0; data.qvel[3:6] = H.quat_rotate_inv(q0, ang0)
        for i in range(24):
            data.qpos[self.qadr[i]] = jp0[i]; data.qvel[self.dadr[i]] = self.jv[0, i]
        mujoco.mj_forward(model, data)

        next_push = U(1.0, 3.0) if cfg["push"] else 1e9
        queue = [np.zeros(24)] * cfg["latency"]
        prev_action = np.zeros(24, np.float32)
        errs, aerrs = [], []
        reason, steps = None, 0
        for t in range(self.T - 1):
            f = min(t + 1, self.T - 1)
            mujoco.mj_forward(model, data)
            jpos = data.qpos[self.qadr].copy(); jvel = data.qvel[self.dadr].copy()
            apos, aquat = data.xpos[self.anchor + self.BODY0].copy(), data.xquat[self.anchor + self.BODY0].copy()
            rel_p, rel_q = H.subtract_frame_transforms(apos, aquat, self.bp[f, self.anchor], self.bq[f, self.anchor])
            ori6 = H.matrix_from_quat_first_two_cols(rel_q)
            lin = data.sensordata[self.lin_adr:self.lin_adr + 3].copy(); ang = data.sensordata[self.ang_adr:self.ang_adr + 3].copy()
            jpos_obs = jpos + bias - H.DEFAULT_POS
            if cfg.get("zero_pos"):
                rel_p = np.zeros(3)
            if cfg.get("zero_lin"):
                lin = np.zeros(3)
            if cfg["obs_noise"]:
                n = OBS_NOISE
                rel_p = rel_p + U(-n["anchor_pos"], n["anchor_pos"], 3); ori6 = ori6 + U(-n["anchor_ori"], n["anchor_ori"], 6)
                lin = lin + U(-n["lin_vel"], n["lin_vel"], 3); ang = ang + U(-n["ang_vel"], n["ang_vel"], 3)
                jpos_obs = jpos_obs + U(-n["joint_pos"], n["joint_pos"], 24); jvel = jvel + U(-n["joint_vel"], n["joint_vel"], 24)
            obs = np.concatenate([self.jp[f], self.jv[f], rel_p, ori6, lin, ang, jpos_obs, jvel, prev_action]).astype(np.float32).reshape(1, -1)
            feeds = {"obs": obs}
            if "time_step" in self.in_names:
                feeds["time_step"] = np.array([[t + 1]], np.float32)
            action = self.sess.run(None, feeds)[0][0]
            prev_action = action
            queue.append(action); applied = queue.pop(0)
            data.ctrl[self.act] = H.DEFAULT_POS + applied * H.ACTION_SCALE - bias
            errs.append(np.linalg.norm(jpos - self.jp[f]))

            for _ in range(4):
                mujoco.mj_step(model, data)
            steps = t + 1

            if (t + 1) * 0.02 >= next_push:
                next_push += U(1.0, 3.0)
                s = cfg["push"]
                data.qvel[0:3] += s * np.array([U(*VEL_RANGE[k]) for k in "xyz"])
                dw = s * np.array([U(*VEL_RANGE[k]) for k in ("roll", "pitch", "yaw")])
                data.qvel[3:6] += H.quat_rotate_inv(data.qpos[3:7], dw)

            mujoco.mj_kinematics(model, data)
            fn = min(t + 2, self.T - 1)
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
        return dict(survived=reason is None, steps=steps, reason=reason, mean_err=float(np.mean(errs)), max_anchor_z=float(np.max(aerrs)), errs=errs)


def sweep(n_train=30, n_other=20, only=None):
    env = Env()
    out = {}
    print(f"{'scenario':24s} {'seeds':>5s} {'survived':>9s} {'mean err (rad)':>15s} {'median steps':>13s}  failure reasons")
    for name, cfg in SCENARIOS.items():
        if only and name not in only:
            continue
        n = 1 if name == "nominal" else (n_train if name == "train_DR" else n_other)
        rs = [env.rollout(cfg, 1000 + s) for s in range(n)]
        surv = sum(r["survived"] for r in rs)
        why = {}
        for r in rs:
            if r["reason"]:
                why[r["reason"]] = why.get(r["reason"], 0) + 1
        me = np.mean([r["mean_err"] for r in rs])
        print(f"{name:24s} {n:5d} {surv:4d}/{n:<4d} {me:15.3f} {int(np.median([r['steps'] for r in rs])):13d}  {why}", flush=True)
        out[name] = [{k: v for k, v in r.items() if k != "errs"} for r in rs]
    json.dump(out, open("results_v5_dr_%s%s.json" % ("subset" if only else "full", os.environ.get("TAG", "")), "w"), indent=1)


def video(scn, seed, path):
    import imageio
    env = Env()
    cfg = SCENARIOS[scn]
    frames = []
    state = {}

    def hook(t, model, data, fn):
        if "r" not in state:
            state["r"] = mujoco.Renderer(model, 480, 640)
            state["cam"] = mujoco.MjvCamera(); state["cam"].distance = 3.2; state["cam"].azimuth = 120; state["cam"].elevation = -8
            state["ghost"] = mujoco.MjData(model); state["opt"] = mujoco.MjvOption(); state["pert"] = mujoco.MjvPerturb()
        r, cam = state["r"], state["cam"]
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.7]
        r.update_scene(data, cam)
        env.set_ref_pose(model, state["ghost"], fn)
        n0 = r.scene.ngeom
        mujoco.mjv_addGeoms(model, state["ghost"], state["opt"], state["pert"], mujoco.mjtCatBit.mjCAT_DYNAMIC.value, r.scene)
        for i in range(n0, r.scene.ngeom):
            r.scene.geoms[i].rgba[:] = [0.1, 0.9, 0.2, 0.30]
        frames.append(r.render().copy())

    res = env.rollout(cfg, seed, hook=hook)
    imageio.mimsave(path, frames, fps=50, macro_block_size=1)
    print(f"[VIDEO] {path}: {len(frames)} frames ({len(frames)/50:.1f}s)  survived={res['survived']} reason={res['reason']} mean_err={res['mean_err']:.3f}")


def viewer(scn, seed):
    import mujoco.viewer
    env = Env()
    cfg = SCENARIOS[scn]
    m0 = copy.copy(env.base)
    d0 = mujoco.MjData(m0)
    with mujoco.viewer.launch_passive(m0, d0) as v:
        v.cam.distance = 3.2; v.cam.azimuth = 120; v.cam.elevation = -8
        ghost = mujoco.MjData(m0); opt = mujoco.MjvOption(); pert = mujoco.MjvPerturb()
        k = 0
        while v.is_running():
            def hook(t, model, data, fn):
                d0.qpos[:] = data.qpos; d0.qvel[:] = data.qvel
                mujoco.mj_kinematics(m0, d0)
                v.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.7]
                env.set_ref_pose(m0, ghost, fn)
                v.user_scn.ngeom = 0
                mujoco.mjv_addGeoms(m0, ghost, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC.value, v.user_scn)
                for i in range(v.user_scn.ngeom):
                    v.user_scn.geoms[i].rgba[:] = [0.1, 0.9, 0.2, 0.30]
                v.sync()
                if not v.is_running():
                    raise SystemExit
            res = env.rollout(cfg, seed + k, hook=hook, realtime=True)
            print(f"[VIEWER] loop {k}: survived={res['survived']} reason={res['reason']} steps={res['steps']}", flush=True)
            k += 1
            time.sleep(1.0)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sweep":
        sweep(only=sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    elif cmd == "video":
        video(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "viewer":
        viewer(sys.argv[2], int(sys.argv[3]))
