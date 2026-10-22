"""Hub-switch probe: run tracking policy A to the end of its motion, then continue the SAME physics state into
policy B's motion (re-anchored to where the robot is), with one of three transitions.
  direct : B starts immediately at its frame 0
  bridge : reference ramps (1 s, smoothstep) from the robot's current pose to B frame 0, tracked by B's policy
  stand  : current -> default stand (1 s) -> hold (0.5 s) -> B frame 0 (1 s), tracked by B's policy

  python switch_probe.py <A> <B> <variant> <scenario nominal|train_DR> <nseeds>   (A,B in tennis,dance,tennis_delay,...)
"""
import copy, json, os, sys
import numpy as np
import mujoco
import onnxruntime as ort
import sim2sim_r1_tennis_v4_standalone as H
from standalone_r1_model import build_model
import sim2sim_v5_dr as S

P = os.path.expanduser("~/claude_code_projects/embodied_ai_r1/video_to_r1_pipeline")
POL = {
    "tennis": (f"{P}/policies/mjlab_tennis/2026-09-24_17-01-04_r1_tennis_run1_continued.onnx", f"{P}/data/r1_motions/tennis_motion.npz", 0),
    "tennis_delay": (f"{P}/policies/mjlab_tennis_delay/2026-10-01_17-41-06_r1_tennis_delay_ft.onnx", f"{P}/data/r1_motions/tennis_motion.npz", 8),
    "dance": (f"{P}/policies/mjlab_dance/2026-09-24_09-20-53_r1_dancer_run1.onnx", f"{P}/data/r1_motions/dancer_motion.npz", 0),
}
HOOK = None
SEG = [""]
SCN = {"nominal": S.NONE, "train_DR": S.TRAIN}


def slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def rot_z(a, v):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


class Ref:
    """Reference arrays used by the policy and terminations."""
    def __init__(self, jp, jv, ap, aq, eez):
        self.jp, self.jv, self.ap, self.aq, self.eez, self.T = jp, jv, ap, aq, eez, len(jp)


class Pol:
    def __init__(self, name):
        onnx, motion, dphys = POL[name]
        os.environ["ONNX"] = onnx
        H.MOTION_NPZ = motion
        self.env = S.Env()
        e = self.env
        self.dphys = dphys
        self.name = name
        self.ref = Ref(e.jp, e.jv, e.bp[:, e.anchor], e.bq[:, e.anchor], np.stack([e.bp[:, i, 2] for i in e.ee], 1))
        self.sess, self.in_names = e.sess, e.in_names


def reanchor(ref, target_xy, target_yaw):
    """Rigid yaw + xy move so ref frame 0 anchor sits at (target_xy, target_yaw)."""
    dyaw = target_yaw - yaw_of(ref.aq[0])
    ap = np.array([rot_z(dyaw, p - np.array([ref.ap[0][0], ref.ap[0][1], 0])) + [target_xy[0], target_xy[1], 0] for p in ref.ap])
    ap[:, 2] = ref.ap[:, 2]
    aq = np.array([S.H.quat_mul(yaw_quat(dyaw), q) for q in ref.aq])
    return Ref(ref.jp, ref.jv, ap, aq, ref.eez)


def ramp(jp0, ap0, aq0, eez0, jp1, ap1, aq1, eez1, n):
    u = np.linspace(0, 1, n + 1)[1:]
    s = u * u * (3 - 2 * u)
    ds = 6 * u * (1 - u) / (n * 0.02)
    jp = jp0 + s[:, None] * (jp1 - jp0)
    jv = ds[:, None] * (jp1 - jp0)
    ap = ap0 + s[:, None] * (ap1 - ap0)
    aq = np.array([slerp(aq0, aq1, x) for x in s])
    eez = eez0 + s[:, None] * (eez1 - eez0)
    return Ref(jp, jv, ap, aq, eez)


def hold(ref_pt, n):
    jp, ap, aq, eez = ref_pt
    return Ref(np.tile(jp, (n, 1)), np.zeros((n, 24)), np.tile(ap, (n, 1)), np.tile(aq, (n, 1)), np.tile(eez, (n, 1)))


def cat(refs):
    return Ref(*[np.concatenate([getattr(r, k) for r in refs]) for k in ("jp", "jv", "ap", "aq", "eez")])


def stand_pose(env):
    """Kinematic upright default stand: torso pose + ee heights with feet just touching the floor (pelvis xy 0, yaw 0)."""
    m = copy.copy(env.base); d = mujoco.MjData(m)
    d.qpos[3] = 1.0
    for i in range(24):
        d.qpos[env.qadr[i]] = H.DEFAULT_POS[i]
    z = 0.8
    for _ in range(5):
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        dmin = min(d.contact[i].dist for i in range(d.ncon)) if d.ncon else 0.0
        z -= dmin
    d.qpos[2] = z
    mujoco.mj_forward(m, d)
    ap = d.xpos[env.anchor + env.BODY0].copy() - np.array([0, 0, 0.0])
    aq = d.xquat[env.anchor + env.BODY0].copy()
    eez = np.array([d.xpos[i + env.BODY0][2] for i in env.ee])
    return ap, aq, eez, 0.0, float(z)


class Runner:
    def __init__(self, cfg, seed, model, env):
        self.cfg, self.env = cfg, env
        self.rng = np.random.default_rng(seed)
        U = self.U = lambda lo, hi, n=None: self.rng.uniform(lo, hi, n)
        if cfg["com"]:
            model.body_ipos[env.torso] += [U(-0.025, 0.025), U(-0.05, 0.05), U(-0.05, 0.05)]
        if cfg["fric"] is not None:
            mu = U(0.3, 1.2) if cfg["fric"] == "train" else float(cfg["fric"])
            for g in env.foot_geoms:
                model.geom_friction[g, 0] = mu
        self.bias = U(-0.01, 0.01, 24) if cfg["enc_bias"] else np.zeros(24)
        self.model = model
        self.next_push = U(1.0, 3.0) if cfg["push"] else 1e9
        self.prev_action = np.zeros(24, np.float32)
        self.t_global = 0

    def run(self, data, pol, ref, kinds=None, tag="", ee_check=True):
        """Track `ref` with `pol`; returns dict(reason, errs). time_step restarts at 1 for each call."""
        env, cfg, model, U = self.env, self.cfg, self.model, self.U
        H_ = S.H
        dphys = pol.dphys
        ctrl_hist = [H_.DEFAULT_POS - self.bias] * dphys
        errs = []
        reason = None
        for t in range(ref.T - 1):
            f = min(t + 1, ref.T - 1)
            mujoco.mj_forward(model, data)
            jpos = data.qpos[env.qadr].copy(); jvel = data.qvel[env.dadr].copy()
            apos, aquat = data.xpos[env.anchor + env.BODY0].copy(), data.xquat[env.anchor + env.BODY0].copy()
            rel_p, rel_q = H_.subtract_frame_transforms(apos, aquat, ref.ap[f], ref.aq[f])
            ori6 = H_.matrix_from_quat_first_two_cols(rel_q)
            lin = data.sensordata[env.lin_adr:env.lin_adr + 3].copy(); ang = data.sensordata[env.ang_adr:env.ang_adr + 3].copy()
            jpos_obs = jpos + self.bias - H_.DEFAULT_POS
            if cfg["obs_noise"]:
                n = S.OBS_NOISE
                rel_p = rel_p + U(-n["anchor_pos"], n["anchor_pos"], 3); ori6 = ori6 + U(-n["anchor_ori"], n["anchor_ori"], 6)
                lin = lin + U(-n["lin_vel"], n["lin_vel"], 3); ang = ang + U(-n["ang_vel"], n["ang_vel"], 3)
                jpos_obs = jpos_obs + U(-n["joint_pos"], n["joint_pos"], 24); jvel = jvel + U(-n["joint_vel"], n["joint_vel"], 24)
            obs = np.concatenate([ref.jp[f], ref.jv[f], rel_p, ori6, lin, ang, jpos_obs, jvel, self.prev_action]).astype(np.float32).reshape(1, -1)
            feeds = {"obs": obs}
            if "time_step" in pol.in_names:
                feeds["time_step"] = np.array([[t + 1]], np.float32)
            action = pol.sess.run(None, feeds)[0][0]
            self.prev_action = action
            target = H_.DEFAULT_POS + action * H_.ACTION_SCALE - self.bias
            errs.append(float(np.linalg.norm(jpos - ref.jp[f])))
            for _ in range(4):
                ctrl_hist.append(target)
                data.ctrl[env.act] = ctrl_hist.pop(0) if dphys else target
                mujoco.mj_step(model, data)
            self.t_global += 1
            if self.t_global * 0.02 >= self.next_push:
                self.next_push += U(1.0, 3.0)
                s = cfg["push"]
                data.qvel[0:3] += s * np.array([U(*S.VEL_RANGE[k]) for k in "xyz"])
                dw = s * np.array([U(*S.VEL_RANGE[k]) for k in ("roll", "pitch", "yaw")])
                data.qvel[3:6] += H_.quat_rotate_inv(data.qpos[3:7], dw)
            mujoco.mj_kinematics(model, data)
            if HOOK:
                HOOK(model, data, SEG[0])
            apos, aquat = data.xpos[env.anchor + env.BODY0], data.xquat[env.anchor + env.BODY0]
            fn = min(t + 2, ref.T - 1)
            g = np.array([0, 0, -1.0])
            if abs(H_.quat_rotate_inv(ref.aq[fn], g)[2] - H_.quat_rotate_inv(aquat, g)[2]) > 0.8:
                reason = "anchor_ori"
            elif abs(ref.ap[fn][2] - apos[2]) > 0.25:
                reason = "anchor_z"
            elif ee_check and any(abs(ref.eez[fn][k] - data.xpos[i + env.BODY0][2]) > 0.25 for k, i in enumerate(env.ee)):
                reason = "ee_z"
            if reason:
                break
        return dict(reason=reason, errs=errs, steps=len(errs))


def episode(A, B, variant, cfg, seed, stand):
    env = A.env
    model = copy.copy(env.base)
    R = Runner(cfg, seed, model, env)
    U = R.U
    data = mujoco.MjData(model)
    cut = float(os.environ.get('CUT', '1.0'))
    ref = A.ref if cut >= 1.0 else Ref(*[getattr(A.ref, k)[:int(cut * A.ref.T)] for k in ('jp', 'jv', 'ap', 'aq', 'eez')])
    p0 = env.bp[0, 0].copy(); q0 = env.bq[0, 0].copy(); lin0 = env.blv[0, 0].copy(); ang0 = env.bav[0, 0].copy(); jp0 = env.jp[0].copy()
    if cfg["reset_dr"]:
        p0 += [U(*S.POSE_RANGE["x"]), U(*S.POSE_RANGE["y"]), U(*S.POSE_RANGE["z"])]
        q0 = H.quat_mul(S.rpy_to_quat(U(*S.POSE_RANGE["roll"]), U(*S.POSE_RANGE["pitch"]), U(*S.POSE_RANGE["yaw"])), q0)
        lin0 += [U(*S.VEL_RANGE[k]) for k in "xyz"]
        ang0 += [U(*S.VEL_RANGE[k]) for k in ("roll", "pitch", "yaw")]
        jp0 = np.clip(jp0 + U(-0.1, 0.1, 24), env.jrange[:, 0], env.jrange[:, 1])
    data.qpos[0:3] = p0; data.qpos[3:7] = q0
    data.qvel[0:3] = lin0; data.qvel[3:6] = H.quat_rotate_inv(q0, ang0)
    for i in range(24):
        data.qpos[env.qadr[i]] = jp0[i]; data.qvel[env.dadr[i]] = env.jv[0, i]
    mujoco.mj_forward(model, data)

    out = dict(A_reason=None, B_reason=None, switch_err=None, B_mean_err=None, B_steps=0, trans_reason=None)
    SEG[0] = 'policy A: ' + A.name
    ra = R.run(data, A, ref)
    out["A_reason"] = ra["reason"]
    out["A_mean_err"] = float(np.mean(ra["errs"]))
    if ra["reason"] or variant == "alone":
        return out                                          # A itself fell; switch not tested
    mujoco.mj_forward(model, data)
    # robot's current pose -> re-anchor B at the robot's torso xy/yaw
    cap = data.xpos[env.anchor + env.BODY0].copy(); caq = data.xquat[env.anchor + env.BODY0].copy()
    cjp = data.qpos[env.qadr].copy()
    ceez = np.array([data.xpos[i + env.BODY0][2] for i in env.ee])
    Bref = reanchor(B.ref, cap[:2], yaw_of(caq))
    if variant == "direct":
        seq = Bref
    elif variant == "bridge":
        seq = cat([ramp(cjp, cap, caq, ceez, Bref.jp[0], Bref.ap[0], Bref.aq[0], Bref.eez[0], 50), Bref])
    elif variant == "stand":
        sap, saq, seez = stand
        yaw = yaw_of(caq)
        sap_w = rot_z(yaw, sap) + [cap[0] - 0.0, cap[1] - 0.0, 0]   # stand torso placed over the robot's current torso xy
        saq_w = H.quat_mul(yaw_quat(yaw), saq)
        r1 = ramp(cjp, cap, caq, ceez, H.DEFAULT_POS, sap_w, saq_w, seez, 50)
        r2 = hold((H.DEFAULT_POS, sap_w, saq_w, seez), 25)
        r3 = ramp(H.DEFAULT_POS, sap_w, saq_w, seez, Bref.jp[0], Bref.ap[0], Bref.aq[0], Bref.eez[0], 50)
        seq = cat([r1, r2, r3, Bref])
    nb = 0 if variant == "direct" else (50 if variant == "bridge" else 125)
    if nb:
        pre = Ref(seq.jp[:nb + 1], seq.jv[:nb + 1], seq.ap[:nb + 1], seq.aq[:nb + 1], seq.eez[:nb + 1])
        SEG[0] = 'transition (' + variant + ')'
        rt = R.run(data, B, pre, ee_check=True)
        out["trans_reason"] = rt["reason"]
        if rt["reason"]:
            out["B_reason"] = "transition:" + rt["reason"]
            return out
        body = Ref(seq.jp[nb:], seq.jv[nb:], seq.ap[nb:], seq.aq[nb:], seq.eez[nb:])
    else:
        body = seq
    out["switch_err"] = float(np.linalg.norm(data.qpos[env.qadr] - body.jp[0]))
    SEG[0] = 'policy B: ' + B.name
    rb = R.run(data, B, body)
    out["B_reason"] = rb["reason"]
    out["B_steps"] = rb["steps"]
    out["B_mean_err"] = float(np.mean(rb["errs"]))
    out["B_err_first1s"] = float(np.mean(rb["errs"][:50]))
    out["B_total"] = body.T - 1
    return out


if __name__ == "__main__":
    a, b, variant, scn, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    A, B = Pol(a), Pol(b)
    sap, saq, seez, vmax, ph = stand_pose(B.env)
    print(f"[stand] settled pelvis z={ph:.3f} torso z={sap[2]:.3f} max|qvel|={vmax:.3f}", flush=True)
    cfg = SCN[scn]
    res = [episode(A, B, variant, cfg, 1000 + s, (sap, saq, seez)) for s in range(n)]
    okA = [r for r in res if r["A_reason"] is None]
    okB = [r for r in okA if r["B_reason"] is None]
    why = {}
    for r in res:
        k = ("A:" + r["A_reason"]) if r["A_reason"] else r["B_reason"]
        if k:
            why[k] = why.get(k, 0) + 1
    se = np.mean([r["switch_err"] for r in okA if r["switch_err"] is not None] or [np.nan])
    me = np.mean([r["B_mean_err"] if variant != "alone" else r["A_mean_err"] for r in okB or okA] or [np.nan])
    e1 = np.mean([r["B_err_first1s"] for r in okB if "B_err_first1s" in r] or [np.nan])
    print(f"{a}->{b} {variant} {scn}: A finished {len(okA)}/{n}; switch+B survived {len(okB)}/{len(okA)} (of finished A); "
          f"switch jp err {se:.2f} rad; B mean err {me:.3f}; B first-1s err {e1:.2f}; failures {why}", flush=True)
    os.makedirs("switch_probe_out", exist_ok=True)
    json.dump(res, open(f"switch_probe_out/{a}_{b}_{variant}_{scn}_cut{os.environ.get('CUT','1.0')}.json", "w"), indent=1)
