"""Watch a TRAINED policy drive the R1 — Labs 2 and 3.

Needs only mujoco + numpy. The policy is a plain .npz (a 45->256->256->12 MLP
plus its observation statistics), exported from the trained network, so there is
no PyTorch and no stable-baselines3 to install.

    python watch_policy.py                                  # Lab 2: it stands
    python watch_policy.py --push 0.30                      # Lab 2: survives
    python watch_policy.py --push 0.40                      # Lab 2: falls
    python watch_policy.py --noise gravity 10               # Lab 3: blind it
    python watch_policy.py --zero command                   # Lab 3: the null control
    python watch_policy.py --latency 8                      # Lab 3: 160 ms delay
    python watch_policy.py --policy walk3b.npz --vx 0.5     # a walking policy
    python watch_policy.py --headless                       # verdict only, no window

KEYS: left-drag orbit · scroll zoom · space pause · Esc quit
"""
import argparse, os, sys, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

LEG = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
       "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
       "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll"]
DEFAULT_LEG = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0])
ARM_HOLD = np.array([0, 0, 0.35, 0.18, 0, 0.87, 0, 0.35, -0.18, 0, 0.87, 0])
ACTION_SCALE, DECIMATION = 0.25, 10
S_ANGVEL, S_DOFVEL = 0.25, 0.05
# observation slices and the noise each channel saw in training
# channel -> (where it sits in the 45-dim observation, the noise it saw in
# TRAINING, the scale already applied to it).  --noise doses in multiples of that
# training sigma, so "x10" means ten times what the policy was trained to expect.
#
# sigma=None means the channel had no injected training noise, so there is no
# natural unit.  For those we dose in multiples of the channel's own training
# standard deviation, read from the policy's obs_var.  Do not invent a fixed
# number here: the command channel's training sd is 5.9e-06, so ANY absolute
# noise is amplified ~170,000x by normalisation and saturates the clip bound.
# That is why --noise command is not a null control.  Use --zero for that.
CH = {"angvel": (slice(0, 3), 0.05, S_ANGVEL), "gravity": (slice(3, 6), 0.02, 1.0),
      "dofpos": (slice(9, 21), 0.01, 1.0), "dofvel": (slice(21, 33), 0.15, S_DOFVEL),
      "command": (slice(6, 9), None, 1.0), "prev_action": (slice(33, 45), None, 1.0)}


def load_policy(path):
    d = np.load(path)
    def act(obs):
        o = np.clip((obs - d["obs_mean"]) / np.sqrt(d["obs_var"] + d["epsilon"]),
                    -d["clip_obs"], d["clip_obs"])
        h = np.tanh(d["w0"] @ o + d["b0"])
        h = np.tanh(d["w1"] @ h + d["b1"])
        return np.clip(d["w2"] @ h + d["b2"], d["act_low"], d["act_high"])
    return act


def projected_gravity(q):
    w, x, y, z = q
    return np.array([-2*(x*z - w*y), -2*(y*z + w*x), -(w*w - x*x - y*y + z*z)])


class Robot:
    def __init__(self, xml, vx):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.m.opt.timestep = 0.002
        self.d = mujoco.MjData(self.m)
        self.m.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
        self.m.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        name2id = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")
        self.qadr = np.array([self.m.jnt_qposadr[name2id(j)] for j in LEG])
        self.vadr = np.array([self.m.jnt_dofadr[name2id(j)] for j in LEG])
        self.gyro = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
        self.cmd = np.array([vx, 0.0, 0.0])
        self.dt = self.m.opt.timestep * DECIMATION
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        allq = np.concatenate([DEFAULT_LEG, ARM_HOLD])
        adr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
               for n in [j + "_joint" for j in LEG] +
               ["waist_roll_joint", "waist_yaw_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint"]]
        self.d.qpos[adr] = allq
        mujoco.mj_forward(self.m, self.d)
        self.prev_action = np.zeros(12)
        self.buf = []

    def obs(self, sl=None, sigma=None, mult=0.0, rng=None, zero_sl=None, zero_val=None):
        d = self.d
        angvel = d.sensordata[self.gyro:self.gyro+3].copy()
        grav = projected_gravity(d.qpos[3:7])
        dofpos = d.qpos[self.qadr] - DEFAULT_LEG
        dofvel = d.qvel[self.vadr].copy()
        o = np.concatenate([angvel*S_ANGVEL, grav, self.cmd, dofpos, dofvel*S_DOFVEL, self.prev_action])
        if sl is not None:
            o[sl] += rng.standard_normal(sl.stop - sl.start) * sigma * mult
        if zero_sl is not None:
            o[zero_sl] = zero_val          # pin a channel to its TRAINING value
        return o

    def step(self, action):
        self.prev_action = action.copy()
        target = DEFAULT_LEG + action * ACTION_SCALE
        self.d.ctrl[:] = np.concatenate([target, ARM_HOLD])
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)

    def tilt(self):
        z = self.d.xmat[1].reshape(3, 3)[:, 2]
        return np.degrees(np.arccos(np.clip(z[2], -1, 1)))


def run(a):
    xml = a.xml or os.path.join(HERE, "model", "r1_walk_train.xml")
    pol_path = a.policy if os.path.isabs(a.policy) else os.path.join(HERE, "policies", a.policy)
    act = load_policy(pol_path)
    stats = np.load(pol_path)
    r = Robot(xml, a.vx)
    rng = np.random.default_rng(a.seed)

    sl = sigma = mult = None
    if a.noise:
        sl, sig, scale = CH[a.noise[0]]
        mult = float(a.noise[1])
        # a channel with injected training noise is dosed in multiples of it;
        # one without is dosed in multiples of its own training spread
        sigma = (np.full(sl.stop - sl.start, sig * scale) if sig is not None
                 else np.sqrt(stats["obs_var"][sl]))
    zero_sl = CH[a.zero][0] if a.zero else None
    zero_val = stats["obs_mean"][zero_sl] if a.zero else None

    label = [os.path.basename(pol_path)]
    if a.vx:      label.append(f"vx {a.vx}")
    if a.xml:     label.append(f"model {os.path.basename(xml)}")
    if a.push:    label.append(f"push {a.push}")
    if a.latency: label.append(f"latency {a.latency*20} ms")
    if a.noise:   label.append(f"{a.noise[0]} noise x{a.noise[1]}")
    if a.zero:    label.append(f"{a.zero} zeroed")
    print(" | ".join(label))

    def one_frame(s):
        o = r.obs(sl, sigma, mult, rng, zero_sl, zero_val)
        r.buf.append(act(o))
        used = r.buf[max(0, len(r.buf) - 1 - a.latency)]     # stale action = control delay
        if a.push and s == int(1.0 / r.dt):
            r.d.qvel[0] += a.push
        r.step(used)

    total, peak = int(a.seconds / r.dt), 0.0
    if a.headless:
        for s in range(total):
            one_frame(s)
            peak = max(peak, r.tilt())          # the PEAK, not the tilt it happens
            if r.tilt() > 45:                   # to be sitting at when time runs out
                print(f"  fell at {s*r.dt:.2f}s"); return
        print(f"  STOOD the full {a.seconds:g}s  (max tilt {peak:.2f} deg)"); return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(r.m, r.d) as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 135, -8
        s, done = 0, None
        while v.is_running():
            t0 = time.time()
            if s < total and done is None:
                one_frame(s); s += 1
                peak = max(peak, r.tilt())
                if r.tilt() > 45:
                    done = f"fell at {s*r.dt:.2f}s"; print("  " + done)
                elif s >= total:
                    done = f"STOOD the full {a.seconds:g}s  (max tilt {peak:.2f} deg)"
                    print("  " + done)
            v.cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.45]
            v.sync()
            lag = r.dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="exp2_stand.npz")
    p.add_argument("--xml", default=None)
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--latency", type=int, default=0, help="control delay in 20 ms steps")
    p.add_argument("--noise", nargs=2, metavar=("CHANNEL", "MULT"),
                   help="corrupt one channel: angvel|gravity|dofpos|dofvel|command|prev_action")
    p.add_argument("--zero", metavar="CHANNEL",
                   help="pin a channel to its training value -- the NULL CONTROL")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true")
    run(p.parse_args())
