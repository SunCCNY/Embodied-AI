"""Gymnasium walking environment for Unitree R1 in plain MuJoCo.

Trains a velocity-tracking walking policy for the 12 leg joints. Arms and waist
are PD-held at the stand pose (matching the unitree_mujoco deployment), so the
policy only outputs 12 leg position-target offsets.

The observation is built ONLY from quantities the real DDS pipeline also
exposes (rt/lowstate): base angular velocity (gyro), projected gravity (from
the base quaternion), the velocity command, leg joint positions/velocities, and
the previous action. Base linear velocity is used for the reward but is NOT in
the observation (not reliably available on hardware) -- this keeps the obs
identical between training and sim2sim.

Control: policy at 50 Hz (decimation 10 over the 500 Hz physics), low-level PD
handled by the position actuators whose gains match the deployment torque PD.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
# default to the original model; set R1_WALK_XML=r1_walk_train_matched.xml to
# train on the deployment-physics-matched model (shrinks the sim2sim gap).
_xml_name = os.environ.get("R1_WALK_XML", "r1_walk_train.xml")
# Look beside this file first, then in a model/ subfolder. The student lab keeps
# its models in model/, the research tree keeps them alongside the code.
XML = next((c for c in (os.path.join(HERE, _xml_name),
                        os.path.join(HERE, "model", _xml_name),
                        _xml_name) if os.path.exists(c)),
           os.path.join(HERE, _xml_name))

# Leg actuator/joint order (indices 0..11), matches build_xml.py LEG_ACTS.
LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
# Default (stand) leg pose, same values as r1_stand_sim2sim.py STAND_POS legs.
DEFAULT_LEG = np.array([-0.1, 0, 0, 0.3, -0.2, 0,
                        -0.1, 0, 0, 0.3, -0.2, 0], dtype=np.float64)
# Arm/waist hold targets (actuator order 12..23, matches ARM_WAIST_ACTS).
ARM_WAIST_HOLD = np.array([0, 0, 0.35, 0.18, 0, 0.87, 0,
                           0.35, -0.18, 0, 0.87, 0], dtype=np.float64)

FOOT_GEOMS = [f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(1, 8)]

DECIMATION = 10          # 500 Hz physics -> 50 Hz control
ACTION_SCALE = 0.25
EP_LEN_S = 20.0

# obs scales
S_ANGVEL = 0.25
S_DOFVEL = 0.05

# ---- domain randomization ranges (per episode unless noted) ----
# The sim2sim gap is a DYNAMICS gap (training capsule contacts vs unitree_mujoco
# full-mesh + sphere-marker feet, plus motor/contact mismatch). These ranges
# build a robustness margin so the policy survives that gap instead of overfitting
# to one model. Applied only when dr=True.
DR = dict(
    friction=(0.5, 1.4),        # sliding friction, floor + foot geoms
    body_mass=(0.90, 1.12),     # per-body mass/inertia scale
    trunk_mass=(0.85, 1.25),    # extra scale on the pelvis (payload/battery slop)
    kp_scale=(0.85, 1.15),      # motor stiffness (models actuator mismatch)
    kv_scale=(0.85, 1.15),      # motor damping
    # Pushes address EXTERNAL-disturbance robustness, which is not the sim2sim
    # failure mode (nothing pushes the robot in the transfer test). Keep them mild
    # so episodes run long enough for the DYNAMICS randomizations above -- the ones
    # that actually close the training->unitree_mujoco gap -- to be learned.
    push_mag=0.3,               # max per-axis push velocity (m/s)
    push_period_s=(2.5, 4.0),   # push interval, randomized
)
# per-step observation noise std (raw units, added before obs scaling)
OBS_NOISE = dict(angvel=0.05, gravity=0.02, dofpos=0.01, dofvel=0.15)

# Control latency (in 50Hz control steps) randomized per episode. THIS is the
# real sim2sim gap: the DDS harness round-trip acts like ~1-3 steps of delay, and
# the zero-margin trained gait can't tolerate even 20-40ms. Training up to MAX_LATENCY
# forces a latency-robust (more grounded) gait. Verified: 2-step delay reproduces the
# ~2.7s unitree_mujoco fall on the exact deployment physics.
MAX_LATENCY = 4

# OPTION 3 (2026-07-11): walk4_lat trained against a FIXED per-episode latency
# (one uniform(0,4) draw held for the whole 20s episode) and got WORSE on the
# real harness, not better -- because measured DDS behavior is NOT a constant
# offset. measure_dds.py found: mean obs staleness 0.7ms (<<1 control step),
# with occasional publish-dt spikes to ~20ms (1 control step) from OS
# scheduling/GC, and the observed real fall matches a 2-step (40ms) spike.
# So the correct training signal is PER-STEP intermittent spikes on top of a
# near-zero baseline, not a per-episode constant delay. Opt-in via jitter_dr=True.
JITTER_DR = dict(
    spike_prob=0.04,        # P(a given control step is a "spike" step)
    spike_latency=(1, 3),   # spike magnitude range (control steps) when triggered
    rare_stall_prob=0.003,  # P(a rare larger GC-pause-like stall)
    rare_stall_latency=(4, 8),
)

# reward weights
# --- OPTION 2 additions (2026-07-11): the push baseline showed the gait
# already has good margin against EXTERNAL disturbance (88% full-survival
# under pushes) -- the actual gap is CONTROL-DELAY margin (60%/high-variance
# at just 2 control steps = 40ms, per r1_walk_latency_sweep.py). Classical
# control theory: delay margin trades off against closed-loop bandwidth,
# and bandwidth/reactivity is roughly what a lower CoM + wider support
# polygon buys mechanically (more passive stability = less reliant on fast
# active correction, which is exactly what stale/delayed commands break).
RW = dict(
    lin_vel=1.5, ang_vel=0.5, alive=0.15,
    lin_vel_z=-2.0, ang_vel_xy=-0.05, orientation=-5.0,
    torque=-1e-4, action_rate=-0.01, dof_acc=-2.5e-7,
    base_height=-10.0, feet_air_time=1.0, feet_slide=-0.1,
    heading=-1.5,    # penalize yaw deviation from the start heading (kills circle-walking)
    lateral=-1.0,    # penalize sideways (body-y) velocity (kills crabbing)
    stance_width=-4.0,  # OPTION 2: penalize lateral foot separation BELOW target (wider support polygon)
)
TARGET_H = 0.65          # OPTION 2: lowered from 0.70 -- more crouched = lower m*g*h
                          # toppling stiffness (see teaching notes Sec 2), mechanically
                          # more stable per unit disturbance/delay, at the cost of some
                          # torque efficiency (acceptable trade for delay-margin).
TARGET_STANCE_WIDTH = 0.22  # nominal lateral foot separation (m) to reward toward


def yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def projected_gravity(q):
    """Gravity vector [0,0,-1] expressed in the base frame = R(q)^T @ [0,0,-1].
    Equals minus the third row of R(q); written out to avoid building a matrix."""
    w, x, y, z = q
    return np.array([-2.0*(x*z - w*y),
                     -2.0*(y*z + w*x),
                     -(w*w - x*x - y*y + z*z)])


class R1WalkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cmd_range=(0.2, 0.6), push=True, dr=True, seed=None,
                 fixed_cmd=None, fixed_latency=None, jitter_dr=False,
                 zero_rew=None):
        super().__init__()
        # EXPERIMENT 3: reward-shaping ablation.  Names listed here are computed
        # as usual (so the UNMODIFIED total is still reported in info["full_rew"]
        # and can be used to score every ablation on one fixed yardstick) but are
        # dropped from the reward the policy actually optimizes.  Scoring an
        # ablated policy by its own ablated objective would be circular --
        # deleting a penalty raises the score by construction.
        self.zero_rew = set(zero_rew or [])
        unknown = self.zero_rew - set(RW)
        if unknown:
            raise ValueError(f"unknown reward term(s) {sorted(unknown)}; "
                             f"valid: {sorted(RW)}")
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.cmd_range = cmd_range
        self.push = push
        self.dr = dr
        self.fixed_cmd = None if fixed_cmd is None else np.asarray(fixed_cmd, float)
        # force an EXACT control-delay (in 50Hz control steps) every episode,
        # bypassing DR's randomized 0..MAX_LATENCY draw -- for measuring the
        # delay-margin curve (survival vs injected latency) precisely.
        self.fixed_latency = fixed_latency
        # OPTION 3: per-step intermittent spike model instead of a per-episode
        # constant latency draw -- see JITTER_DR comment above.
        self.jitter_dr = jitter_dr
        self.dt = self.model.opt.timestep * DECIMATION
        self.max_steps = int(EP_LEN_S / self.dt)

        m = self.model
        self.leg_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LEG_JOINTS])
        self.leg_vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LEG_JOINTS])
        # all 24 joint qpos/qvel adr for reset (leg + arm order == actuator order)
        all_joints = LEG_JOINTS + [
            "waist_roll_joint", "waist_yaw_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint",
        ]
        self.all_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in all_joints])
        self.default_all = np.concatenate([DEFAULT_LEG, ARM_WAIST_HOLD])

        self.gyro_adr = self._sensor_adr("imu_ang_vel")
        self.linvel_adr = self._sensor_adr("imu_lin_vel")
        self.floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # discover foot collision geoms from the model (robust to the capsule model
        # with 7/foot and the deployment-matched model with 1 box/foot)
        all_gnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        foot_names = [g for g in all_gnames if g and "foot" in g and "collision" in g]
        self.foot_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in foot_names]
        self.left_gids = {gid for gid, g in zip(self.foot_gids, foot_names) if g.startswith("left")}
        self.right_gids = {gid for gid, g in zip(self.foot_gids, foot_names) if g.startswith("right")}

        self.nominal_h = self._compute_nominal_height()

        # pristine copies of everything DR mutates, so each reset randomizes from
        # the nominal model rather than compounding previous episodes' scaling.
        self.orig_body_mass = m.body_mass.copy()
        self.orig_body_inertia = m.body_inertia.copy()
        self.orig_gainprm = m.actuator_gainprm.copy()
        self.orig_biasprm = m.actuator_biasprm.copy()
        self.orig_geom_friction = m.geom_friction.copy()
        self.pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_bid < 0:   # fall back to the base body (child of world/free joint)
            self.pelvis_bid = int(m.jnt_bodyid[0]) if m.njnt else 1
        # OPTION 2: ankle body ids for the stance-width reward
        self.left_ankle_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_ankle_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        n_obs = 3 + 3 + 3 + 12 + 12 + 12  # angvel, gravity, cmd, dofpos, dofvel, prev_action
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

        self.prev_action = np.zeros(12)
        self.command = np.zeros(3)
        self.air_time = np.zeros(2)
        self.prev_leg_vel = np.zeros(12)
        self.steps = 0
        self._push_every = int(2.0 / self.dt)

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return self.model.sensor_adr[sid]

    def _compute_nominal_height(self):
        """Set base z so the feet rest on the floor at the stand pose."""
        d = mujoco.MjData(self.model)
        d.qpos[self.all_qadr] = self.default_all
        d.qpos[2] = 1.0
        mujoco.mj_forward(self.model, d)
        min_fz = min(d.geom_xpos[g][2] for g in self.foot_gids)
        # geom center z; foot capsule radius 0.01 -> lowest point ~min_fz-0.01
        return 1.0 - min_fz + 0.01

    def _apply_dr(self):
        """Randomize model dynamics from the pristine nominal each reset."""
        m = self.model
        rng = self.rng
        # --- mass + inertia (per-body scale, extra slop on the pelvis) ---
        mass_scale = rng.uniform(*DR["body_mass"], size=m.nbody)
        mass_scale[self.pelvis_bid] *= rng.uniform(*DR["trunk_mass"])
        m.body_mass[:] = self.orig_body_mass * mass_scale
        m.body_inertia[:] = self.orig_body_inertia * mass_scale[:, None]
        # --- sliding friction: floor + both feet (independent) ---
        m.geom_friction[:] = self.orig_geom_friction
        m.geom_friction[self.floor_gid, 0] = rng.uniform(*DR["friction"])
        for g in self.foot_gids:
            m.geom_friction[g, 0] = rng.uniform(*DR["friction"])
        # --- PD gains (position actuator: gainprm[0]=kp, biasprm[1]=-kp, [2]=-kv) ---
        kp_s = rng.uniform(*DR["kp_scale"]) * rng.uniform(0.96, 1.04, size=m.nu)
        kv_s = rng.uniform(*DR["kv_scale"]) * rng.uniform(0.96, 1.04, size=m.nu)
        m.actuator_gainprm[:] = self.orig_gainprm
        m.actuator_biasprm[:] = self.orig_biasprm
        m.actuator_gainprm[:, 0] = self.orig_gainprm[:, 0] * kp_s
        m.actuator_biasprm[:, 1] = self.orig_biasprm[:, 1] * kp_s
        m.actuator_biasprm[:, 2] = self.orig_biasprm[:, 2] * kv_s

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.all_qadr] = self.default_all + self.rng.uniform(-0.03, 0.03, 24)
        d.qpos[2] = self.nominal_h + self.rng.uniform(-0.01, 0.01)
        # small random yaw
        yaw = self.rng.uniform(-0.1, 0.1)
        d.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        d.qvel[:] = 0
        self.yaw0 = yaw          # heading to hold while walking straight

        if self.dr:
            self._apply_dr()
            # random control latency 0..MAX_LATENCY steps (models the DDS round-trip;
            # this is the dominant sim2sim gap -- see MAX_LATENCY note)
            self.latency = int(self.rng.integers(0, MAX_LATENCY + 1))
            self._push_every = max(1, int(self.rng.uniform(*DR["push_period_s"]) / self.dt))
        else:
            self.latency = 0
        if self.fixed_latency is not None:
            self.latency = int(self.fixed_latency)   # override, for delay-margin measurement

        mujoco.mj_forward(self.model, d)
        self.prev_action[:] = 0
        self.prev_leg_vel[:] = d.qvel[self.leg_vadr]
        jitter_max = JITTER_DR["rare_stall_latency"][1] if self.jitter_dr else 0
        buf_len = max(MAX_LATENCY, self.latency, jitter_max) + 1
        self.act_buf = [np.zeros(12) for _ in range(buf_len)]  # control-latency delay
        self.air_time[:] = 0
        self.steps = 0
        # sample command (forward-biased) unless a fixed command was set
        if self.fixed_cmd is not None:
            self.command = self.fixed_cmd.copy()
        else:
            vx = self.rng.uniform(*self.cmd_range)
            self.command = np.array([vx, 0.0, 0.0])
        return self._obs(), {}

    def _obs(self):
        d = self.data
        angvel = d.sensordata[self.gyro_adr:self.gyro_adr+3].copy()
        grav = projected_gravity(d.qpos[3:7])
        dofpos = d.qpos[self.leg_qadr] - DEFAULT_LEG
        dofvel = d.qvel[self.leg_vadr].copy()
        if self.dr:
            # sensor noise -> robustness to the deployment's noisier obs
            n = OBS_NOISE
            angvel += self.rng.normal(0, n["angvel"], 3)
            grav = grav + self.rng.normal(0, n["gravity"], 3)
            dofpos = dofpos + self.rng.normal(0, n["dofpos"], 12)
            dofvel = dofvel + self.rng.normal(0, n["dofvel"], 12)
        obs = np.concatenate([
            angvel * S_ANGVEL,
            grav,
            self.command,
            dofpos,
            dofvel * S_DOFVEL,
            self.prev_action,
        ]).astype(np.float32)
        return obs

    def _foot_contacts(self):
        d = self.data
        lc = rc = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            pair = {g1, g2}
            if self.floor_gid in pair:
                other = g1 if g2 == self.floor_gid else g2
                if other in self.left_gids:
                    lc = True
                elif other in self.right_gids:
                    rc = True
        return np.array([lc, rc])

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        # control latency: apply the action from `latency` control-steps ago
        self.act_buf.append(action)
        jitter_max = JITTER_DR["rare_stall_latency"][1] if self.jitter_dr else 0
        buf_len = max(MAX_LATENCY, self.latency, jitter_max) + 1
        if len(self.act_buf) > buf_len:
            self.act_buf.pop(0)
        # OPTION 3: per-step intermittent spike, drawn fresh each control step
        # (not held constant for the episode like the original DR/fixed_latency).
        step_latency = self.latency
        if self.jitter_dr:
            u = self.rng.random()
            if u < JITTER_DR["rare_stall_prob"]:
                step_latency = int(self.rng.integers(*JITTER_DR["rare_stall_latency"]))
            elif u < JITTER_DR["rare_stall_prob"] + JITTER_DR["spike_prob"]:
                step_latency = int(self.rng.integers(*JITTER_DR["spike_latency"]))
            else:
                step_latency = 0
        applied = self.act_buf[-(step_latency + 1)]
        leg_target = DEFAULT_LEG + ACTION_SCALE * applied
        ctrl = np.concatenate([leg_target, ARM_WAIST_HOLD])
        self.data.ctrl[:] = ctrl

        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        d = self.data

        # random push (bigger + randomized interval when DR is on)
        if self.push and self.steps % self._push_every == 0:
            pm = DR["push_mag"] if self.dr else 0.3
            d.qvel[0:2] += self.rng.uniform(-pm, pm, 2)

        # --- reward terms ---
        linvel = d.sensordata[self.linvel_adr:self.linvel_adr+3]     # base frame
        angvel = d.sensordata[self.gyro_adr:self.gyro_adr+3]
        grav = projected_gravity(d.qpos[3:7])
        base_z = d.qpos[2]
        leg_vel = d.qvel[self.leg_vadr]
        leg_force = d.actuator_force[:12]

        r = {}
        dvx = linvel[0] - self.command[0]; dvy = linvel[1] - self.command[1]
        r["lin_vel"] = RW["lin_vel"] * np.exp(-(dvx*dvx + dvy*dvy) / 0.25)
        wz_err = (angvel[2] - self.command[2])**2
        r["ang_vel"] = RW["ang_vel"] * np.exp(-wz_err / 0.25)
        r["alive"] = RW["alive"]
        r["lin_vel_z"] = RW["lin_vel_z"] * linvel[2]**2
        r["ang_vel_xy"] = RW["ang_vel_xy"] * (angvel[0]**2 + angvel[1]**2)
        r["orientation"] = RW["orientation"] * (grav[0]**2 + grav[1]**2)
        r["torque"] = RW["torque"] * float(leg_force @ leg_force)
        da = action - self.prev_action
        r["action_rate"] = RW["action_rate"] * float(da @ da)
        acc = (leg_vel - self.prev_leg_vel) / self.dt
        r["dof_acc"] = RW["dof_acc"] * float(acc @ acc)
        r["base_height"] = RW["base_height"] * (base_z - self.nominal_h)**2
        # keep a straight heading + no crabbing (fixes the reward-hacked curved gait)
        yaw_err = wrap_pi(yaw_from_quat(d.qpos[3:7]) - self.yaw0)
        r["heading"] = RW["heading"] * yaw_err * yaw_err
        r["lateral"] = RW["lateral"] * linvel[1] * linvel[1]

        # OPTION 2: reward lateral foot separation toward TARGET_STANCE_WIDTH (wider
        # base of support = more margin before the CoM projection exits the support
        # polygon under a stale/delayed correction). Only reward reaching the target
        # or beyond -- no bonus (or penalty) for going wider than target.
        stance_w = abs(d.xpos[self.left_ankle_bid][1] - d.xpos[self.right_ankle_bid][1])
        shortfall = max(0.0, TARGET_STANCE_WIDTH - stance_w)
        r["stance_width"] = RW["stance_width"] * shortfall ** 2

        # feet air time (encourages stepping, not shuffling)
        contacts = self._foot_contacts()
        first_contact = contacts & (self.air_time > 0)
        self.air_time += self.dt
        air_reward = np.sum((self.air_time - 0.4) * first_contact)
        self.air_time[contacts] = 0.0
        # only reward stepping when actually commanded to move
        cmd_mag = np.linalg.norm(self.command[:2])
        r["feet_air_time"] = RW["feet_air_time"] * air_reward * (cmd_mag > 0.1)

        # EXPERIMENT 3: full_reward is the unmodified objective -- kept for
        # scoring; `reward` is what this policy is actually trained on.
        full_reward = sum(r.values())
        for k in self.zero_rew:
            r[k] = 0.0
        reward = sum(r.values())

        self.prev_action = action.copy()
        self.prev_leg_vel = leg_vel.copy()

        # termination
        fell = base_z < 0.4
        tilted = grav[2] > -0.5
        bad = not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
        terminated = bool(fell or tilted or bad)
        truncated = self.steps >= self.max_steps
        if terminated:
            reward -= 5.0
            full_reward -= 5.0

        tilt_deg = float(np.degrees(np.arccos(np.clip(-grav[2], -1.0, 1.0))))
        info = {"rew": r, "full_rew": full_reward, "cmd": self.command.copy(),
                "fwd_vel": float(linvel[0]), "base_z": float(base_z),
                "base_xy": (float(d.qpos[0]), float(d.qpos[1])),
                "tilt_deg": tilt_deg, "max_leg_torque": float(np.max(np.abs(leg_force)))}
        obs = self._obs() if not bad else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, truncated, info


if __name__ == "__main__":
    # quick self-test
    env = R1WalkEnv(seed=0)
    print("nominal_h", round(env.nominal_h, 4), "max_steps", env.max_steps, "dt", env.dt)
    o, _ = env.reset()
    print("obs dim", o.shape, "act dim", env.action_space.shape)
    tot = 0.0
    for i in range(200):
        o, rew, term, trunc, info = env.step(env.action_space.sample() * 0.0)  # hold stand
        tot += rew
        if term or trunc:
            print("episode end at", i, "term", term, "trunc", trunc)
            break
    print("held-stand 200-step return:", round(tot, 2), "final base_z", round(info["base_z"], 3))
