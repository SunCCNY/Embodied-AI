"""Experiment 4 -- is this motion physically possible?

The retargeted motion LOOKS right.  This asks whether a real robot could have
produced it.  Run it on your own output:

    python exp4_check_motion.py reference/r1_boxing.pkl
    python exp4_check_motion.py out/my_motion.pkl --robot unitree_r1

Needs the `gmr` conda environment (it reads the robot model out of GMR).

WHAT IT CHECKS
  1. FEET     do they ever touch the ground
  2. PELVIS   does the body height move at all
  3. LIMITS   is any joint driven past its mechanical stop
  4. SPEED    does any joint move faster than a motor could

A motion can pass every visual inspection and fail all four.  That is the
point of the experiment.
"""
import argparse, os, pickle, sys
import numpy as np
import mujoco

try:
    from general_motion_retargeting import ROBOT_XML_DICT
except ImportError:
    sys.exit("Run this inside the `gmr` environment:  conda activate gmr")

# ------------------------------------------------------------------ EDIT ZONE
FLOAT_TOL_CM = 1.0      # closer than this to the floor counts as "touching"
STILL_TOL_CM = 0.5      # pelvis range below this counts as "frozen"
MAX_JOINT_SPEED = 20.0  # rad/s -- generous for a real servo
# ---------------------------------------------------------------------------


def load(path):
    d = pickle.load(open(path, "rb"))
    need = ("root_pos", "root_rot", "dof_pos")
    if not all(k in d for k in need):
        sys.exit(f"{path} is missing {need}; is it a GMR motion pickle?")
    return (np.asarray(d["root_pos"]), np.asarray(d["root_rot"]),
            np.asarray(d["dof_pos"]), float(d.get("fps", 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--robot", default="unitree_r1")
    a = ap.parse_args()

    rp, rr, dp, fps = load(a.motion)
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[a.robot]))
    d = mujoco.MjData(m)
    n = len(rp)

    print(f"\n{'='*64}\n  {os.path.basename(a.motion)}  ->  {a.robot}")
    print(f"  {n} frames @ {fps:g} fps = {n/fps:.1f} s   |   {dp.shape[1]} joints")
    print("=" * 64)

    # the spheres and capsules that would actually hit the floor -- selected on
    # physics, not naming: a geom counts if it takes part in contact and belongs
    # to a real body.  (The R1 names its collision geoms `*_collision`; the G1
    # names none of them, and the old name test fell back to ALL geoms including
    # the floor plane, which reported the G1 as touching down on every frame.)
    foot = [g for g in range(m.ngeom)
            if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0]

    low = np.empty(n)
    for i in range(n):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = rp[i], rr[i], dp[i]
        mujoco.mj_forward(m, d)
        # bottom of each contact primitive, not its centre
        low[i] = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in foot)

    # ------------------------------------------------------- 1. FEET
    print("\n  1. FEET -- does the robot ever touch the ground?")
    print(f"       lowest contact point:  min {low.min()*100:+6.2f} cm"
          f"   mean {low.mean()*100:+6.2f} cm   max {low.max()*100:+6.2f} cm")
    touch = int((np.abs(low) < FLOAT_TOL_CM / 100).sum())
    if touch == 0:
        state = "FLOATING the whole time" if low.min() > 0 else "SUNK THROUGH THE FLOOR"
        print(f"       -> {state}.  0 of {n} frames make contact.")
    else:
        print(f"       -> touches down on {touch}/{n} frames ({100*touch/n:.0f}%)")

    # ------------------------------------------------------- 2. PELVIS
    rng = (rp[:, 2].max() - rp[:, 2].min()) * 100
    print("\n  2. PELVIS -- does the body rise and fall like a real one?")
    print(f"       height {rp[:,2].min():.3f} .. {rp[:,2].max():.3f} m   (range {rng:.2f} cm)")
    verdict = ('FROZEN. A moving human does not hold this constant.'
               if rng < STILL_TOL_CM else 'moves, as a real body would')
    print(f"       -> {verdict}")

    # ------------------------------------------------------- 3. LIMITS
    jnt = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    lim = m.jnt_range[jnt]
    lim = lim[:dp.shape[1]]
    lo, hi = lim[:, 0], lim[:, 1]
    over = (dp < lo - 1e-6) | (dp > hi + 1e-6)
    print("\n  3. LIMITS -- is any joint pushed past its mechanical stop?")
    if over.any():
        worst = np.argsort(-over.sum(0))[:3]
        print(f"       {over.any(1).sum()}/{n} frames violate a limit")
        for j in worst:
            if over[:, j].sum():
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jnt[j]) or f"joint{j}"
                ex = max(abs(dp[:, j].min() - lo[j]), abs(dp[:, j].max() - hi[j]))
                print(f"         {nm:28s} {over[:,j].sum():5d} frames, worst by {ex:.3f} rad")
    else:
        print(f"       -> none. every joint stays inside its range.")

    # ------------------------------------------------------- 4. SPEED
    vel = np.diff(dp, axis=0) * fps
    peak = np.abs(vel).max()
    print("\n  4. SPEED -- could a motor actually move this fast?")
    print(f"       peak joint speed {peak:.1f} rad/s   (limit set to {MAX_JOINT_SPEED:g})")
    fast = int((np.abs(vel) > MAX_JOINT_SPEED).any(1).sum())
    print(f"       -> {'exceeded on ' + str(fast) + ' frames' if fast else 'within a plausible servo budget'}")

    # --------------------------------------------- IS THIS EVEN THE MOTION?
    # Xsens files open with a stationary calibration T-pose.  Measure that by
    # accident and you will "discover" a frozen pelvis and floating feet that
    # belong to the calibration, not to the capture.  This cost the author of
    # this script an hour, so the check is built in.
    static = rng < STILL_TOL_CM and peak < 2.0
    if static:
        print("\n  !! WARNING -- THIS LOOKS LIKE A CALIBRATION POSE, NOT A MOTION.")
        print(f"     The pelvis moves {rng:.2f} cm and the fastest joint reaches")
        print(f"     {peak:.1f} rad/s.  Real motion does not look like this.")
        print("     Xsens captures begin with a stationary T-pose.  Skip past it:")
        print("        --start 600 --end 1800")
        print("     Every number above describes the T-pose.  Re-run before")
        print("     drawing any conclusion from them.")

    # ------------------------------------------------------- VERDICT
    fails = [t for t, bad in (
        (f"feet penetrate the floor by {abs(low.min())*100:.2f} cm", low.min() < -FLOAT_TOL_CM/100),
        ("feet never contact the ground", touch == 0 and low.min() > 0),
        ("pelvis height is frozen", rng < STILL_TOL_CM),
        ("joint limits violated", bool(over.any())),
        (f"joint speed implausible on {fast} frames", fast > 0)) if bad]
    print("\n" + "=" * 64)
    if static:
        print("  VERDICT: withheld -- you measured a calibration pose.")
        print("  Re-run with --start 600 --end 1800 and read the result then.")
        print("=" * 64 + "\n"); return
    if fails:
        print("  VERDICT: this motion is NOT physically realisable.")
        for f in fails:
            print(f"     - {f}")
        print("""
  This is the lesson of Experiment 4.  Retargeting solves ONE problem: put the
  robot's hands and feet where the human's were.  It succeeds at that, which is
  why the video looks convincing.  Nothing in the pipeline ever asked about
  gravity, contact, torque, or whether the robot could stay upright -- so
  nothing in the output respects them.

  A retargeted motion is a REFERENCE, not a controller.  Making it real is the
  job of the policy you trained in Lab 2: track this trajectory while actually
  balancing.  That two-stage design -- retarget, then RL -- is the architecture
  behind every paper in the reading packet.""")
    else:
        print("  VERDICT: nothing implausible found.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
