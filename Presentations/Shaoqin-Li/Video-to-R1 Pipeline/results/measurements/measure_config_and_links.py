"""What is actually R1-specific about our GMR config, and are the numbers derived?

Two questions this answers, both from files rather than memory:

  1. WHAT DID WE CHANGE?  Diffs our bvh_xsens_to_r1.json against GMR's own
     bvh_xsens_to_g1.json -- the config it was adapted from.
  2. WHERE DO THE SCALE FACTORS COME FROM?  Measures the R1's real
     joint-to-joint link lengths and the capture skeleton's own segment
     offsets, and puts the implied ratio next to the number in the config.

    python measure_config_and_links.py            # needs the `gmr` env
"""
import argparse
import json
import os

import numpy as np

HOME = os.path.expanduser("~")
DEF_GMR = os.path.join(HOME, "GMR")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

# (label, R1 joint A, R1 joint B, BVH joint whose OFFSET is that segment,
#  the human_scale_table group the config puts it in)
SEGMENTS = [
    ("thigh",     "left_hip_pitch_joint",      "left_knee_joint",        "LeftKnee",  "legs"),
    ("shank",     "left_knee_joint",           "left_ankle_pitch_joint", "LeftAnkle", "legs"),
    ("upper arm", "left_shoulder_pitch_joint", "left_elbow_joint",       "LeftElbow", "arms"),
    ("forearm",   "left_elbow_joint",          "left_wrist_roll_joint",  "LeftWrist", "arms"),
]
GROUP_KEY = {"legs": "LeftKnee", "arms": "LeftElbow"}


def config_diff(gmr):
    d = os.path.join(gmr, "general_motion_retargeting", "ik_configs")
    r1 = json.load(open(os.path.join(d, "bvh_xsens_to_r1.json")))
    g1 = json.load(open(os.path.join(d, "bvh_xsens_to_g1.json")))
    print("=" * 68)
    print("  1. bvh_xsens_to_r1.json  vs  bvh_xsens_to_g1.json")
    print("=" * 68)
    for k in r1:
        if not isinstance(r1[k], dict) and r1[k] != g1.get(k):
            print("  scalar changed: %s  %s -> %s" % (k, g1.get(k), r1[k]))
    for tbl in ("ik_match_table1", "ik_match_table2"):
        a, b = r1[tbl], g1[tbl]
        only_r1 = [k for k in a if k not in b]
        only_g1 = [k for k in b if k not in a]
        changed = [k for k in a if k in b and a[k] != b[k]]
        print("\n  %s -- %d bodies each" % (tbl, len(a)))
        print("    bodies renamed for the R1 : %s" % (only_r1 or "none"))
        print("    bodies they replace       : %s" % (only_g1 or "none"))
        print("    shared bodies retuned     : %d" % len(changed))
    print("\n  human_scale_table:")
    for k, v in r1["human_scale_table"].items():
        w = g1["human_scale_table"].get(k)
        print("    %-16s G1 %-6s -> R1 %-6s%s" % (k, w, v, "" if v == w else "   changed"))
    return r1


def link_lengths(gmr, r1cfg, bvh):
    import mujoco
    xml = os.path.join(gmr, "assets", "unitree_r1", "r1_scene.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def anchor(j):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        return d.xanchor[i].copy() if i >= 0 else None

    # the capture's own skeleton: each joint's OFFSET is the vector from its
    # parent, i.e. the segment length of the performer who was recorded
    off, cur = {}, None
    for line in open(bvh):
        s = line.strip()
        if s.startswith(("ROOT", "JOINT")):
            cur = s.split()[1]
        elif s.startswith("OFFSET") and cur:
            off[cur] = np.linalg.norm([float(x) for x in s.split()[1:4]]) / 100.0  # cm -> m
            cur = None
        elif s.startswith("MOTION"):
            break

    print("\n" + "=" * 68)
    print("  2. where do the scale factors come from?")
    print("=" * 68)
    print("  %-10s %8s %10s %8s %10s" % ("segment", "R1", "performer", "ratio", "config"))
    print("  " + "-" * 50)
    for label, ja, jb, bvh_joint, group in SEGMENTS:
        pa, pb = anchor(ja), anchor(jb)
        if pa is None or pb is None or bvh_joint not in off:
            print("  %-10s  (joint not found)" % label)
            continue
        robot = float(np.linalg.norm(pa - pb))
        human = off[bvh_joint]
        cfg = r1cfg["human_scale_table"][GROUP_KEY[group]]
        print("  %-10s %7.3fm %9.3fm %8.3f %10.3f" % (label, robot, human, robot / human, cfg))
    print("""
  The ratio column is what the geometry implies; the config column is what we
  actually use.  They do not agree, so the scale factors were TUNED, not read
  off the robot.  Setting them from the ratios and re-measuring the motion is
  an open experiment -- see GMR_HANDOFF.md.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmr", default=DEF_GMR)
    ap.add_argument("--bvh", default=None)
    a = ap.parse_args()
    bvh = a.bvh or os.path.join(a.gmr, "assets", "xsens_bvh_test",
                                "251021_04_boxing_120Hz_cm_3DsMax.bvh")
    r1 = config_diff(a.gmr)
    link_lengths(a.gmr, r1, bvh)


if __name__ == "__main__":
    main()
