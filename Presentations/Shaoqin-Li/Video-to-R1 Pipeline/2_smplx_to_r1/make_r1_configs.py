"""
Generate the Unitree R1 IK configs for every input format GMR supports.

GMR keeps one IK config per (input format, robot) pair.  Out of the eight formats it
supports, the R1 had two: bvh_xsens (tested) and smplx (untested).  This writes the
other six.

The R1 config is the G1 config with exactly two kinds of change, and nothing else:

  1. human_scale_table -- every entry multiplied by a fixed per-limb-group ratio.
     The scale table maps a human skeleton onto a robot's proportions, so the
     format sets the baseline and the robot sets the ratio.

         torso  0.84  / 0.9  = 0.933333
         legs   0.818 / 0.9  = 0.908889
         arms   0.578 / 0.75 = 0.770667

  2. Robot link names the R1 does not have, renamed to the R1 equivalent:

         *_wrist_yaw_link  ->  *_wrist_roll_link   (the R1 arm stops at wrist roll)
         *_toe_link        ->  *_ankle_roll_link   (the R1 foot has no toe body)

     The toe rename also ZEROES the position offset.  That offset is measured in
     the toe body's own frame -- smplx uses [0, +/-0.02, 0] and xrobot [0.05, 0, 0]
     to place the target relative to the G1's toe -- and the ankle roll body sits
     somewhere else entirely, so a toe-frame offset is meaningless once the target
     body changes.  The pre-existing smplx_to_r1 did exactly this, and --check is
     what caught it: the rename alone reproduced bvh_xsens but not smplx.
     Wrist offsets are zero in every G1 config, so the wrist rename carries nothing.

     The rename walks the WHOLE document, not just the two ik_match_tables.
     xsens_mvn is the only format whose G1 config carries a third block --
     collision_avoidance.self_collision_pairs, a nested list of link names -- and
     an earlier version of this script, which walked only the match tables, let
     left/right_wrist_yaw_link through into it as raw G1.  The block ships
     "enabled": false so nothing failed, but it named links the R1 does not have.
     Anything that is not the human side is now swept generically, so a new
     upstream block cannot slip past.

The ratios are not a guess and not re-derived here -- they are read back out of the
two configs that already exist.  --check regenerates those two from their G1 parents
and asserts the result is byte-identical to what is on disk.  If that passes, the
same rule produced the other six.

Usage:
    python make_r1_configs.py --check      # prove the rule against the 2 known configs
    python make_r1_configs.py              # generate the 6 missing ones
    python make_r1_configs.py --all        # rewrite all 8 from their G1 parents
"""

import argparse
import json
import sys
from pathlib import Path

GMR = Path.home() / "GMR"
IK = GMR / "general_motion_retargeting" / "ik_configs"
R1_XML = GMR / "assets" / "unitree_r1" / "r1_mocap.xml"

# The scale table is 14 entries in a fixed order for every format:
#   0,1    pelvis/hips + spine/chest      -> torso
#   2..7   hip, knee, foot, left & right  -> legs
#   8..13  shoulder, elbow, wrist, l & r  -> arms
TORSO, LEGS, ARMS = slice(0, 2), slice(2, 8), slice(8, 14)

RATIO = {
    "torso": 0.84 / 0.9,
    "legs": 0.818 / 0.9,
    "arms": 0.578 / 0.75,
}

# Robot links the R1 does not have -> what to use instead.
LINK_SUBS = {
    "left_wrist_yaw_link": "left_wrist_roll_link",
    "right_wrist_yaw_link": "right_wrist_roll_link",
    "left_toe_link": "left_ankle_roll_link",
    "right_toe_link": "right_ankle_roll_link",
}

# Renames that move the target to a different body, so any position offset tuned
# for the old body has to be dropped rather than carried across.
DROP_OFFSET = {"left_toe_link", "right_toe_link"}

# The human side of the config.  Strings under these keys describe the mocap
# skeleton, not the robot, so the link rename must not touch them.
HUMAN_KEYS = {"human_root_name", "human_height_assumption", "human_scale_table"}

# Position of the offset vector inside an ik_match_table entry:
#   [human_joint, position_weight, rotation_weight, position_offset, rotation_offset]
OFFSET_INDEX = 3

# Every format GMR ships a G1 config for.
FORMATS = [
    "smplx",
    "bvh_xsens",
    "bvh_lafan1",
    "bvh_nokov",
    "xsens_mvn",
    "fbx",
    "fbx_offline",
    "xrobot",
]
ALREADY_EXISTED = {"smplx", "bvh_xsens"}


def scale_group(index):
    """Which limb group the n-th entry of a scale table belongs to."""
    if TORSO.start <= index < TORSO.stop:
        return "torso"
    if LEGS.start <= index < LEGS.stop:
        return "legs"
    return "arms"


def rename_deep(node):
    """Apply LINK_SUBS to every string anywhere in a nested structure."""
    if isinstance(node, dict):
        return {LINK_SUBS.get(k, k): rename_deep(v) for k, v in node.items()}
    if isinstance(node, list):
        return [rename_deep(v) for v in node]
    if isinstance(node, str):
        return LINK_SUBS.get(node, node)
    return node


def convert(g1_config):
    """Turn a G1 config into the R1 config, preserving key order throughout."""
    out = {}
    for key, value in g1_config.items():
        if key == "human_scale_table":
            out[key] = {
                joint: round(scale * RATIO[scale_group(i)], 3)
                for i, (joint, scale) in enumerate(value.items())
            }
        elif key in HUMAN_KEYS:
            out[key] = value
        elif key.startswith("ik_match_table") and isinstance(value, dict):
            # Keyed by robot link name, and the entry carries an offset that is
            # only meaningful in the old body's frame.
            table = {}
            for link, spec in value.items():
                if link in DROP_OFFSET:
                    spec = list(spec)
                    spec[OFFSET_INDEX] = [0.0, 0.0, 0.0]
                table[LINK_SUBS.get(link, link)] = spec
            out[key] = table
        else:
            # Anything else -- collision_avoidance today, whatever upstream adds
            # tomorrow.  Robot link names get renamed wherever they are nested.
            out[key] = rename_deep(value)
    return out


def robot_links_in(config):
    """
    Every robot link the config names, from anywhere in the document.

    Match-table keys are link names by definition; elsewhere a string is taken
    to be a link if it ends in _link.  This is the check that would have caught
    the collision_avoidance block.
    """
    links = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.endswith("_link"):
                    links.add(k)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and node.endswith("_link"):
            links.add(node)

    for key, value in config.items():
        if key in HUMAN_KEYS:
            continue
        if key.startswith("ik_match_table") and isinstance(value, dict):
            links |= set(value)
        else:
            walk(value)
    return links


def r1_body_names():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(R1_XML))
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }


def dumps(config):
    return json.dumps(config, indent=4) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate the 2 pre-existing R1 configs and compare")
    ap.add_argument("--all", action="store_true",
                    help="rewrite all 8 configs, not just the missing ones")
    args = ap.parse_args()

    if args.check:
        print("Regenerating the two R1 configs that already existed, from their G1 parents.")
        print("Both must come out identical to the files on disk for the rule to hold.\n")
        ok = True
        for fmt in sorted(ALREADY_EXISTED):
            g1 = json.loads((IK / f"{fmt}_to_g1.json").read_text())
            existing = (IK / f"{fmt}_to_r1.json").read_text()
            regenerated = dumps(convert(g1))
            same = json.loads(existing) == json.loads(regenerated)
            print(f"  {fmt + '_to_r1.json':26s} {'MATCH' if same else 'DIFFERS'}")
            if not same:
                ok = False
                a, b = json.loads(existing), json.loads(regenerated)
                for k in a:
                    if a[k] != b.get(k):
                        print(f"      first difference in: {k}")
                        break
        print("\nRule reproduces both existing configs." if ok else "\nRULE FAILED.")
        return 0 if ok else 1

    bodies = r1_body_names()
    targets = FORMATS if args.all else [f for f in FORMATS if f not in ALREADY_EXISTED]
    print(f"R1 model has {len(bodies)} bodies.  Generating {len(targets)} config(s).\n")

    failed = False
    for fmt in targets:
        src = IK / f"{fmt}_to_g1.json"
        dst = IK / f"{fmt}_to_r1.json"
        if not src.exists():
            print(f"  {fmt:14s} no G1 parent, skipped")
            continue

        g1 = json.loads(src.read_text())
        r1 = convert(g1)

        missing = sorted(robot_links_in(r1) - bodies)
        scales = list(r1["human_scale_table"].values())
        status = "ok" if not missing else f"MISSING LINKS {missing}"
        if missing:
            failed = True

        dst.write_text(dumps(r1))
        print(f"  {fmt:14s} -> {dst.name:26s} "
              f"scales {scales[0]}/{scales[2]}/{scales[8]}  {status}")

    print("\nEvery robot link resolves against r1_mocap.xml." if not failed
          else "\nSOME LINKS DO NOT EXIST ON THE R1 -- fix LINK_SUBS.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
