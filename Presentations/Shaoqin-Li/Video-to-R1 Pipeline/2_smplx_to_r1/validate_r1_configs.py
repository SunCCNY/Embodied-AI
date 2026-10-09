"""
Validate every Unitree R1 IK config against its G1 parent and the R1 model.

Week 26 generated six new *_to_r1.json configs and asserted "every robot link
resolves against r1_mocap.xml".  That claim was true only of the two
ik_match_tables: the check walked those and nothing else.  xsens_mvn is the one
format whose G1 config carries a third block -- collision_avoidance -- and its
self_collision_pairs came through as raw G1, referencing left/right_wrist_yaw_link,
links the R1 does not have.

So this validator does not trust the generator's own notion of where links live.
It sweeps the WHOLE document for anything that names a robot body, using the G1
model as the authority on what a robot body name looks like, and reports the JSON
path of every reference it finds.  A new upstream block cannot slip past it.

Five checks per format, all against the G1 parent:

  A  link resolution   every robot-link reference ANYWHERE in the file exists in
                       r1_mocap.xml
  B  human side        scale-table keys and every ik_match_table human joint are
                       identical to G1 -- the human skeleton is not ours to change
  C  scale table       == G1 x {torso .9333, legs .9089, arms .7707}, rounded to 3
  D  rename offsets    a link renamed to a DIFFERENT body carries a zeroed position
                       offset (the old offset was in the old body's frame)
  E  weights + scalars ik task weights, rotation offsets, and every scalar key are
                       byte-identical to G1

Usage:
    python validate_r1_configs.py                     # report to stdout
    python validate_r1_configs.py --out FILE          # and write it to FILE
"""

import argparse
import json
import sys
from pathlib import Path

GMR = Path.home() / "GMR"
IK = GMR / "general_motion_retargeting" / "ik_configs"
R1_XML = GMR / "assets" / "unitree_r1" / "r1_mocap.xml"
G1_XML = GMR / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"

FORMATS = ["smplx", "bvh_xsens", "bvh_lafan1", "bvh_nokov",
           "xsens_mvn", "fbx", "fbx_offline", "xrobot"]
TESTED = {"bvh_xsens", "bvh_lafan1"}          # actually run end to end

RATIO = {"torso": 0.84 / 0.9, "legs": 0.818 / 0.9, "arms": 0.578 / 0.75}
TORSO, LEGS = slice(0, 2), slice(2, 8)

LINK_SUBS = {
    "left_wrist_yaw_link": "left_wrist_roll_link",
    "right_wrist_yaw_link": "right_wrist_roll_link",
    "left_toe_link": "left_ankle_roll_link",
    "right_toe_link": "right_ankle_roll_link",
}
# Renames that move the target to a physically different body.  A wrist_yaw ->
# wrist_roll rename stays on the same limb tip; a toe -> ankle_roll does not.
DIFFERENT_BODY = {"left_toe_link", "right_toe_link"}

OFFSET_INDEX = 3        # [human_joint, pos_weight, rot_weight, pos_offset, rot_offset]

# Strings that describe the MOCAP skeleton, not the robot.  They have to be kept
# out of the generic sweep because some of them collide with robot body names:
# smplx calls its root joint "pelvis", and so does the G1.  Sweeping them would
# report references that are not references, which is how a real one gets missed.
HUMAN_KEYS = {"human_root_name", "human_height_assumption", "human_scale_table"}
HUMAN_JOINT_INDEX = 0   # element 0 of every ik_match_table entry


def body_names(xml):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}


def scale_group(i):
    return "torso" if i < TORSO.stop else ("legs" if i < LEGS.stop else "arms")


def walk_strings(node, path="$"):
    """Yield (json_path, string) for every string anywhere in the document."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def is_human_side(path):
    """True for a JSON path that names the mocap skeleton rather than the robot."""
    head = path.split(".")[1].split("[")[0] if path.count(".") >= 1 else ""
    if head in HUMAN_KEYS:
        return True
    # $.ik_match_table1.<link>[0] is the human joint the link is matched to
    return head.startswith("ik_match_table") and path.endswith(f"[{HUMAN_JOINT_INDEX}]")


def link_refs(config, g1_bodies):
    """
    Every place in the document that names a robot body, with its JSON path.

    Two sources, unioned so neither can hide a reference:
      - structural: robot_root_name, and the keys of any ik_match_table*, which
        are robot link names by definition
      - generic:    any string ANYWHERE in the document that names a G1 body or
        ends in _link -- this is what catches a block nobody thought about
    """
    refs = {}
    if isinstance(config.get("robot_root_name"), str):
        refs["$.robot_root_name"] = config["robot_root_name"]
    for key, value in config.items():
        if key.startswith("ik_match_table") and isinstance(value, dict):
            for link in value:
                refs[f"$.{key}[{link}] (key)"] = link
    for path, s in walk_strings(config):
        if is_human_side(path):
            continue
        if s in g1_bodies or s.endswith("_link"):
            refs[path] = s
    return refs


def human_joints(config):
    out = {"human_scale_table": list(config.get("human_scale_table", {}))}
    for key, value in config.items():
        if key.startswith("ik_match_table") and isinstance(value, dict):
            out[key] = [spec[0] for spec in value.values()]
    return out


def check(fmt, r1_bodies, g1_bodies, log):
    g1_path, r1_path = IK / f"{fmt}_to_g1.json", IK / f"{fmt}_to_r1.json"
    if not r1_path.exists():
        log(f"  {fmt}: NO R1 CONFIG"); return False
    g1 = json.loads(g1_path.read_text())
    r1 = json.loads(r1_path.read_text())
    fails = []

    # -- A: every robot-link reference anywhere in the file resolves ------------
    refs = link_refs(r1, g1_bodies)
    bad = {p: v for p, v in refs.items() if v not in r1_bodies}
    log(f"  A link resolution   {len(refs)} reference(s) swept, "
        f"{len(set(refs.values()))} distinct")
    if bad:
        fails.append("A")
        for p, v in sorted(bad.items()):
            log(f"      MISSING ON R1  {v!r}  at  {p}")
    extra = sorted({p for p in refs if not p.startswith("$.ik_match_table")
                    and p != "$.robot_root_name"})
    if extra:
        log(f"      outside the match tables: {', '.join(extra)}")

    # -- B: the human side is untouched ----------------------------------------
    hj_g1, hj_r1 = human_joints(g1), human_joints(r1)
    if hj_g1 != hj_r1:
        fails.append("B")
        for k in hj_g1:
            if hj_g1[k] != hj_r1.get(k):
                log(f"      HUMAN JOINTS DIFFER in {k}")
                log(f"        G1: {hj_g1[k]}")
                log(f"        R1: {hj_r1.get(k)}")
    log(f"  B human side        {'identical to G1' if 'B' not in fails else 'DIFFERS'}"
        f"  ({len(hj_g1['human_scale_table'])} scale keys)")

    # -- C: scale table is G1 x the fixed per-group ratio -----------------------
    g1_scale, r1_scale = g1["human_scale_table"], r1["human_scale_table"]
    mismatched = []
    for i, (joint, s) in enumerate(g1_scale.items()):
        want = round(s * RATIO[scale_group(i)], 3)
        got = r1_scale.get(joint)
        if got != want:
            mismatched.append((joint, s, want, got))
    if mismatched:
        fails.append("C")
        for joint, s, want, got in mismatched:
            log(f"      SCALE  {joint}: G1 {s} -> expected {want}, found {got}")
    vals = list(r1_scale.values())
    log(f"  C scale table       {'ok' if 'C' not in fails else 'MISMATCH'}   "
        f"torso {vals[0]}  legs {vals[2]}  arms {vals[8]}")

    # -- D: renames onto a different body drop the position offset -------------
    renamed = []
    for key, value in g1.items():
        if not (key.startswith("ik_match_table") and isinstance(value, dict)):
            continue
        for link, spec in value.items():
            if link not in LINK_SUBS:
                continue
            new = LINK_SUBS[link]
            got = r1[key].get(new)
            if got is None:
                fails.append("D"); log(f"      {new} absent from R1 {key}"); continue
            offset = got[OFFSET_INDEX]
            if link in DIFFERENT_BODY:
                if any(offset):
                    fails.append("D")
                    log(f"      OFFSET NOT ZEROED  {key}.{new}  {offset}  "
                        f"(was {spec[OFFSET_INDEX]} in the {link} frame)")
                renamed.append(f"{link}->{new} offset {spec[OFFSET_INDEX]}->{offset}")
            else:
                if offset != spec[OFFSET_INDEX]:
                    fails.append("D")
                    log(f"      OFFSET CHANGED on same-limb rename  {key}.{new}")
                renamed.append(f"{link}->{new}")
    log(f"  D rename offsets    {len(renamed)} rename(s)"
        + (f"  [{'; '.join(sorted(set(renamed)))}]" if renamed else "  none in this format"))

    # -- E: weights, rotation offsets and scalars are GMR's, unchanged ---------
    for key, value in g1.items():
        if key.startswith("ik_match_table") and isinstance(value, dict):
            for link, spec in value.items():
                new = LINK_SUBS.get(link, link)
                got = r1[key].get(new)
                if got is None:
                    continue
                if [spec[1], spec[2], spec[4]] != [got[1], got[2], got[4]]:
                    fails.append("E")
                    log(f"      WEIGHTS CHANGED  {key}.{new}: "
                        f"G1 {[spec[1], spec[2], spec[4]]} vs R1 {[got[1], got[2], got[4]]}")
        elif not isinstance(value, (dict, list)):
            if r1.get(key) != value:
                fails.append("E")
                log(f"      SCALAR CHANGED  {key}: G1 {value!r} vs R1 {r1.get(key)!r}")
    if set(g1) != set(r1):
        fails.append("E")
        log(f"      TOP-LEVEL KEYS DIFFER: only in G1 {sorted(set(g1)-set(r1))}, "
            f"only in R1 {sorted(set(r1)-set(g1))}")
    log(f"  E weights+scalars   {'unchanged from G1' if 'E' not in fails else 'CHANGED'}")

    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args()

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    r1_bodies, g1_bodies = body_names(R1_XML), body_names(G1_XML)
    log("Validating every *_to_r1.json against its G1 parent and r1_mocap.xml.")
    log(f"R1 model: {len(r1_bodies)} bodies.  G1 model: {len(g1_bodies)} bodies.")
    log("")

    results = {}
    for fmt in FORMATS:
        mark = "run end-to-end" if fmt in TESTED else "structural only"
        log(f"{fmt}_to_r1.json   ({mark})")
        results[fmt] = check(fmt, r1_bodies, g1_bodies, log)
        log(f"  => {'PASS' if results[fmt] else 'FAIL'}")
        log("")

    passed = sum(results.values())
    log(f"{passed} of {len(FORMATS)} configs pass all five checks.")
    if passed != len(FORMATS):
        log("FAILING: " + ", ".join(f for f, ok in results.items() if not ok))

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwritten to {args.out}")
    return 0 if passed == len(FORMATS) else 1


if __name__ == "__main__":
    sys.exit(main())
