"""Add the Unitree R1 to a stock GMR clone -- Lab 4, Part 0.

GMR ships eleven robots.  The R1 is not one of them: support for it was written
by this project, and a clone straight from GitHub cannot retarget onto an R1 at
all.  Adding a robot to a retargeting tool is four things, and this script does
all four, printing each one:

  1. the robot model            assets/unitree_r1/r1_mocap.xml + r1_scene.xml
  2. its 43 shapes              the same STL files you downloaded in Lab 1
  3. two IK configs             bvh_xsens_to_r1.json, smplx_to_r1.json
  4. five entries in params.py  model path, 2 IK configs, root body, camera

Run it once, from the exp4 folder, with the `gmr` environment active:

    python install_r1_into_gmr.py

It is safe to run twice -- anything already in place is reported and skipped.
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GMR = os.path.join(os.path.expanduser("~"), "GMR")
DEFAULT_MESHES = os.path.join(os.path.expanduser("~"), "r1_lab", "exp1",
                              "model", "assets")

# (file in support/, destination relative to the GMR clone)
FILES = [
    ("r1_mocap.xml", os.path.join("assets", "unitree_r1", "r1_mocap.xml")),
    ("r1_scene.xml", os.path.join("assets", "unitree_r1", "r1_scene.xml")),
    ("bvh_xsens_to_r1.json", os.path.join("general_motion_retargeting",
                                          "ik_configs", "bvh_xsens_to_r1.json")),
    ("smplx_to_r1.json", os.path.join("general_motion_retargeting",
                                      "ik_configs", "smplx_to_r1.json")),
]

# (anchor line already in params.py, the line to insert after it)
PARAMS = [
    ('    "unitree_h1": ASSET_ROOT / "unitree_h1" / "h1.xml",',
     '    "unitree_r1": ASSET_ROOT / "unitree_r1" / "r1_scene.xml",'),
    ('        "unitree_h1": IK_CONFIG_ROOT / "smplx_to_h1.json",',
     '        "unitree_r1": IK_CONFIG_ROOT / "smplx_to_r1.json",'),
    ('        "unitree_h1_2": IK_CONFIG_ROOT / "bvh_xsens_to_h1_2.json",',
     '        "unitree_r1": IK_CONFIG_ROOT / "bvh_xsens_to_r1.json",'),
    ('    "unitree_h1_2": "pelvis",', '    "unitree_r1": "pelvis",'),
    ('    "unitree_h1_2": 3.0,', '    "unitree_r1": 2.0,'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmr", default=DEFAULT_GMR, help="your GMR clone")
    ap.add_argument("--meshes", default=DEFAULT_MESHES,
                    help="the 43 STL files from Lab 1")
    ap.add_argument("--support", default=os.path.join(HERE, "support"),
                    help="the four R1 support files")
    a = ap.parse_args()

    if not os.path.isdir(a.gmr):
        raise SystemExit("no GMR clone at %s -- see the setup guide" % a.gmr)
    print("GMR clone:", a.gmr)

    # 1 + 3. the four text files
    for name, rel in FILES:
        src = os.path.join(a.support, name)
        if not os.path.exists(src):
            raise SystemExit("missing %s -- did the support download land?" % src)
        dst = os.path.join(a.gmr, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        same = os.path.exists(dst) and open(dst, "rb").read() == open(src, "rb").read()
        shutil.copyfile(src, dst)
        print("  %-24s -> %s%s" % (name, rel, "  (already there)" if same else ""))

    # 2. the shapes -- the same STLs Lab 1 pulled from Unitree's repository
    dst_assets = os.path.join(a.gmr, "assets", "unitree_r1", "assets")
    if not os.path.isdir(a.meshes):
        raise SystemExit("no meshes at %s -- pass --meshes" % a.meshes)
    os.makedirs(dst_assets, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(a.meshes)):
        shutil.copyfile(os.path.join(a.meshes, f), os.path.join(dst_assets, f))
        n += 1
    print("  %d meshes copied from %s" % (n, a.meshes))

    # 4. five entries in params.py
    params = os.path.join(a.gmr, "general_motion_retargeting", "params.py")
    text = open(params).read()
    added = 0
    for anchor, line in PARAMS:
        if line in text:
            continue
        if anchor not in text:
            raise SystemExit("cannot find the anchor line in params.py:\n  %s\n"
                             "GMR has changed upstream; patch it by hand." % anchor)
        text = text.replace(anchor, anchor + "\n" + line, 1)
        added += 1
    if added:
        shutil.copyfile(params, params + ".before_r1")
        open(params, "w").write(text)
    print("  params.py: %d entr%s added%s" % (added, "y" if added == 1 else "ies",
                                              " (backup: params.py.before_r1)"
                                              if added else " -- already patched"))

    # verify, the only line that matters
    sys.path.insert(0, a.gmr)
    from general_motion_retargeting import IK_CONFIG_DICT
    have = {k: "unitree_r1" in v for k, v in IK_CONFIG_DICT.items()}
    print("\n  R1 accepted from:", ", ".join(k for k, v in have.items() if v) or "nothing")
    assert have.get("bvh_xsens") and have.get("smplx"), have
    print("  OK -- the R1 is installed in GMR")


if __name__ == "__main__":
    main()
