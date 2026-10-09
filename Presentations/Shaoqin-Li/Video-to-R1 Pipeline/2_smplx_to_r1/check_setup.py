"""Am I ready for Lab 4?  Run this BEFORE the session.

    conda activate gmr
    python check_setup.py

It prints one line per requirement and a single verdict at the end.  Bring that
verdict to the lab.  If it says READY, you are done -- nothing else to install.

If something fails, the message tells you which command in LAB4_setup_gmr.md
fixes it.  Do not wait until the session to find out.
"""
import importlib, os, sys

OK, BAD, SKIP = "  [ok]  ", "  [--]  ", "  [..]  "
problems = []


def check(label, fn, fix, needs=True):
    # `needs` gates a check on an earlier one.  Without this, a wrong conda env
    # makes every downstream check fail and tell you to re-clone the repo --
    # four confident wrong answers instead of the one real cause.
    if not needs:
        print(SKIP + f"{label:38s} skipped -- fix the above first")
        return False
    try:
        detail = fn()
    except Exception as e:
        print(BAD + f"{label:38s} {type(e).__name__}")
        problems.append((label, fix))
        return False
    if detail is False:
        print(BAD + f"{label:38s} not found")
        problems.append((label, fix))
        return False
    print(OK + f"{label:38s} {detail if detail is not True else ''}")
    return True


print("\n" + "=" * 66)
print("  Lab 4 setup check")
print("=" * 66)

check("python 3.10+",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor}" if sys.version_info >= (3, 10)
      else False,
      "conda create -n gmr python=3.10 -y")

check("mujoco",
      lambda: importlib.import_module("mujoco").__version__,
      "pip install -e .   (inside the GMR folder)")

check("mink (the IK solver)",
      lambda: importlib.import_module("mink") and True,
      "pip install -e .   (inside the GMR folder)")

gmr_ok = check("general_motion_retargeting",
               lambda: importlib.import_module("general_motion_retargeting") and True,
               "you are probably in the wrong environment.  Run: conda activate gmr\n"
               "        If that is not it: cd into the GMR folder and run  pip install -e .")


def r1_model():
    from general_motion_retargeting import ROBOT_XML_DICT
    p = ROBOT_XML_DICT.get("unitree_r1")
    return str(p).split(os.sep)[-1] if p and os.path.exists(p) else False


check("R1 robot model", r1_model,
      "the R1 assets are missing -- re-clone with --recurse-submodules", gmr_ok)


def r1_configs():
    from general_motion_retargeting import IK_CONFIG_DICT
    have = [k for k, v in IK_CONFIG_DICT.items() if "unitree_r1" in v]
    return ", ".join(have) if "bvh_xsens" in have else False


check("R1 motion mappings", r1_configs,
      "bvh_xsens_to_r1.json is missing from general_motion_retargeting/ik_configs/", gmr_ok)


def the_motion():
    from general_motion_retargeting import ASSET_ROOT
    p = os.path.join(str(ASSET_ROOT), "xsens_bvh_test",
                     "251021_04_boxing_120Hz_cm_3DsMax.bvh")
    return f"{os.path.getsize(p)//1024} KB" if os.path.exists(p) else False


check("the boxing capture", the_motion,
      "assets/xsens_bvh_test/ is missing -- re-clone with --recurse-submodules", gmr_ok)


def menagerie():
    from general_motion_retargeting import ASSET_ROOT
    p = os.path.join(os.path.dirname(str(ASSET_ROOT).rstrip(os.sep)), "mujoco_menagerie")
    return True if os.path.isdir(p) and os.listdir(p) else False


check("mujoco_menagerie submodule", menagerie,
      "git submodule update --init --recursive", gmr_ok)

check("video writer (imageio)",
      lambda: importlib.import_module("imageio") and True,
      "pip install imageio imageio-ffmpeg")

print("=" * 66)
if not problems:
    print("""
  READY.  Nothing else to install.

  Try the real thing now, so nothing surprises you in the session:

    python scripts/xsens_bvh_to_robot_headless.py \\
        --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \\
        --robot unitree_r1 --start 600 --end 1800 --save_path out/mine.pkl

  It should finish in well under a minute and report an IK solve rate above
  100 FPS.  Then:

    python exp4_check_motion.py out/mine.pkl
""")
else:
    print(f"\n  NOT READY -- {len(problems)} thing(s) to fix:\n")
    for label, fix in problems:
        print(f"    {label}\n        {fix}\n")
    print("  Full instructions: LAB4_setup_gmr.md")
    print("  Fix these BEFORE the session -- there will not be time during it.\n")
print("=" * 66 + "\n")
sys.exit(1 if problems else 0)
