"""Headless GVHMR -> robot retargeting: the join between stage 1 (video) and stage 2.

Same job as GMR's `scripts/gvhmr_to_robot.py`, minus the interactive window.
Upstream constructs `RobotMotionViewer` unconditionally, and that calls
`mujoco.viewer.launch_passive`, which fails under WSL2 (`gladLoadGL error`) -- so
there is no headless path even with `--save_path`.  This drives `mujoco.Renderer`
offscreen instead.

Fourth sibling of `xsens_bvh_to_robot_headless.py`, `lafan1_bvh_to_robot_headless.py`
and `smplx_to_robot_headless.py`.  It differs from the smplx one only in the loader:
GVHMR's `hmr4d_results.pt` holds SMPL-X parameters in a y-up world with 10 betas;
GMR's `load_gvhmr_pred_file` pads betas to 16 and `get_gvhmr_data_offline_fast`
applies the y-up -> z-up rotation.  The retargeting config is the same
`smplx_to_<robot>.json` that OMOMO used -- no new config for video.

GROUND PASS (added 2026-09-17).  GVHMR's world-frame output has no floor: on
tennis.mp4 the lowest foot joint climbs 0.005 -> 0.15 m over 10 s, a clean linear
ramp (1.62 cm/s, residual sd 1.7 cm), and the robot inherits it as a +15 cm float.
`--ground linear` (default) fits a line to the lower envelope of the lowest foot
joint and subtracts it per frame, so the floor is flat but real vertical motion
(crouch, hop) survives.  `const` subtracts only the clip minimum (keeps the
drift); `perframe` uses GMR's own `offset_to_ground=True`, which pins the lowest
foot to +0.10 m on EVERY frame (kills vertical motion); `none` is the old
behaviour.  The pass is on the HUMAN frames, before GMR; the retargeter is untouched.

Input:  GVHMR/outputs/demo/<clip>/hmr4d_results.pt
Output: the usual GMR motion pickle (root_pos, root_rot wxyz, dof_pos, fps),
        which is the reference-motion format imitation learning consumes.
"""
import argparse
import os
import pathlib
import pickle
import time

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

BODY_MODELS = pathlib.Path.home() / "GMR" / "assets" / "body_models"  # SMPL-X models (license-gated, not shipped)
FOOT_JOINTS = ("left_foot", "right_foot")   # SMPL-X toe joints, the lowest points of the body


def lowest_foot_z(frames):
    return np.array([min(f[j][0][2] for j in FOOT_JOINTS) for f in frames])


def fit_floor(low, fps, mode):
    """Return the per-frame floor height to subtract, plus a description."""
    n = len(low)
    if mode == "const":
        return np.full(n, low.min()), f"const {low.min():.3f} m"
    # lower envelope: rolling minimum over +-0.5 s, then a straight line through it
    w = max(1, int(round(0.5 * fps)))
    env = np.array([low[max(0, i - w):i + w + 1].min() for i in range(n)])
    t = np.arange(n)
    a, b = np.polyfit(t, env, 1)
    line = a * t + b
    line += (low - line).min()          # so the clip's lowest foot lands exactly on the floor
    return line, (f"linear slope {a * fps * 100:+.2f} cm/s, intercept {b:.3f} m, "
                  f"envelope resid sd {np.std(env - (a * t + b)) * 100:.2f} cm")


def apply_ground_pass(frames, fps, mode, target):
    """Shift every joint of every frame down by the fitted floor (+ target)."""
    low = lowest_foot_z(frames)
    floor, desc = fit_floor(low, fps, mode)
    shift = floor - target
    out = []
    for f, dz in zip(frames, shift):
        out.append({k: (v[0] - np.array([0.0, 0.0, dz]), v[1]) for k, v in f.items()})
    low2 = lowest_foot_z(out)
    print(f"[ground] mode={mode}: {desc}")
    print(f"[ground] lowest foot joint before {low.min():.3f}..{low.max():.3f} m "
          f"-> after {low2.min():.3f}..{low2.max():.3f} m (target {target:.3f})")
    return out, {"mode": mode, "fit": desc, "target_m": target,
                 "before_m": [float(low.min()), float(low.max())],
                 "after_m": [float(low2.min()), float(low2.max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvhmr_pred_file", required=True, help="GVHMR hmr4d_results.pt")
    ap.add_argument("--robot", default="unitree_r1")
    ap.add_argument("--body_models", default=str(BODY_MODELS),
                    help="folder holding smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz")
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--save_path", default=None, help="pickle of retargeted qpos frames")
    ap.add_argument("--start", type=int, default=None, help="first frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame (exclusive)")
    ap.add_argument("--tgt_fps", type=int, default=30)
    ap.add_argument("--src_fps", type=float, default=None,
                    help="frame rate of the source VIDEO. GMR's GVHMR loader assumes 30 and never resamples a slower\n"
                         "clip, so a 25 fps video would play 1.2x too fast. Pass the real value (e.g. 25, 24).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--ground", choices=["none", "const", "linear", "perframe"], default="linear",
                    help="floor pass on the human frames before IK (see module docstring)")
    ap.add_argument("--ground_target", type=float, default=0.0,
                    help="height (m) the clip's lowest foot joint is placed at after the pass")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    args = ap.parse_args()

    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, args.body_models)
    tgt_fps = args.tgt_fps
    if args.src_fps is not None:
        import torch
        smplx_data["mocap_frame_rate"] = torch.tensor(args.src_fps)
        if args.src_fps <= args.tgt_fps:   # GMR only downsamples; keep the native rate and label it correctly
            tgt_fps = args.src_fps
    frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    total = len(frames)
    lo = args.start or 0
    hi = args.end if args.end is not None else total
    frames = frames[lo:hi]
    clip = pathlib.Path(args.gvhmr_pred_file).parent.name
    print(f"[headless] {clip}/{os.path.basename(args.gvhmr_pred_file)}: {total} frames total, "
          f"using [{lo}:{hi}] = {len(frames)} @ {aligned_fps} fps, "
          f"human height {actual_human_height:.3f} m (from betas[0])")

    ground_info = {"mode": args.ground}
    if args.ground in ("const", "linear"):
        frames, ground_info = apply_ground_pass(frames, aligned_fps, args.ground, args.ground_target)
    elif args.ground == "perframe":
        print("[ground] mode=perframe: GMR offset_to_ground=True (lowest foot pinned to +0.10 m each frame)")

    retargeter = GMR(src_human="smplx", tgt_robot=args.robot,
                     actual_human_height=actual_human_height)

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mujoco.MjData(model)
    base_id = model.body(ROBOT_BASE_DICT[args.robot]).id

    cam = mujoco.MjvCamera()
    cam.distance = VIEWER_CAM_DISTANCE_DICT[args.robot]
    cam.elevation = args.elevation
    cam.azimuth = args.azimuth

    renderer = writer = None
    if args.video_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        writer = imageio.get_writer(args.video_path, fps=aligned_fps)

    qpos_list = []
    solve_time = 0.0
    for frame in tqdm(frames, desc=f"retargeting -> {args.robot}"):
        t0 = time.perf_counter()
        qpos = retargeter.retarget(frame, offset_to_ground=(args.ground == "perframe"))
        solve_time += time.perf_counter() - t0
        qpos_list.append(qpos.copy())

        if renderer is not None:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cam.lookat = data.xpos[base_id]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

    ik_fps = len(frames) / solve_time
    print(f"[headless] IK solve rate: {ik_fps:.1f} FPS "
          f"({solve_time:.1f}s for {len(frames)} frames), motion is {aligned_fps} fps "
          f"-> {'real-time capable' if ik_fps >= aligned_fps else 'slower than real time'}")

    if writer is not None:
        writer.close()
        print(f"[headless] video saved to {args.video_path}")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        with open(args.save_path, "wb") as f:
            pickle.dump({"root_pos": np.array([q[:3] for q in qpos_list]),
                         "root_rot": np.array([q[3:7] for q in qpos_list]),
                         "dof_pos": np.array([q[7:] for q in qpos_list]),
                         "fps": aligned_fps,
                         "robot": args.robot,
                         "source": f"gvhmr/{clip}",
                         "frame_range": [lo, hi],
                         "ground_pass": ground_info}, f)
        print(f"[headless] motion saved to {args.save_path}")


if __name__ == "__main__":
    main()
