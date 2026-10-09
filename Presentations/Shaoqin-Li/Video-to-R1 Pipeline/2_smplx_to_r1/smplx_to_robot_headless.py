"""Headless SMPL-X -> robot retargeting.

Same job as GMR's `scripts/smplx_to_robot.py`, minus the interactive window.
`RobotMotionViewer` calls `mujoco.viewer.launch_passive` in its constructor, which
fails under WSL2 with `gladLoadGL error`, and that constructor runs unconditionally
in the upstream script -- there is no headless path even when you only want
`--save_path`.  This drives `mujoco.Renderer` offscreen instead, which does work.

Third sibling of `xsens_bvh_to_robot_headless.py` (Xsens BVH) and
`lafan1_bvh_to_robot_headless.py` (LAFAN1/Nokov BVH).  SMPL-X differs from both: the
motion is not a skeleton animation but pose parameters that have to be pushed through
an SMPL-X body model first, so this one needs `assets/body_models/smplx/` present.

Reports the IK solve rate, which is the number worth quoting per robot/CPU pairing.
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
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

BODY_MODELS = pathlib.Path.home() / "GMR" / "assets" / "body_models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx_file", required=True)
    ap.add_argument("--robot", default="unitree_r1")
    ap.add_argument("--body_models", default=str(BODY_MODELS),
                    help="folder holding smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz")
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--save_path", default=None, help="pickle of retargeted qpos frames")
    ap.add_argument("--start", type=int, default=None, help="first frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame (exclusive)")
    ap.add_argument("--tgt_fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    args = ap.parse_args()

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, args.body_models)
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps)

    total = len(frames)
    lo = args.start or 0
    hi = args.end if args.end is not None else total
    frames = frames[lo:hi]
    print(f"[headless] {os.path.basename(args.smplx_file)}: {total} frames total, "
          f"using [{lo}:{hi}] = {len(frames)} @ {aligned_fps} fps, "
          f"human height {actual_human_height:.3f} m")

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
        qpos = retargeter.retarget(frame)
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
                         "source": os.path.basename(args.smplx_file),
                         "frame_range": [lo, hi]}, f)
        print(f"[headless] motion saved to {args.save_path}")


if __name__ == "__main__":
    main()
