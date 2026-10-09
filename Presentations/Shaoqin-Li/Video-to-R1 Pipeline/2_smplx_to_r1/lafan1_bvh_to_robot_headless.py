"""Headless LAFAN1-BVH -> robot retargeting.

Same job as GMR's `scripts/bvh_to_robot.py`, minus the interactive window.
`RobotMotionViewer` calls `mujoco.viewer.launch_passive` in its constructor, which
fails under WSL2 with `gladLoadGL error`, taking video recording down with it.  This
drives `mujoco.Renderer` offscreen instead, which does work.

The sibling script `xsens_bvh_to_robot_headless.py` does this for Xsens captures;
this one is for LAFAN1 (and Nokov, via --format), which use a different skeleton and
a different loader.

Reports the IK solve rate, which is the number worth quoting per robot/CPU pairing.
"""
import argparse
import os
import pickle
import time

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_file", required=True)
    ap.add_argument("--robot", default="unitree_r1")
    ap.add_argument("--format", default="lafan1", choices=["lafan1", "nokov"])
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--save_path", default=None, help="pickle of retargeted qpos frames")
    ap.add_argument("--start", type=int, default=None, help="first frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame (exclusive)")
    ap.add_argument("--motion_fps", type=int, default=30, help="LAFAN1 is captured at 30 fps")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    args = ap.parse_args()

    frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    total = len(frames)
    lo = args.start or 0
    hi = args.end if args.end is not None else total
    frames = frames[lo:hi]
    print(f"[headless] {os.path.basename(args.bvh_file)}: {total} frames total, "
          f"using [{lo}:{hi}] = {len(frames)} @ {args.motion_fps} fps, "
          f"human height {actual_human_height:.3f} m")

    retargeter = GMR(src_human=f"bvh_{args.format}", tgt_robot=args.robot,
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
        writer = imageio.get_writer(args.video_path, fps=args.motion_fps)

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
          f"({solve_time:.1f}s for {len(frames)} frames), motion is {args.motion_fps} fps "
          f"-> {'real-time capable' if ik_fps >= args.motion_fps else 'slower than real time'}")

    if writer is not None:
        writer.close()
        print(f"[headless] video saved to {args.video_path}")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        with open(args.save_path, "wb") as f:
            pickle.dump({"root_pos": np.array([q[:3] for q in qpos_list]),
                         "root_rot": np.array([q[3:7] for q in qpos_list]),
                         "dof_pos": np.array([q[7:] for q in qpos_list]),
                         "fps": args.motion_fps,
                         "robot": args.robot,
                         "source": os.path.basename(args.bvh_file),
                         "frame_range": [lo, hi]}, f)
        print(f"[headless] motion saved to {args.save_path}")


if __name__ == "__main__":
    main()
