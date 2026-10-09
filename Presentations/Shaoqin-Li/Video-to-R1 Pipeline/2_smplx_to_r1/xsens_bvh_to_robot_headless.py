"""Headless Xsens-BVH -> robot retargeting.

Same pipeline as `xsens_bvh_to_robot.py`, but it never opens an interactive
window.  `RobotMotionViewer` always calls `mujoco.viewer.launch_passive`, which
fails under WSL2 (`gladLoadGL error`) even with software GL, so video recording
is unreachable there.  This script drives `mujoco.Renderer` directly instead,
which works offscreen with the default backend.

Also reports the IK solve rate, which is the number worth quoting for a given
robot/CPU pairing.
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
from general_motion_retargeting.utils.xsens import load_xsens_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_file", required=True)
    ap.add_argument("--robot", default="unitree_r1")
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--save_path", default=None, help="pickle of retargeted qpos frames")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--scale", type=float, default=0.01,
                    help="BVH translation units -> metres (Xsens 3DsMax exports are cm)")
    ap.add_argument("--reset_to_zero", action="store_true", default=False)
    ap.add_argument("--bvh_format", default="3DSM", choices=["3DSM"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    args = ap.parse_args()

    frames, actual_human_height, frame_time = load_xsens_file(args)
    motion_fps = int(round(1.0 / frame_time))
    print(f"[headless] {len(frames)} frames @ {motion_fps} fps, "
          f"human height {actual_human_height:.3f} m")

    retargeter = GMR(src_human="bvh_xsens", tgt_robot=args.robot,
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
        writer = imageio.get_writer(args.video_path, fps=motion_fps)

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
          f"({solve_time:.1f}s for {len(frames)} frames), motion is {motion_fps} fps "
          f"-> {'real-time capable' if ik_fps >= motion_fps else 'slower than real time'}")

    if writer is not None:
        writer.close()
        print(f"[headless] video saved to {args.video_path}")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        with open(args.save_path, "wb") as f:
            pickle.dump({"root_pos": np.array([q[:3] for q in qpos_list]),
                         "root_rot": np.array([q[3:7] for q in qpos_list]),
                         "dof_pos": np.array([q[7:] for q in qpos_list]),
                         "fps": motion_fps,
                         "robot": args.robot}, f)
        print(f"[headless] motion saved to {args.save_path}")


if __name__ == "__main__":
    main()
