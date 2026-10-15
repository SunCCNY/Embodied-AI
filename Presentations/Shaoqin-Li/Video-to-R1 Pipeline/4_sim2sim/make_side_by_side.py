"""Side-by-side video: the original clip next to the R1 policy performing the same motion in plain MuJoCo (sim2sim).

Run from the mjlab root (needs MUJOCO_GL=osmesa or another offscreen GL):
  ONNX=<policy.onnx> MOTION=<motion.npz> python make_side_by_side.py <video.mp4> <video_fps> <azimuth> <out.mp4> \
      [--stills 40,150] [--scenario nominal] [--seed 0]
The R1 frame for video frame i is harness step round(i * 50 / video_fps) - 1, so the two sides stay in sync as long as
the motion was retargeted with the video's real frame rate (gvhmr_to_robot_headless.py --src_fps).
"""
import argparse
import os
import sys

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.getcwd())
import sim2sim_v5_dr as V  # noqa: E402
import sim2sim_r1_tennis_v4_standalone as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("fps", type=float); ap.add_argument("azimuth", type=float); ap.add_argument("out")
ap.add_argument("--stills", default=""); ap.add_argument("--scenario", default="nominal"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--height", type=int, default=720); ap.add_argument("--robot_width", type=int, default=600)
a = ap.parse_args()

env = V.Env(); qs = []
res = env.rollout(V.SCENARIOS[a.scenario], a.seed, hook=lambda t, m, d, fn: qs.append(d.qpos.copy()))
print(f"rollout {a.scenario} seed {a.seed}: survived={res['survived']} steps={res['steps']} joint err={res['mean_err']:.3f}")

spec = mujoco.MjSpec.from_file(H.R1_XML)                   # same body tree / qpos layout as the harness model
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.10, 0.20, 0.30], rgb2=[0.20, 0.30, 0.40], width=300, height=300)
spec.add_material(name="grid", textures=["", "grid"], texrepeat=[1, 1], texuniform=True, reflectance=0.2)
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6])
spec.visual.headlight.ambient = [0.35, 0.35, 0.35]; spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
m = spec.compile(); d = mujoco.MjData(m)
H_, W_ = a.height, a.robot_width
m.vis.global_.offwidth, m.vis.global_.offheight = max(W_, 640), max(H_, 480)
r = mujoco.Renderer(m, H_, W_)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
torso = m.body("torso_link").id

font = ImageFont.truetype("/mnt/c/Windows/Fonts/arialbd.ttf", 24) if os.path.exists("/mnt/c/Windows/Fonts/arialbd.ttf") else ImageFont.load_default()
stills = {int(s) for s in a.stills.split(",") if s}
reader = iio.get_reader(a.video)
writer = iio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
for i, frame in enumerate(reader):
    t = max(0, min(len(qs) - 1, round(i * 50 / a.fps) - 1))
    d.qpos[:] = qs[t]; mujoco.mj_kinematics(m, d)
    cam.lookat[:] = d.xpos[torso] - [0, 0, 0.2]; cam.azimuth, cam.elevation, cam.distance = a.azimuth, -5, 2.6
    r.update_scene(d, cam)
    vid = Image.fromarray(frame); vid = vid.resize((int(vid.width * H_ / vid.height) // 2 * 2, H_))
    rob = Image.fromarray(r.render())
    canvas = Image.new("RGB", (vid.width + 8 + W_, H_), "white")
    canvas.paste(vid, (0, 0)); canvas.paste(rob, (vid.width + 8, 0))
    dr = ImageDraw.Draw(canvas)
    for x, label in ((10, "Original video"), (vid.width + 18, "R1 policy in MuJoCo (sim2sim)")):
        tw = dr.textlength(label, font=font); dr.rectangle([x - 4, 8, x + tw + 6, 40], fill=(0, 0, 0)); dr.text((x, 10), label, font=font, fill="white")
    writer.append_data(np.asarray(canvas))
    if i in stills:
        canvas.save(os.path.splitext(a.out)[0] + f"_frame{i}.png")
writer.close(); print("wrote", a.out, i + 1, "frames")
