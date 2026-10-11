"""Record the R1 tennis policy's nominal sim2sim rollout and render chosen moments (poster figures)."""
import os, sys, json, numpy as np, mujoco
sys.path.insert(0, os.getcwd())
import sim2sim_v5_dr as V, sim2sim_r1_tennis_v4_standalone as H

OUT = sys.argv[1]; MODE = sys.argv[2]          # MODE: "azimuths" or "final"
env = V.Env(); qs = []
env.rollout(V.SCENARIOS["nominal"], 0, hook=lambda t, m, d, fn: qs.append(d.qpos.copy()))
qs = np.array(qs); print("recorded", qs.shape)

spec = mujoco.MjSpec.from_file(H.R1_XML)       # same body tree / qpos layout as the harness model
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.10, 0.20, 0.30], rgb2=[0.20, 0.30, 0.40], width=300, height=300, mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.2, 0.3, 0.4])
spec.add_material(name="grid", textures=["", "grid"], texrepeat=[1, 1], texuniform=True, reflectance=0.2)
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6])
spec.visual.headlight.ambient = [0.35, 0.35, 0.35]; spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
m = spec.compile(); d = mujoco.MjData(m)
W, Hh = (int(os.environ.get("RW", 1624)), int(os.environ.get("RH", 1440)))
m.vis.global_.offwidth, m.vis.global_.offheight = W, Hh
r = mujoco.Renderer(m, Hh, W)
torso = m.body("torso_link").id
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE

def shot(t, az, el=-5, dist=2.55):
    d.qpos[:] = qs[t]; mujoco.mj_kinematics(m, d)
    cam.lookat[:] = d.xpos[torso] - [0, 0, 0.2]; cam.azimuth, cam.elevation, cam.distance = az, el, dist
    r.update_scene(d, cam); return r.render()

import imageio.v2 as iio
def step_for_video_frame(i):                     # GMR 30 fps frame i -> 50 Hz reference -> harness step
    return max(0, min(len(qs) - 1, round(i * 50 / 30) - 1))

if MODE == "azimuths":
    t = step_for_video_frame(int(sys.argv[3]))
    ims = [shot(t, az) for az in range(0, 360, 45)]
    small = [im[::3, ::3] for im in ims]
    iio.imwrite(OUT, np.vstack([np.hstack(small[:4]), np.hstack(small[4:])]))
else:
    az = float(sys.argv[3])
    for i in map(int, sys.argv[4].split(",")):
        iio.imwrite(f"{OUT}/r1_{i:03d}.png", shot(step_for_video_frame(i), az))
