# Render the reference motion (kinematic, no physics) at given 50 Hz steps, front view.
import os, sys
sys.path.insert(0, os.getcwd())
import mujoco, numpy as np
from PIL import Image
import sim2sim_r1_tennis_v4_standalone as H
npz, out = sys.argv[1], sys.argv[2]; steps = [int(s) for s in sys.argv[3].split(",")]
d0 = np.load(npz)
spec = mujoco.MjSpec.from_file(H.R1_XML)
spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
m = spec.compile(); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 480, 360); cam = mujoco.MjvCamera(); cam.distance = 3.0; cam.azimuth = 270; cam.elevation = -5
imgs = []
for s in steps:
    d.qpos[:3] = d0["body_pos_w"][s, 0]; d.qpos[3:7] = d0["body_quat_w"][s, 0]; d.qpos[7:] = d0["joint_pos"][s]
    mujoco.mj_forward(m, d); cam.lookat[:] = d.qpos[:3]; r.update_scene(d, cam); imgs.append(r.render())
Image.fromarray(np.hstack(imgs)).save(out)
print("njnt", m.njnt, "nq", m.nq, d0["joint_pos"].shape)
