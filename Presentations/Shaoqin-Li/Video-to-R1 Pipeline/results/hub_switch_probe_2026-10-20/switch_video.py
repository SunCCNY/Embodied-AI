import sys, os
pass
import numpy as np, mujoco, imageio
import switch_probe as W
a, b, variant, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
A, B = W.Pol(a), W.Pol(b)
st = W.stand_pose(B.env)[:3]
frames = []; state = {}
def hook(model, data, seg):
    if "r" not in state:
        state["r"] = mujoco.Renderer(model, 480, 640)
        c = mujoco.MjvCamera(); c.distance = 3.2; c.azimuth = 120; c.elevation = -8; state["cam"] = c
    state["cam"].lookat[:] = [data.qpos[0], data.qpos[1], 0.7]
    state["r"].update_scene(data, state["cam"])
    img = state["r"].render().copy()
    state["segs"].append(seg); frames.append(img)
state["segs"] = []
W.HOOK = hook
res = W.episode(A, B, variant, W.SCN["train_DR"], seed, st)
print(res)
from PIL import Image, ImageDraw
for i, (f, s) in enumerate(zip(frames, state["segs"])):
    im = Image.fromarray(f); ImageDraw.Draw(im).text((10, 10), s, fill=(255, 255, 255)); frames[i] = np.asarray(im)
imageio.mimsave(out, frames, fps=50, macro_block_size=1)
print("video", out, len(frames), "frames")
