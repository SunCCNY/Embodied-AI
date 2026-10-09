"""Measure GVHMR's vertical drift on the SOURCE human data (before any retargeting).

Prints the lowest SMPL-X foot joint per frame after GMR's y-up -> z-up rotation,
its straight-line fit, and the pelvis/head heights, so the drift can be seen on
the human before it is blamed on the robot.  Run inside the `gmr` env.
"""
import argparse, numpy as np
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

ap = argparse.ArgumentParser()
ap.add_argument("pred", help="GVHMR hmr4d_results.pt")
ap.add_argument("--body_models", default=str(__import__("pathlib").Path.home() / "GMR" / "assets" / "body_models"))
ap.add_argument("--every", type=int, default=30)
a = ap.parse_args()

d, bm, out, h = load_gvhmr_pred_file(a.pred, a.body_models)
frames, fps = get_gvhmr_data_offline_fast(d, bm, out, tgt_fps=30)
z = lambda n: np.array([f[n][0][2] for f in frames])
low = np.minimum(z("left_foot"), z("right_foot"))
pel, head = z("pelvis"), z("head")
print(f"{len(frames)} frames @ {fps} fps, human height {h:.3f} m")
print(f"{'frame':>5} {'t(s)':>5} {'lowest foot':>12} {'pelvis':>8} {'head':>7}")
for i in range(0, len(frames), a.every):
    print(f"{i:5d} {i/fps:5.1f} {low[i]:12.3f} {pel[i]:8.3f} {head[i]:7.3f}")
t = np.arange(len(low)); s, b = np.polyfit(t, low, 1)
print(f"lowest foot joint: {low.min():.3f} .. {low.max():.3f} m")
print(f"straight-line fit: {s*fps*100:+.2f} cm/s, residual sd {np.std(low-(s*t+b))*100:.2f} cm")
