"""Diagnostic: per-joint tracking error of a nominal rollout (records the 24-vector the harness norms)."""
import numpy as np
import sim2sim_v5_dr as V, sim2sim_r1_tennis_v4_standalone as H
rec, _norm = [], np.linalg.norm
def norm(x, *a, **k):
    if not a and not k and np.shape(x) == (24,): rec.append(np.array(x))
    return _norm(x, *a, **k)
np.linalg.norm = norm
env = V.Env(); r = env.rollout(V.SCENARIOS["nominal"], 0)
np.linalg.norm = _norm
E = np.array(rec); rms = np.sqrt((E**2).mean(0)); bias = E.mean(0)
print(f"survived={r['survived']} norm-mean={r['mean_err']:.3f}  per-joint RMS overall={np.sqrt((E**2).mean()):.3f} rad")
for i in np.argsort(-rms)[:8]:
    print(f"  {H.JOINT_NAMES[i]:28s} rms {rms[i]:.3f}  mean(signed) {bias[i]:+.3f}")
# lag check: does the robot trail the reference by k control steps?
jp = env.jp[1:1+len(E)]; robot = E + jp
for k in range(0, 6):
    e = np.sqrt(((robot[k:] - jp[:len(jp)-k])**2).mean()); print(f"  lag {k*20:3d} ms: per-joint RMS {e:.3f}")
