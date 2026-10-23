import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
import sim2sim_v5_dr as V
env = V.Env(); rec = []
def hook(t, m, d, fn):
    floor = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    n = sum(1 for k in range(d.ncon) if d.contact[k].geom1 in floor or d.contact[k].geom2 in floor)
    rec.append((n, d.qpos[2]))
res = env.rollout(V.SCENARIOS["nominal"], 0, hook=hook)
a = np.array(rec); air = a[:, 0] == 0
starts = np.sum(air[1:] & ~air[:-1])
print(os.environ["MOTION"], "survived", res["survived"], "steps", len(a), f"airborne {air.mean()*100:.1f}% of steps,", starts, "flight phases, pelvis z", a[:,1].min().round(3), a[:,1].max().round(3))
