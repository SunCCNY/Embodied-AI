"""Build the R1 training-equivalent MuJoCo model from r1.xml alone (no mjlab import).
KEEP_RAW is a set of settings left at the raw r1.xml value, for ablations:
  floor, timestep, iters, condim, friction, priority"""
import re
import mujoco
import numpy as np

R1_XML = "src/mjlab/asset_zoo/robots/unitree_r1/xmls/r1.xml"
FOOT = re.compile(r"^(left|right)_foot[1-7]_collision$")


def build_model(joint_names, stiffness, damping, effort_limit, keep_raw=()):
    keep = set(keep_raw)
    spec = mujoco.MjSpec.from_file(R1_XML)

    if "floor" not in keep:
        g = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
        g.friction = [1.0, 0.005, 0.0001]
        g.condim = 3
        g.solref = [0.02, 1.0]
        g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    for g in spec.geoms:
        if not (g.name or "").endswith("_collision"):
            continue
        is_foot = bool(FOOT.match(g.name))
        if "condim" not in keep:
            g.condim = 3 if is_foot else 1
        if "priority" not in keep:
            g.priority = 1 if is_foot else 0
        if "friction" not in keep and is_foot:
            f = list(g.friction)
            f[0] = 0.6
            g.friction = f

    for i, j in enumerate(joint_names):
        a = spec.add_actuator(name=j, target=j)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = float(stiffness[i])
        a.biasprm[1] = -float(stiffness[i])
        a.biasprm[2] = -float(damping[i])
        a.inheritrange = 0.0
        a.ctrllimited = False
        a.forcelimited = True
        a.forcerange = [-float(effort_limit[i]), float(effort_limit[i])]

    if "timestep" not in keep:
        spec.option.timestep = 0.005
    if "iters" not in keep:
        spec.option.iterations = 10
        spec.option.ls_iterations = 20
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return spec.compile()
