"""Compile mjlab r1.xml and emit a collision-only URDF (serial chain, no meshes) for Isaac Lab import."""
import os
import re
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

XML = os.environ.get("R1_XML", "src/mjlab/asset_zoo/robots/unitree_r1/xmls/r1.xml")  # run from the mjlab clone, or set R1_XML
OUT = sys.argv[1] if len(sys.argv) > 1 else "r1.urdf"
VEL = 50.0
m = mujoco.MjModel.from_xml_path(XML)
EFFORT = [(r"hip_(pitch|roll|yaw)", 88.0), (r"knee", 139.0), (r"ankle", 50.0), (r"waist_roll", 50.0),
          (r"waist_yaw", 88.0), (r"shoulder|elbow|wrist", 25.0)]


def effort(name):
    for pat, e in EFFORT:
        if re.search(pat, name):
            return e
    raise KeyError(name)


def rpy(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def f(v):
    return " ".join("%.9g" % x for x in v)


out = ['<?xml version="1.0"?>', '<robot name="r1">']
for b in range(1, m.nbody):
    name = m.body(b).name
    out.append('  <link name="%s">' % name)
    out.append("    <inertial>")
    out.append('      <origin xyz="%s" rpy="%s"/>' % (f(m.body_ipos[b]), f(rpy(m.body_iquat[b]))))
    out.append('      <mass value="%.9g"/>' % m.body_mass[b])
    i = m.body_inertia[b]
    out.append('      <inertia ixx="%.9g" iyy="%.9g" izz="%.9g" ixy="0" ixz="0" iyz="0"/>' % (i[0], i[1], i[2]))
    out.append("    </inertial>")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_contype[g] == 0:
            continue
        gt = int(m.geom_type[g])
        org = '<origin xyz="%s" rpy="%s"/>' % (f(m.geom_pos[g]), f(rpy(m.geom_quat[g])))
        if gt == 3:
            geo = '<cylinder radius="%.9g" length="%.9g"/>' % (m.geom_size[g][0], 2 * m.geom_size[g][1])
        elif gt == 2:
            geo = '<sphere radius="%.9g"/>' % m.geom_size[g][0]
        else:
            raise ValueError(gt)
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        out.append('    <collision name="%s">%s<geometry>%s</geometry></collision>' % (gname, org, geo))
    out.append("  </link>")
for b in range(2, m.nbody):
    name = m.body(b).name
    assert m.body_jntnum[b] == 1
    j = m.body_jntadr[b]
    assert m.jnt_type[j] == 3 and np.allclose(m.jnt_pos[j], 0), name
    p = m.body(m.body_parentid[b]).name
    jn = m.joint(j).name
    lo, hi = m.jnt_range[j]
    out.append('  <joint name="%s" type="revolute">' % jn)
    out.append('    <parent link="%s"/><child link="%s"/>' % (p, name))
    out.append('    <origin xyz="%s" rpy="%s"/>' % (f(m.body_pos[b]), f(rpy(m.body_quat[b]))))
    out.append('    <axis xyz="%s"/>' % f(m.jnt_axis[j]))
    out.append('    <limit lower="%.9g" upper="%.9g" effort="%s" velocity="%s"/>' % (lo, hi, effort(jn), VEL))
    out.append("  </joint>")
out.append("</robot>")
open(OUT, "w").write("\n".join(out))
print("wrote", OUT, "links", m.nbody - 1, "joints", m.nbody - 2)
