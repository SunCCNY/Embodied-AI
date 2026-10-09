"""Wrap mjlab's r1 motion npz (MuJoCo body/joint order) with name arrays + fps so Isaac Lab can reorder by name."""
import os
import sys

import mujoco
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
XML = os.environ.get("R1_XML", "src/mjlab/asset_zoo/robots/unitree_r1/xmls/r1.xml")  # run from the mjlab clone, or set R1_XML
m = mujoco.MjModel.from_xml_path(XML)
body_names = [m.body(b).name for b in range(1, m.nbody)]
joint_names = [m.joint(j).name for j in range(1, m.njnt)]
d = dict(np.load(src))
print({k: v.shape for k, v in d.items()})
assert d["joint_pos"].shape[1] == len(joint_names) and d["body_pos_w"].shape[1] == len(body_names)
d["fps"] = np.array([50])
d["joint_names"] = np.array(joint_names)
d["body_names"] = np.array(body_names)
np.savez(dst, **d)
print("saved", dst, "T =", d["joint_pos"].shape[0])
