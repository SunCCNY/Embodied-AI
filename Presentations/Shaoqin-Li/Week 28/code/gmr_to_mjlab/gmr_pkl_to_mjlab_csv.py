"""Convert a GMR motion pickle (root_pos, root_rot wxyz, dof_pos) into the flat
CSV format mjlab's `csv_to_npz.py` expects: one row per frame,
[root_pos(3), root_quat_xyzw(4), dof_pos(N)], comma-delimited, no header.

GMR's dof_pos is already in the target robot's native MJCF joint order
(it's qpos[7:] straight from the retargeted qpos vector), which for R1
matches the 24-joint order registered for "r1" in csv_to_npz.py.

Usage:
  python gmr_pkl_to_mjlab_csv.py --pkl out/tennis_workstation.pkl --csv out/tennis_workstation.csv
"""

import argparse
import pickle

import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--pkl", required=True, help="GMR motion pickle path")
  ap.add_argument("--csv", required=True, help="Output CSV path")
  args = ap.parse_args()

  with open(args.pkl, "rb") as f:
    data = pickle.load(f)

  root_pos = data["root_pos"]  # (T, 3)
  root_rot_wxyz = data["root_rot"]  # (T, 4)
  dof_pos = data["dof_pos"]  # (T, N)

  # mjlab's csv_to_npz.py reads columns 3:7 as xyzw then reorders to wxyz
  # itself, so we must hand it xyzw here.
  root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

  rows = np.concatenate([root_pos, root_rot_xyzw, dof_pos], axis=1)
  np.savetxt(args.csv, rows, delimiter=",")

  print(f"Wrote {rows.shape[0]} frames x {rows.shape[1]} cols to {args.csv}")
  print(f"  (fps in source pkl: {data.get('fps')}, robot: {data.get('robot')})")


if __name__ == "__main__":
  main()
