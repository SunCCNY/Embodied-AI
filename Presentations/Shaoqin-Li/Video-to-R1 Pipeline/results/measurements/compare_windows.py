"""Does the frame range you retarget change the motion you get?

An earlier write-up says our R1 retarget floats 3.7 cm across all 4,249 frames
and never contacts the ground.  A later one says it penetrates the floor by
1.15 cm on 64% of frames.  Both were produced by the same pipeline.  The only
difference is that the second retargeted frames 600-1800 as their own clip.

This compares the SAME frames from the two runs.  If the pipeline were
window-independent the two would be identical.

    python compare_windows.py --full out/full_capture.pkl --window out/mine.pkl
"""
import argparse
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="out/full_capture.pkl",
                    help="the whole capture, retargeted in one run")
    ap.add_argument("--window", default="out/mine.pkl",
                    help="a sub-range retargeted as its own run")
    ap.add_argument("--start", type=int, default=600)
    a = ap.parse_args()

    full = pickle.load(open(a.full, "rb"))
    win = pickle.load(open(a.window, "rb"))
    fz = np.asarray(full["root_pos"])[:, 2]
    wz = np.asarray(win["root_pos"])[:, 2]
    n = len(wz)
    seg = fz[a.start:a.start + n]

    print("full capture   : %5d frames, root z %.4f .. %.4f" % (len(fz), fz.min(), fz.max()))
    print("windowed run   : %5d frames, root z %.4f .. %.4f" % (n, wz.min(), wz.max()))
    print("same frames in the full run  : root z %.4f .. %.4f" % (seg.min(), seg.max()))

    d = seg - wz
    print("\nroot height difference (full - windowed):")
    print("   mean %+.4f m   sd %.6f m   min %+.4f   max %+.4f"
          % (d.mean(), d.std(), d.min(), d.max()))

    df = np.abs(np.asarray(full["dof_pos"])[a.start:a.start + n] - np.asarray(win["dof_pos"]))
    print("\njoint angles, same frames: max |difference| = %.3e rad  (mean %.3e)"
          % (df.max(), df.mean()))

    print("""
A near-constant vertical offset with a small sd points at the GROUND OFFSET,
which GMR computes from the clip it is given.  A joint difference of this size
points at the IK itself: mink is warm-started from the previous frame, so
arriving at frame %d having tracked from frame 0 lands in a different local
solution than starting there cold.  Separating the two is the open experiment:
re-run with the ground offset disabled and see how much of the gap survives.""" % a.start)


if __name__ == "__main__":
    main()
