"""Convert OMOMO sequences to the SMPL-X .npz layout GMR's loader expects.

GMR ships `scripts/convert_omomo_to_smplx.py`, but it carries the author's own
absolute paths (`/home/yanjieze/...`) and converts the entire dataset. This takes
the raw `.p` file as an argument and writes only as many sequences as asked for,
which is all that is needed to test that the smplx -> R1 config runs.

`utils/smpl.py:load_smplx_file` calls `np.load(..., allow_pickle=True)` and then
indexes gender / betas / root_orient / pose_body / trans, so those five keys have to
be present. numpy falls back to `pickle.load` for a file that is not in .npy format,
which is why upstream's raw `pickle.dump` output works -- but writing a real .npz is
less surprising, so that is what this does.

Usage:
    python convert_omomo_local.py --pkl <train_or_test_...joints24.p> \
        --outdir ~/GMR/assets/omomo/smplx --limit 3
"""
import argparse
import os

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="OMOMO *_diffusion_manip_seq_joints24.p")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=3, help="how many sequences to write")
    ap.add_argument("--fps", type=int, default=30, help="OMOMO is captured at 30 fps")
    args = ap.parse_args()

    print(f"loading {args.pkl} ...")
    data = joblib.load(args.pkl)
    print(f"  {len(data)} sequences")

    os.makedirs(args.outdir, exist_ok=True)
    written = 0
    for key in list(data.keys()):
        if written >= args.limit:
            break
        seq = data[key]
        missing = [k for k in ("pose_body", "betas", "root_orient", "trans") if k not in seq]
        if missing:
            print(f"  skip {key}: missing {missing}")
            continue

        name = str(seq.get("seq_name", key))
        n = seq["pose_body"].shape[0]
        # upstream pads pose_body (N,63) out to a full (N,165) `poses` array
        poses = np.concatenate([seq["pose_body"], np.zeros((n, 102))], axis=1)

        out = os.path.join(args.outdir, f"{name}.npz")
        np.savez(out,
                 gender=str(seq.get("gender", "neutral")),
                 betas=np.asarray(seq["betas"]),
                 root_orient=np.asarray(seq["root_orient"]),
                 pose_body=np.asarray(seq["pose_body"]),
                 trans=np.asarray(seq["trans"]),
                 poses=poses,
                 mocap_frame_rate=np.array(args.fps))
        print(f"  wrote {name}.npz  {n} frames  betas{np.asarray(seq['betas']).shape}")
        written += 1

    print(f"{written} sequence(s) -> {args.outdir}")


if __name__ == "__main__":
    main()
