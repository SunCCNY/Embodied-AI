# Setup — environments for each stage

Tested machine: Ubuntu 22.04.5 under WSL2, NVIDIA RTX 5080 (Blackwell, sm_120, 16 GB), 47 GB RAM.
Stage 3b (Isaac Lab) ran **natively on Windows 11** on the same machine — see the note in that section.

Each stage has its own environment. You only need the environments for the stages you want to run.
If you only want to *check the results*, Stage 4 alone is enough (it uses the policies shipped in `policies/`).

| Stage | Environment | Upstream repo, pinned commit |
|---|---|---|
| 1 video → SMPL-X | conda `gvhmr`, Python 3.10, torch 2.7.0+cu128 | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) `ee960bb` |
| 2 SMPL-X → R1 | conda `gmr`, Python 3.10 | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) `bb1bbe4` |
| 3a train (MuJoCo-Warp) | uv venv inside the mjlab clone, Python 3.13 | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) `c2e1e06` |
| 3b train (Isaac Lab / PhysX) | Isaac Lab venv, Python 3.11, Isaac Sim 5.1.0 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) `b0542fe2d` + [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) `cd65172` |
| 4 sim2sim | the same mjlab venv as 3a | — |

Below, `$PIPE` means the path to this folder:

```bash
export PIPE="/path/to/Video-to-R1 Pipeline"
```

---

## Stage 1 — GVHMR (video → SMPL-X)

```bash
conda create -n gvhmr python=3.10 -y && conda activate gvhmr
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/zju3dv/GVHMR.git ~/GVHMR && cd ~/GVHMR && git checkout ee960bb
```

The problems below are why the stock install does not work on a Blackwell GPU. Each one was hit and fixed on the tested machine.

1. **Keep torch pinned.** Install everything else with `-c $PIPE/1_video_to_smplx/constraints.txt`.
   Without it, `ultralytics`/`lightning` silently upgrade torch and break the CUDA build.
2. **GVHMR's `requirements.txt` hard-pins torch 2.3.0+cu121 and an old pytorch3d wheel.** Drop those lines
   and install the rest with the constraints file. Also install `timm` and `yacs`; `yacs` is an
   undeclared dependency of HMR2, and without it the demo crashes on start. Then run `pip install -e . --no-deps`.
3. **pytorch3d has no prebuilt wheel for torch 2.7.0+cu128 / Python 3.10, so build it from source:**
   ```bash
   conda install -y -c nvidia/label/cuda-12.8.0 cuda-nvcc=12.8.61    # pip's nvidia-cuda-nvcc ships no nvcc
   git clone https://github.com/facebookresearch/pytorch3d.git ~/src/pytorch3d && cd ~/src/pytorch3d && git checkout v0.7.9
   E=$CONDA_PREFIX; NV=$E/lib/python3.10/site-packages/nvidia
   export CPATH=$(ls -d $NV/*/include | grep -v /cu13/ | tr "\n" ":")$E/targets/x86_64-linux/include:$E/include
   # ^ torch's pip CUDA headers are split per package; conda's nvcc keeps its own (nv/target) under targets/
   export CUDA_HOME=$E FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0"
   pip install --no-build-isolation --no-deps -c $PIPE/1_video_to_smplx/constraints.txt .
   ```
   Install it as a normal package (no `-e`) from a folder that will not be deleted. An editable install from `/tmp` breaks once WSL clears `/tmp`.
4. **Checkpoints (about 5.3 GB):** dpvo, gvhmr, hmr2, vitpose and yolo go in `~/GVHMR/inputs/checkpoints/`, following GVHMR's
   `docs/INSTALL.md`. The Hugging Face mirror `camenduru/GVHMR` has all five.
5. **SMPL-X body model (not included here; its license forbids redistribution):** register at
   https://smpl-x.is.tue.mpg.de and put `SMPLX_NEUTRAL.npz` in `~/GVHMR/inputs/checkpoints/body_models/smplx/`.

6. **Moving camera (no `-s`) needs a patch** with pycolmap ≥ 3.11 and torch ≥ 2.6:
   `git apply $PIPE/1_video_to_smplx/gvhmr_ee960bb_pycolmap4.patch`. It fixes two crashes. pycolmap flags near-static
   frame pairs as a "watermark" and returns no camera pose (`'NoneType' object has no attribute 'matrix'`), and torch
   refuses to reload GVHMR's own camera-trajectory file (`Weights only load failed`). The fixed-camera path is unaffected.

Run it on GVHMR's own tennis clip (static camera):

```bash
cd ~/GVHMR && python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
# -> outputs/demo/tennis/hmr4d_results.pt
```

The demo then tries to render an overlay video, and that step needs the SMPL (not SMPL-X) `.pkl`. If you don't have it,
the demo stops there, but `hmr4d_results.pt` has already been written and it is all Stage 2 needs.

## Stage 2 — GMR (SMPL-X → R1 joint trajectories)

```bash
conda create -n gmr python=3.10 -y && conda activate gmr
git clone https://github.com/YanjieZe/GMR.git ~/GMR && cd ~/GMR && git checkout bb1bbe4 && pip install -e .
# GMR has no R1. This adds the R1 model, its 43 meshes, its IK configs and its params.py entries:
python "$PIPE/2_smplx_to_r1/install_r1_into_gmr.py" --gmr ~/GMR \
    --meshes "$PIPE/3_motion_to_policy/mjlab_overlay/src/mjlab/asset_zoo/robots/unitree_r1/xmls/assets"
# expected last line:  OK -- the R1 is installed in GMR
```

**Give the real frame rate.** GMR's GVHMR loader assumes 30 fps and never resamples a slower video, so a 25 fps clip
would play 1.2× too fast on the robot. Pass the video's frame rate to both steps:
`gvhmr_to_robot_headless.py ... --src_fps 25` and later `csv_to_npz.py ... --input-fps 25`.

MuJoCo offscreen rendering under WSL2 (only needed for the optional render scripts): the default GL backend crashes
with `D3D12: Removing Device`. Use `MUJOCO_GL=osmesa`. If `libOSMesa` is missing and you have no sudo, you can get it without root:
`apt-get download libosmesa6 libglapi-mesa`, run `dpkg-deb -x <deb> ~/local_osmesa` on each, then add the extracted `usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH`.

## Stage 3a / Stage 4 — mjlab (training on MuJoCo-Warp, and the sim2sim harness)

```bash
git clone https://github.com/mujocolab/mjlab.git ~/mjlab && cd ~/mjlab && git checkout c2e1e06
git apply "$PIPE/3_motion_to_policy/mjlab_overlay/mjlab_c2e1e06.patch"     # registers the R1, adds --robot r1 to csv_to_npz
cp -r "$PIPE/3_motion_to_policy/mjlab_overlay/src/." src/                   # R1 robot + R1 tracking tasks
cp "$PIPE"/4_sim2sim/*.py "$PIPE"/4_sim2sim/isaac_info.json .                # sim2sim harness (run from the mjlab root)
uv sync
```

`git status` should then list exactly 2 modified files (`asset_zoo/robots/__init__.py`, `scripts/csv_to_npz.py`) and
2 new folders (`asset_zoo/robots/unitree_r1/`, `tasks/tracking/config/r1/`).

Weights & Biases is optional. Put `WANDB_MODE=disabled` in front of any command and use `--agent.logger tensorboard` when training.

## Stage 3b — Isaac Lab (training on PhysX)

On this RTX 5080, GPU PhysX hangs inside WSL2, even on Isaac Lab's own Cartpole demo. So Isaac Lab was installed and run
**natively on Windows** (`C:\dev\IsaacLab`, its own venv, Isaac Sim 5.1.0 via pip). On native Linux the same steps should work
unchanged, but that has not been tested.

1. Install Isaac Lab `b0542fe2d` + Isaac Sim 5.1.0 following Isaac Lab's pip install guide.
2. `git clone https://github.com/HybridRobotics/whole_body_tracking.git && cd whole_body_tracking && git checkout cd65172`,
   then `git apply "$PIPE/3_motion_to_policy/isaac_overlay/whole_body_tracking_cd65172.patch"` (lets the motion loader reorder joints and bodies by name)
   and copy the overlay files in:
   - `isaac_overlay/whole_body_tracking/robots/r1.py` → `source/whole_body_tracking/whole_body_tracking/robots/`
   - `isaac_overlay/whole_body_tracking/tasks/tracking/config/r1/` → `.../tasks/tracking/config/r1/`
   - `isaac_overlay/whole_body_tracking/tasks/tracking/mdp/actions.py` → `.../tasks/tracking/mdp/`
   - `isaac_overlay/r1_isaac/r1.urdf` → `.../whole_body_tracking/assets/r1/r1.urdf`
   - then `pip install -e source/whole_body_tracking`
3. The training scripts are in `isaac_overlay/r1_isaac/` (see the README for the commands).
