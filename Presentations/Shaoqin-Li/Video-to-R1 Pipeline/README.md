# Video-to-R1 Pipeline

**From an ordinary video of a person to a Unitree R1 humanoid policy that reproduces the motion and survives a
second physics engine.**
Shaoqin Li, Embodied AI group (Prof. Yi Sun), The City College of New York.

```
 video (.mp4)
   │  Stage 1  GVHMR ............ human pose + shape per frame, world frame         -> hmr4d_results.pt  (SMPL-X)
   ▼
 SMPL-X body motion
   │  Stage 2  GMR + R1 configs .. inverse kinematics onto the R1's 24 joints        -> *.pkl / *.csv     (30 fps)
   ▼
 R1 joint trajectory
   │  Stage 3  BeyondMimic-style motion tracking, PPO, 4,096 parallel robots
   │           3a mjlab (MuJoCo-Warp)   or   3b Isaac Lab (PhysX)                  -> policy .onnx
   ▼
 R1 tracking policy
   │  Stage 4  sim2sim: plain CPU MuJoCo, training randomization + stress tests     -> survival / joint error
   ▼
 survives?  which perturbations break it?  (the sim2real to-do list)
```

The whole chain was run on two motions: a tennis serve (GVHMR's own example clip, 10.4 s) and a dance
("Woman Dancing" by Antoni Shkraba, Pexels, 26.5 s at 25 fps; see the frame-rate note below). This folder has everything needed to re-run each stage,
the intermediate data of each stage, the trained policies, and the results.

**Status: work in progress.** Everything here runs in simulation, on two third-party, static-camera clips.
Not done yet: running my own phone video through the pipeline, and sim2real on the physical R1.
The paper in `paper/draft.md` is a first draft for the same reason.

**New 2026-10-14:** three unseen stock clips, two with a moving camera, went through the whole pipeline. See
[results/new_clips_2026-10-14/README.md](results/new_clips_2026-10-14/README.md) for side-by-side videos and sim2sim results.

**Writing the paper or slides?** Start with [EVIDENCE.md](EVIDENCE.md): every figure, video and number, mapped to the claim it supports.

Weekly decks with the step-by-step story: `../Week 26` (GMR), `../Week 27`, `../Week 28` (imitation learning + sim2sim),
`../Week 29` (Isaac Lab dance).

---

## Try it in 10 minutes (Stage 4 only — no training, runs on the CPU)

Set up the mjlab clone as described in [SETUP.md](SETUP.md) (Stage 3a / 4 section). Then, from the mjlab root:

```bash
PIPE="/path/to/Video-to-R1 Pipeline"
ONNX="$PIPE/policies/mjlab_tennis_delay/2026-10-01_17-41-06_r1_tennis_delay_ft.onnx" \
MOTION="$PIPE/data/r1_motions/tennis_motion.npz" \
uv run python sim2sim_v5_dr.py sweep nominal,train_DR+latency2
```

Output on the tested machine (re-run from a fresh clone on 2026-10-08):

```
scenario                 seeds  survived  mean err (rad)  median steps  failure reasons
nominal                      1    1/1              0.597           518  {}
train_DR+latency2           20   15/20             0.800           518  {'ee_z': 5}
```

To watch it: `... sim2sim_v5_dr.py video train_DR 1000 out.mp4` writes a video. `... sim2sim_v5_dr.py viewer train_DR 1000`
opens a live MuJoCo window (needs a display; under WSL2 use `LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw`).
In the videos, grey is the policy-controlled robot and the green ghost is the reference motion.

The scenario list is at the top of `sim2sim_v5_dr.py`. Run `sweep` with no argument for all 16.

---

## Running each stage

All commands assume the environments from [SETUP.md](SETUP.md).

### Stage 1 — video → SMPL-X (GVHMR)

```bash
conda activate gvhmr && cd ~/GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s       # -s: static camera
```

Output: `outputs/demo/tennis/hmr4d_results.pt`. The outputs used for both motions are in `data/gvhmr_outputs/`, so you
can skip this stage.

### Stage 2 — SMPL-X → R1 trajectory (GMR)

```bash
conda activate gmr && cd ~/GMR
python "$PIPE/2_smplx_to_r1/gvhmr_to_robot_headless.py" \
    --gvhmr_pred_file "$PIPE/data/gvhmr_outputs/tennis_hmr4d_results.pt" \
    --body_models ~/GVHMR/inputs/checkpoints/body_models --src_fps 30 --save_path tennis_r1.pkl   # --src_fps = the video's frame rate
python "$PIPE/2_smplx_to_r1/exp4_check_motion.py" tennis_r1.pkl          # plausibility report
python "$PIPE/3_motion_to_policy/gmr_pkl_to_mjlab_csv.py" --pkl tennis_r1.pkl --csv tennis_r1.csv
```

The R1 configs for all eight GMR input formats are in `2_smplx_to_r1/support/`; `make_r1_configs.py` generates them and
`validate_r1_configs.py` checks them. Only `smplx_to_r1.json` is used by this pipeline.
`Making a Robot Copy a Human.md` explains retargeting from the beginning.

### Stage 3a — train in mjlab (MuJoCo-Warp)

```bash
cd ~/mjlab
# trajectory CSV (30 fps) -> reference motion NPZ (50 fps, full-body kinematics). Writes /tmp/motion.npz
WANDB_MODE=disabled uv run src/mjlab/scripts/csv_to_npz.py --input-file tennis_r1.csv --output-name r1_tennis --robot r1
# train
WANDB_MODE=disabled uv run train Mjlab-Tracking-Flat-Unitree-R1 \
    --env.commands.motion.motion-file /tmp/motion.npz \
    --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard --video False
```

The tasks this folder adds:

| Task id | What it is |
|---|---|
| `Mjlab-Tracking-Flat-Unitree-R1` | the R1 tracking task (BeyondMimic recipe, R1 PD gains and torque limits) |
| `Mjlab-Tracking-Flat-Unitree-R1-Delay` | same, plus a random 0–50 ms command delay (0–10 physics steps of 5 ms) |
| `Mjlab-Tracking-Flat-Unitree-R1-No-State-Estimation` | the policy is not given base position or velocity |

`mjlab_overlay/run_delay_ft.sh` is the exact command that fine-tuned the tennis policy with delay.

### Stage 3b — train in Isaac Lab (PhysX)

Run from `isaac_overlay/r1_isaac/` with the Isaac Lab Python:

```bash
python make_motion.py tennis_or_dance_motion.npz motion_named.npz      # adds joint/body names (run from mjlab root, or set R1_XML)
python train_r1.py --headless --task Tracking-Flat-R1-v0 --num_envs 4096 --max_iterations 3000 --motion_file motion_named.npz
python export_onnx.py logs/rsl_rl/r1_flat_isaac/<run>/model_2999.pt policy.onnx
python isaac_info.py --headless --motion_file motion_named.npz          # joint order + first observation -> isaac_info.json
```

The delay version is `--task Tracking-Flat-R1-Delay-v0`. PhysX lists the joints in a different order than MuJoCo, so
`sim2sim_isaac.py` matches every joint by name using `isaac_info.json`.

### Stage 4 — sim2sim

`4_sim2sim/` is copied into the mjlab root (see SETUP). The files:

| File | What it does |
|---|---|
| `standalone_r1_model.py` | rebuilds the exact training physics model from `r1.xml` (floor, contacts, 5 ms timestep, solver) without importing mjlab's training code |
| `sim2sim_r1_tennis_v4_standalone.py` | policy constants (PD gains, torque limits, action scale) and observation building |
| `sim2sim_v5_dr.py` | the sweep: training randomization + 16 stress scenarios, termination rules copied from training; `sweep` / `video` / `viewer` |
| `sim2sim_isaac.py` | the same harness for Isaac-trained policies (`validate` checks the harness observation against Isaac's own) |

---

## Results

### Stages 1–2 (RTX 5080)

| | Tennis (10.4 s, 312 frames) | Dance (26.5 s video, 662 frames at 25 fps) |
|---|---|---|
| GVHMR preprocess + predict | 39.8 s + 0.85 s | 99.9 s + 1.03 s |
| GMR inverse kinematics | 332–449 frames/s | 305 frames/s |
| Joint-limit violations | 0 | — |

For tennis, the plausibility check gave: peak joint speed 22.5 rad/s, above a 20 rad/s servo budget on 4 frames;
9% of frames in ground contact; 18.5 cm pelvis height range. GVHMR's vertical drift (about 15 cm over the clip) is a
Stage 1 property. `gvhmr_to_robot_headless.py` corrects it with a linear ground pass. Details: `results/measurements/`.

### Stage 3 — training

| Policy | Engine | Training | Success in its own engine |
|---|---|---|---|
| tennis | mjlab | 3,000 iterations (59 min), continued to 12,998 (+3 h 19 min) | 96.9% → 98.4% of 64 episodes (mjlab `evaluate`) |
| tennis + delay | mjlab | fine-tuned from 12,998, 3,000 more (55 min) | — |
| dance | mjlab | 3,000 iterations | 100% of 64 episodes (mjlab `evaluate`) |
| dance | Isaac Lab | 3,000 iterations (about 100 min) | — |
| dance + delay | Isaac Lab | 3,000 iterations | — |

All runs used 4,096 parallel robots. Checkpoints, ONNX exports and the exact `env.yaml` / `agent.yaml` are in `policies/`.

### Stage 4 — sim2sim survival (out of N seeds)

| Scenario | tennis | tennis + delay | dance (mjlab) | dance (Isaac) | dance (Isaac) + delay |
|---|---|---|---|---|---|
| nominal | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| training randomization | 30/30 | 30/30 | 29/30 | 29/30 | **17/30** |
| + 20 ms command delay | 19/20 | 20/20 | | | 18/20 |
| + 40 ms command delay | **2/20** | **15/20** | | **0/20** | **15/20** |
| + 60 ms command delay | | 8/20 | | | 11/20 |
| + 3× pushes | 11/20 | 11/20 | | | |
| + foot friction 0.15 | 5/20 | 10/20 | | | |
| + PD gain −20% | 20/20 | 19/20 | | | |
| + 2 kg payload | 20/20 | 20/20 | | | |
| no base linear velocity | 5/20 | 4/20 | | | |
| no base position | 20/20 | | | | |
| joint error, nominal (rad) ¹ | 0.577 | 0.597 | 2.24 | 2.01 | 2.19 |

¹ The L2 norm of the 24-joint error vector, averaged over time; this is the same quantity as mjlab's `error_joint_pos` metric.
Divide by √24 ≈ 4.9 for the per-joint RMS: 0.12 rad for tennis, 0.46 rad for dance (mjlab).

Raw numbers: `results/sweeps/*.json` and `results/terminal_captures/`. Videos: `results/videos/`.

### What the results say

1. **The chain works end to end.** Video-derived motion trains into a policy that completes the whole clip in a
   second simulator, under the randomization it was trained with, for both motions and both training engines.
2. **The first sim2sim "failure" was a bug in the test harness, not a gap between physics engines.** The early
   harness loaded the raw `r1.xml`, which has **no floor**, and stepped physics at 125 Hz instead of training's 200 Hz
   (5 ms × 4 = 50 Hz control). Removing each difference one at a time: without the floor the error is 6.97 rad, with the
   wrong timestep 1.72 rad. Contact dimension, solver iterations, friction and contact priority each change it by less than 0.006 rad
   (`results/terminal_captures/floor_timestep_ablation.txt`). Once the model matched training exactly, the error dropped to 0.577 rad.
3. **Command delay is the main weakness, and training with delay fixes most of it.** The same pattern shows up in both engines: 40 ms
   took survival from 2/20 to 15/20 (mjlab, tennis) and from 0/20 to 15/20 (Isaac, dance).
   A seed study (3 training seeds per arm, `results/sweeps/seed_study/`) confirms the gain comes from the delay, not from
   the extra 3,000 iterations: at 40 ms, 2–8/20 without delay vs 15–20/20 with it; at 60 ms, 0/20 vs 8–14/20.
4. **The policy depends on base linear velocity.** Removing it leaves 4–5/20 survivors. A real R1 does not measure this
   directly, so sim2real needs a velocity estimator or a policy trained without it. Removing base position is harmless (20/20).

### Known limitations and open questions

- **CORRECTED 2026-10-14, frame rate:** GMR's GVHMR loader assumes every video is 30 fps and never resamples a slower one.
  The dance video is 25 fps, so its robot reference runs **1.2× too fast**: the whole 26.5 s dance in 22.1 s. Every dance
  result here (training, sweeps, videos, shoulder analysis) is for that sped-up reference. Tennis (29.8 fps) is off by
  only 0.6%. `gvhmr_to_robot_headless.py --src_fps <video fps>` now keeps the real timing (used for the 2026-10-14 clips);
  pass the same value to `csv_to_npz.py --input-fps`.

- **The Isaac delay-trained policy got worse with no delay** (29/30 → 17/30) and has a higher joint error. Not yet explained.
  It may have tuned itself toward mid-range delay; a longer run would show whether it recovers.
- The harness holds the delay constant for the whole episode. Training re-draws it every second and starts each episode at zero,
  so the 40–60 ms tests are stricter than anything it trained on.
- Dance joint error (2.0–2.2) is much higher than tennis (0.6). This is **not a harness artifact**: mjlab's own training metric ends at 2.04
  for dance. It comes almost entirely from the left shoulder, in the 27% of frames where the reference pins that shoulder at a joint
  limit (left-shoulder RMS error 2.06 rad there vs 0.45 rad elsewhere), while the left hand is placed nearly as well (19 vs 16 cm).
  The policy reaches the pose with a different shoulder configuration. GMR's "0 joint-limit violations" hides this, because IK
  clamps at the limit. The dance policy also had 3,000 iterations against tennis's 13,000.
- One training seed per policy and two motions. Idealized actuators (PD with torque limits, no motor model).
  Nothing has been run on the physical R1 yet.
- Isaac Lab training ran on native Windows (GPU PhysX hangs under WSL2 on this GPU), and the sim2sim harness ran in WSL2.

### What was re-checked for this folder (2026-10-08)

Each step below was re-run from **fresh upstream clones at the pinned commits, using only files from this folder**, and compared
with the outputs that produced the results above:

| Step | Result |
|---|---|
| Stage 1, GVHMR on the tennis clip | identical: max difference 0.0 in every output array (41.9 s + 0.79 s) |
| R1 install into a stock GMR clone | `OK -- the R1 is installed in GMR` |
| Stage 2 retarget of tennis (`hmr4d_results.pt` → `.pkl`) | identical: max difference 0.0 in root position, rotation and all 24 joints |
| `.pkl` → `.csv` | byte-identical |
| `.csv` → `.npz` (mjlab, `--robot r1`) | identical: max difference 0.0 in all 7 arrays |
| Training, both R1 tasks (5-iteration smoke run) | runs, saves checkpoint + ONNX |
| Stage 4 sweep, tennis + delay policy | identical to the recorded results, value for value |
| Isaac harness `validate` + nominal, Isaac dance policy | observation mismatch 0.00000; joint error 2.011 rad, identical |
| Isaac URDF regeneration (`make_urdf.py`) | byte-identical to the one used in training |

Not re-run from scratch: full-length training (each run takes about an hour) and Isaac Lab training.
Training on GPU is not bit-deterministic, so a retrained policy will be close to these results but not identical.

---

## Folder map

```
README.md, SETUP.md
1_video_to_smplx/     constraints.txt (torch pin), measure_gvhmr_drift.py
2_smplx_to_r1/        R1-into-GMR installer, R1 model + 8 IK configs (support/), config generator/validator,
                      headless retarget scripts (GVHMR, SMPL-X, LAFAN1, Xsens), plausibility checker
3_motion_to_policy/   gmr_pkl_to_mjlab_csv.py
  mjlab_overlay/      R1 robot (MJCF + 43 meshes) and tracking tasks for mjlab, patch for mjlab c2e1e06
  isaac_overlay/      R1 for whole_body_tracking (robot, task, delayed action), URDF, Isaac training/export scripts
4_sim2sim/            standalone sim2sim harness, randomization sweep, Isaac harness
data/gvhmr_outputs/   Stage 1 outputs (tennis, dance)
data/r1_motions/      Stage 2 outputs (.pkl, .csv) and Stage 3 reference motions (.npz)
policies/             5 trained policies: ONNX + final checkpoint + env/agent config
results/              sweeps (JSON), measurements, terminal captures, plots, videos, figures (paper figures + video vs R1 frames; see EVIDENCE.md)
paper/                paper draft (in progress)
```

## Not included

| Item | Why | How to get it |
|---|---|---|
| SMPL-X body model | its license does not allow redistribution | register at https://smpl-x.is.tue.mpg.de |
| GVHMR checkpoints (5.3 GB) | size | GVHMR `docs/INSTALL.md`, or the `camenduru/GVHMR` Hugging Face mirror |
| Dance source video | third-party (Pexels) | "Woman Dancing" by Antoni Shkraba on pexels.com |
| Tennis source video | already ships with GVHMR | `GVHMR/docs/example_video/tennis.mp4` |

## Credits

GVHMR (Shen et al., SIGGRAPH Asia 2024) · GMR (Ze et al.) · mjlab (MuJoCo Lab) · BeyondMimic / whole_body_tracking
(HybridRobotics) · Isaac Lab (NVIDIA) · Unitree R1 model and meshes from
[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (BSD-3-Clause). The R1 support for GMR, mjlab and
Isaac Lab, the motion converters and the sim2sim harnesses are this project's own work.
