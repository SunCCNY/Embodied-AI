# Week 28 package — imitation learning (BeyondMimic via mjlab, Unitree R1)

Everything produced this week: the presentation, the code written against the `mjlab` framework,
and the training artifacts referenced in both. This is a **transfer package**, not a runnable repo
by itself — the code here is the *diff* on top of an existing clone of
[mjlab](https://github.com/mujocolab/mjlab); see "How to re-integrate the code" below.

## presentation/

- `embodied ai week 28.pptx` — the deck (30 slides).
- `week28_box_by_box.html` — speaker's companion, every text box with what it says and what it
  means. Open directly in a browser.
- `build_ppt_week28.py` — the script that generates the .pptx from `week28_shots/` and
  `week28_media/`. Requires `python-pptx` and `Pillow`. Run with the paths in the script's `HERE`
  pointed at wherever you place this folder.
- `week28_shots/` — plots (`training_curves.png`), video poster frames, and `captures/` (real,
  unedited terminal output the deck's terminal panels are built from — nothing in them is typed in
  or invented).
- `week28_media/` — the three result videos (`r1_tennis_tracking.mp4`, `r1_tennis_sim2sim.mp4`,
  `r1_dancer_tracking.mp4`).

## code/

All of it is new code written this week, laid out here by where it lives inside the `mjlab` repo
(or, for the GMR converter, a separate small repo):

- `mjlab_r1_integration/asset_zoo_unitree_r1/` — real code, not a config file. `r1_constants.py`
  defines R1 as a robot inside mjlab (actuator PD gains/effort limits sourced from this project's
  own validated `r1_stand_sim2sim.py` deployment values and the official Unitree `R1_C++.xml`
  ctrlrange, not guessed). `xmls/` is the GMR-clean 24-joint R1 MJCF + meshes it points at.
  Belongs at `mjlab/src/mjlab/asset_zoo/robots/unitree_r1/` in a real mjlab clone.
- `mjlab_r1_integration/tasks_tracking_config_r1/` — `env_cfgs.py` + `rl_cfg.py`, the R1 tracking
  task config, registered as `Mjlab-Tracking-Flat-Unitree-R1`. Belongs at
  `mjlab/src/mjlab/tasks/tracking/config/r1/`.
- `mjlab_r1_integration/csv_to_npz.py` — edited (not new): added a `--robot {g1,r1}` flag so the
  motion converter isn't hardcoded to G1. Replaces
  `mjlab/src/mjlab/scripts/csv_to_npz.py`.
- `gmr_to_mjlab/gmr_pkl_to_mjlab_csv.py` — converts GMR's retargeting output pickle
  (root_pos, root_rot wxyz, dof_pos) into the CSV `csv_to_npz.py` expects (reorders the quaternion
  to xyzw). Standalone script, run from wherever GMR's output lives.
- `sim2sim/` — the independent cross-engine verification harness, all standalone scripts run with
  `uv run python <script>.py` from an mjlab checkout root:
  - `sim2sim_r1_tennis.py` — plain MuJoCo (CPU, `mj_step`) + onnxruntime harness. Reads the
    exported ONNX's embedded metadata (joint order, PD gains, effort limits, action scale, default
    pose) directly, so it needs no separately-maintained config. Docstring at the top documents
    all four real bugs found and fixed during development.
  - `sim2sim_r1_tennis_cuda_policy.py` — same harness, but with ONNX/CPU inference swapped for a
    standalone `PolicyNet` loading the raw `actor_state_dict` and running on CUDA. Built
    specifically to test (and disconfirm) an inference-backend-mismatch hypothesis.
  - `capture_native_rollout.py` — runs the trained policy inside mjlab's own native environment
    and logs the real observation/action/state at every step, for ground-truth verification.
  - `replay_native_actions.py` — open-loop replay of a captured native action sequence through
    plain MuJoCo; used to prove the two physics engines agree given identical actions.

## training_artifacts/

One folder per training run, each with: the exported ONNX policy (self-describing — everything
needed to run it is embedded as metadata), the raw TensorBoard event file, and the final `.pt`
checkpoint (needed by `sim2sim_r1_tennis_cuda_policy.py`, which loads the raw state dict rather
than the ONNX graph). Intermediate checkpoints were left out — only the final one from each run is
included.

- `r1_tennis_run1/` — the original tennis run, 3,000 iterations, `model_2999.pt`. Native eval:
  success_rate 0.9688, mpkpe 0.2991.
- `r1_tennis_run1_continued/` — resumed from the above out to 12,998 iterations, `model_12998.pt`.
  Native eval: success_rate 0.9844, mpkpe 0.1588 — but sim2sim error barely moved, which is the key
  finding that ruled out undertraining as the cause of the sim2sim divergence.
- `r1_dancer_run1/` — the second-motion (dance) run, 3,000 iterations, `model_2999.pt`. Native
  eval: success_rate 1.0000, mpkpe 0.2195 — better than tennis on identical training budget.

## How to re-integrate the code

This package assumes the recipient already has (or will clone) `mjlab`
(https://github.com/mujocolab/mjlab) and GMR (for producing new retargeted motions). To restore
working code:

1. Clone `mjlab`, then copy `code/mjlab_r1_integration/asset_zoo_unitree_r1/` to
   `mjlab/src/mjlab/asset_zoo/robots/unitree_r1/` and
   `code/mjlab_r1_integration/tasks_tracking_config_r1/` to
   `mjlab/src/mjlab/tasks/tracking/config/r1/`.
2. Replace `mjlab/src/mjlab/scripts/csv_to_npz.py` with
   `code/mjlab_r1_integration/csv_to_npz.py`.
3. Copy the four `code/sim2sim/*.py` scripts to the mjlab repo root (they assume they're run from
   there, via `uv run python <script>.py`).
4. `code/gmr_to_mjlab/gmr_pkl_to_mjlab_csv.py` can live anywhere with access to GMR's `.pkl`
   output; it has no mjlab dependency itself.
5. To reproduce training without re-running from scratch, either resume from a `training_artifacts/
   */model_*.pt` checkpoint (rsl_rl's `--agent.resume True --agent.load-run <dir>`), or run the
   exported `.onnx` directly against `sim2sim_r1_tennis.py`.

Full narrative context (why each decision was made, what broke, what was tested and ruled out) is
in the `embodied-ai-il-mjlab` memory file and in the box-by-box companion doc above — the deck and
that doc are the most complete record of the reasoning, not just the artifacts.
