# R1 dance: train in Isaac Lab (PhysX), transfer to plain MuJoCo

Question: does a BeyondMimic-style tracking policy trained in Isaac Lab (PhysX) run in plain MuJoCo (CPU, single env), a different physics engine? This is a stronger sim2sim test than mjlab (MuJoCo-Warp) to MuJoCo.

## Result (2026-10-01, one 22 s dance clip, 1102 frames at 50 Hz)

Joint error is the L2 norm over 24 joints, averaged over the clip. "Survive" means no termination (anchor height/orientation, end-effector height) over the whole clip.

| Policy | Nominal | Training-style randomization (30 seeds) | Randomization + 40 ms action delay (20 seeds) | Joint error (L2) |
|---|---|---|---|---|
| mjlab (MuJoCo-Warp), baseline | 1/1 | 29/30 | not run | 2.24 - 2.28 |
| Isaac Lab (PhysX), 500 iterations | falls at ~1.7 s | - | - | - |
| Isaac Lab (PhysX), 1500 iterations | 1/1 | 26/30 | 0/20 | 2.48 |
| Isaac Lab (PhysX), final (3000 iterations) | 1/1 | 29/30 | 0/20 | 2.01 |

The Isaac policy was not trained with action delay, so the 40 ms failure is expected and matches what the mjlab policy did before delay randomization. Videos: `videos/` (grey = robot in MuJoCo, green = reference).

## Layout

- `training/`: `train_r1.py` (rsl-rl PPO, no wandb), `export_onnx.py` (checkpoint to ONNX, obs in Isaac joint order), `isaac_info.py` / `isaac_info.json` (ground-truth joint order, defaults, gains and one observation dumped from the live Isaac env).
- `whole_body_tracking_r1/`: the R1 additions to `whole_body_tracking` (robot config, URDF, task registration `Tracking-Flat-R1-v0`, PPO config, and `commands.py` with by-name joint/body reordering of the motion npz).
- `harness/`: `sim2sim_isaac.py` (`validate`, `sweep`, `video`), helper scripts it imports, and the result JSONs.
- `policy/`: final checkpoint, ONNX, motion npz, and the training env/agent yaml.

## How it was run

- Training: 4096 envs, 3000 iterations, about 2 s/iteration (~100 min) on an RTX 5080. GPU PhysX hangs in WSL2 on this card, so it ran natively on Windows (Python 3.11, isaaclab 0.54.4, `tensordict==0.9.1`; 0.14.2 crashes on import).
- Transfer: joints are mapped by name (PhysX order is breadth-first with left/right interleaved; the MuJoCo model is in MJCF order). The harness observation matches Isaac's own 135-number observation to within 1e-5 at frame 0 (`python sim2sim_isaac.py validate`).

## Caveats

- MuJoCo joint friction loss (0.2) has no counterpart in the PhysX setup; the cylinder-to-capsule collision conversion was not verified; ankle linkage rods are not modeled in either engine.
- The randomization scenarios are the ones from the mjlab harness, not Isaac's exact event list.
- One motion clip; the nominal video is a single seed.
