# R1 walk/run policy + command-delay randomization (2026-10-23)

Follows [`../walk_run_eval_2026-10-22/`](../walk_run_eval_2026-10-22/README.md), where the walk/run policy (model_10000) was weakest
at **standing** under command delay (6/10 at 40 ms in the CPU sim2sim). This applies the delay recipe from the L4DC paper to the
locomotion policy and checks it with a matched control.

## What was trained (RTX 5080 workstation, 4096 robots each)

| arm | task | start | iterations | wall time |
|---|---|---|---|---|
| base | `Unitree-R1-Flat` (Unitree's recipe) | scratch | 10,001 | (2026-10-22 run) |
| control | `Unitree-R1-Flat-FT`: same, no delay | base model_10000 | +3,000 → model_12999 | 1 h 52 min |
| **delay** | `Unitree-R1-Flat-Delay-FT`: random command delay 0–50 ms | base model_10000 | +3,000 → model_12999 | 4 h 09 min (GPU shared with the jump runs) |

- **Delay model**: lag ~ U{0..10} physics steps of 5 ms (0–50 ms), drawn again every 1 s with a different phase in each env.
  This is the recipe of `Mjlab-Tracking-Flat-Unitree-R1-Delay` (L4DC paper).
- **One lag for all 24 joints.** mjlab 1.2.0 (the version `unitree_rl_mjlab` pins) gives each actuator group its own `DelayBuffer`,
  so the 5 R1 groups would each draw a different lag. `SharedLagDelayedActuatorCfg` in `scripts/env_cfgs.py` gives every group's
  buffer a generator with the same seed, so they draw the same lags. `scripts/check_shared_lag.py` steps the env and checks that the lags match:
  equal in every step, lags 0–10 all seen.
- **No speed curriculum in the fine-tunes.** A resumed run restarts mjlab's step counter at 0. The repo's curriculum would then
  narrow forward speed back to −0.5..1.0 m/s for 5,000 iterations. Both fine-tunes keep the final ranges from the first iteration:
  x −1..2, y −1..1, yaw rate −1..1.
- The control arm uses the same +3,000 iterations without delay. Any gain over the control therefore comes from the delay, not from more training.
  This is a single seed per arm.

## Result 1: CPU sim2sim, constant command delay (`sim2sim_cpu/`)

Plain CPU MuJoCo + onnxruntime with the observation/action wiring of Unitree's C++ deploy (`../walk_run_eval_2026-10-22/scripts/sim2sim_cpu.py`).
10 seeds, 20 s, commanded speed 0 / 0.5 / 1.0 / 1.5 / 2.0 m/s. The delay is held constant for the whole run, which is stricter than training
(training redraws it every second).

**Survived, all 5 speeds pooled (out of 50):**

| delay | 0 ms | 20 | 30 | 40 | 50 | 60 | 80 |
|---|---|---|---|---|---|---|---|
| base (model_10000) | 50 | 50 | 50 | 44 | 15 | 0 | 0 |
| control (+3,000 plain) | 50 | 50 | 50 | 48 | 16 | 0 | 0 |
| **delay (+3,000 with 0–50 ms)** | **50** | **50** | **50** | **50** | **50** | **50** | **46** |

- Base and control both fail between 40 and 60 ms. The delay policy survives everything up to 60 ms and 46/50 at 80 ms.
  The only 80 ms failures are 4 of 10 seeds at **zero command (standing)**. Standing was the weak skill before as well.
- Standing alone (0 m/s): base 6/10 at 40 ms, 1/10 at 50 ms. Control 10/10 at 40 ms, 0/10 at 50 ms. **Delay 10/10 up to 60 ms, 6/10 at 80 ms.**
- The extra 3,000 plain iterations (control) did not move the limit. Same finding as the tracking policies in the paper.

## Result 2: speed tracking in the training simulator (`speed_tracking/`)

64 robots per speed, 20 s, mean forward speed over the last 16 s, no delay, no pushes. No robot fell in any arm.

| commanded vx (m/s) | −0.5 | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| base | −0.43 | 0.00 | 0.45 | 0.96 | 1.50 | 1.97 |
| control | −0.43 | 0.00 | 0.43 | 0.96 | 1.50 | 1.98 |
| delay | −0.39 | 0.00 | 0.46 | 0.97 | 1.45 | **1.85** |

**Cost:** the delay policy is about 7 % slow at 2.0 m/s and about 10 % slow backwards. It matches the others from 0 to 1.0 m/s.
In the CPU sim2sim at 0 ms it also reaches 1.82 m/s at the 2.0 command (base 1.98). The training rewards are not comparable across the arms.
The delay arm's episode reward is lower (18.7 vs 32.2) because the same reward terms are scored while its commands arrive late.

## Video

`walk_run_delay_model12999.mp4`: delay policy, schedule 0 → 0.5 → 1.0 → 1.5 → 2.0 → 0 m/s every 4 s, rendered in the training simulator, no falls.

## Files

| path | what |
|---|---|
| `walk_run_delay.onnx`, `walk_run_control.onnx` | exported actors (same I/O as `../walk_run_eval_2026-10-22/run_model10000.onnx`) |
| `sim2sim_cpu/sim2sim_{base,control,delay}.{json,txt}` | per-run results and the summary tables |
| `speed_tracking/walk_run_{base,control,delay}.json` | Result 2 |
| `scripts/env_cfgs.py`, `scripts/__init__.py` | the R1 velocity config with the two fine-tune tasks (goes to `unitree_rl_mjlab/src/tasks/velocity/config/r1/`) |
| `scripts/run_walk_delay_followup.sh` | control run + all evaluation, exactly as run |
| `scripts/check_shared_lag.py` | shared-lag check |

Training command of the delay arm (from `~/unitree_rl_mjlab`):
```
.venv/bin/python scripts/train.py Unitree-R1-Flat-Delay-FT --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run 2026-10-21_21-54-28_walk_run_flat --agent.load-checkpoint model_10000.pt \
  --agent.max-iterations 3000 --agent.run-name walk_run_delay_ft --agent.logger tensorboard
```
Run dirs: `logs/rsl_rl/r1_velocity/2026-10-23_00-30-49_walk_run_delay_ft/`, `.../2026-10-23_04-43-27_walk_run_control_ft/`.

## Limits

- One seed per arm. Simulation only. The CPU harness has no pushes, observation noise or terrain.
- For a hub "stand" skill, the delay policy at zero command is now the best candidate (60 ms margin). The handoff from a tracking policy to this stand has not been tested.
- For a first hardware test, the deploy config `../walk_run_eval_2026-10-22/deploy_v1/` still applies (corrected action scale).
  Swap in `walk_run_delay.onnx` as `exported/policy.onnx`, and cap the joystick forward speed at 1.0 m/s.
