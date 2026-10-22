# R1 walk/run velocity policy — evaluation (2026-10-22)

Policy: `unitreerobotics/unitree_rl_mjlab`, task `Unitree-R1-Flat`, trained on the RTX 5080 workstation,
4096 envs, 10001 iterations (2026-10-21 21:54 -> 2026-10-22 02:49).
Run dir: `~/unitree_rl_mjlab/logs/rsl_rl/r1_velocity/2026-10-21_21-54-28_walk_run_flat/`.
The repo's curriculum widens forward speed from (-0.5, 1.0) to (-1.0, 2.0) m/s at iteration 5000, so
model_5000 = "walk" policy (trained up to 1.0 m/s) and model_10000 = "walk+fast" policy (up to 2.0 m/s).

Scripts (copies in `scripts/`; they run from `~/unitree_rl_mjlab/scripts/` inside the unitree_rl_mjlab checkout): `eval_walk_run.py` (fixed-command metrics + ONNX export + video),
`eval_stop_flight.py` (flight phase + stop behaviour), `eval_yaw_trace.py` (heading over the video schedule).
Sections 1-4: same simulator the policy was trained in (mjlab / MuJoCo-Warp, flat plane, no pushes, no obs noise).
Section 5: sim2sim — exported ONNX in plain CPU MuJoCo with the observation/action wiring of Unitree's C++ deploy.

## Speed tracking (64 robots per speed, 20 s, mean over the last 16 s, robots that never fell)

| commanded vx (m/s) | model_4000 | model_5000 | model_10000 |
|---|---|---|---|
| -0.5 | — | — | -0.43 |
| 0.0  | 0.00 (drift ≤5 mm) | 0.00 (drift ≤3 mm) | 0.00 (drift ≤3 mm) |
| 0.25 | 0.24 | 0.23 | — |
| 0.5  | 0.47 | 0.47 | 0.45 |
| 0.75 | 0.72 | 0.71 | — |
| 1.0  | 0.98 | 0.96 | 0.96 |
| 1.5  | 1.44 | 1.44 | 1.51 |
| 2.0  | — | 37/64 FELL | 1.98 |

Falls: 0 everywhere except model_5000 at 2.0 m/s (outside its training range).
Note: the training-log metric `error_vel_xy` (~0.8 at the end) is NOT a fixed-speed tracking error — it averages
random lateral + yaw commands and command switches; use this table instead.

## Is 2.0 m/s a run? (`stop_flight_model*.json`)

Flight phase (both feet off the ground) at 2.0 m/s = **2.7 %** of the time (human running ≈ 30 %+).
So model_10000 is a **fast walk up to 2 m/s, not a true run** — call it "fast walk" on the slide.

## Stopping (command steps from vx to 0)

- No falls at any speed for model_10000.
- Stopping distance: 0.17 m from 0.5 m/s, 0.40 m from 1.0, 0.65 m from 1.5, 0.86 m from 2.0.
- Heading change during the stop: mean 2–4°, worst 15°.
- Heading drifts ~1.5°/s while walking (yaw command 0 but no heading hold): ~30° over the 23 s video.
  In the videos the robot ends up facing the camera mostly because of this drift + viewpoint, not a spin.

## Standing at zero command (hub candidate)

All 192 standing robots: 0 falls, base drift ≤ 5 mm over 16 s, feet planted — unlike the PD-only default stand
that tips over in ~3 s (hub-switch probe, 2026-10-20). Not yet tested: pushes, starting from a tracker's
end pose, and the hand-off from the dance/tennis trackers.

## 5. Sim2sim: ONNX in plain CPU MuJoCo, deploy wiring (`scripts/sim2sim_cpu.py`, `sim2sim_cpu/`)

Model = the exact MjModel mjlab compiles for training (saved as .mjb, nominal — no DR); rollout uses
onnxruntime + mujoco only, obs built as `deploy/include/isaaclab/envs/mdp/observations/observations.h` does.
**Parity check first** (same states copied GPU -> CPU): ONNX vs torch actor max diff 2e-6; obs identical
(only step 0 differs = test artifact, command not yet set); motor targets differ by a constant ≤0.015 rad =
the training env's encoder-bias randomization. So the harness reproduces the deploy wiring.
Rollouts: 20 s, 10 seeds (±0.05 rad initial joint noise), action delay applied at physics rate (200 Hz).

**Survival (of 10) / achieved vx (m/s)**

| cmd vx | model_10000, 0 ms | 20 ms | 40 ms | model_5000, 0 ms | 20 ms | 40 ms |
|---|---|---|---|---|---|---|
| 0 (stand) | 10 / 0.00 | 10 / 0.00 | **6** / 0.01 | 10 / 0.00 | 10 / 0.00 | **2** / 0.00 |
| 0.5 | 10 / 0.44 | 10 / 0.44 | 10 / 0.43 | 10 / 0.48 | 10 / 0.44 | 10 / 0.45 |
| 1.0 | 10 / 0.95 | 10 / 0.90 | 10 / 0.88 | 10 / 0.94 | 10 / 0.91 | 10 / 1.04 |
| 1.5 | 10 / 1.50 | 10 / 1.42 | 10 / 1.39 | 10 / 1.43 | 10 / 1.44 | **0** |
| 2.0 | 10 / 1.98 | 10 / 1.89 | 8 / 1.92 | — | — | — |

- Cross-engine transfer is clean at 0 ms (150/150 for model_10000).
- **Standing is the most latency-sensitive skill**: at 40 ms the stand falls after 4–8 s (slow-building
  oscillation). The training config has NO action-delay randomization -> retrain with delay DR (the L4DC paper's
  fix) before using it as the hub stand on hardware.

**Deploy-config bug found:** Unitree's shipped `deploy/robots/r1/config/policy/velocity/v0/params/deploy.yaml`
uses a flat action scale 0.25, but training uses per-joint 0.25·effort/stiffness (legs/waist 0.15, ankles
0.3125, shoulder pitch/roll 0.375, shoulder yaw/elbow/wrist 0.4125). With the shipped 0.25 the policy falls in
**every** walking run (~1.5 s) and in standing with any delay. `deploy_v1/deploy.yaml` = v0 with the trained
scales and lin_vel_x up to 2.0 (installed locally as `velocity/v1/` with `exported/policy.onnx` = model_10000).
For a first hardware test, cap lin_vel_x back to 1.0. The deploy also clamps the joystick command to this range.

## Files

- `walk_model5000.mp4` — stand 3 s -> 0.5 m/s -> 1.0 m/s -> stop (16 s)
- `run_model10000.mp4` — stand -> 0.5 -> 1.0 -> 1.5 -> 2.0 m/s -> stop (23 s)
- `walk_model5000.onnx`, `run_model10000.onnx` — exported actors. Unitree's C++ deploy loads
  `deploy/robots/r1/config/policy/velocity/<ver>/exported/policy.onnx` + `params/deploy.yaml`; the shipped
  v0 deploy.yaml matches this task's obs layout (gait_phase period 0.6, 24 joints, action scale 0.25).
- `*.json` — raw numbers.
