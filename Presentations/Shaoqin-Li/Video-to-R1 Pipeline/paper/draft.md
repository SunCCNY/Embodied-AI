# From Video to Humanoid: A Reproducible Motion-Imitation Pipeline for the Unitree R1, and What Breaks It in Transfer

*Working paper — draft 1, 2026-10-08. Shaoqin Li¹, Yi Sun¹ (author list to be confirmed).*
*¹ Applied and Embodied AI Laboratory, The City College of New York.*

*Status: every number in this draft comes from a run whose output is in this folder. Section 6.4 (seed study)
is still running; its table is marked PENDING and must not be cited until it is filled in.*

---

## Abstract

We build and release a complete pipeline that turns an ordinary monocular video of a person into a motion-tracking
policy for the Unitree R1, a 24-joint humanoid for which no motion-imitation work has been published. The pipeline
chains four existing systems: GVHMR for world-grounded human motion recovery, GMR for kinematic retargeting, a
BeyondMimic-style tracking policy trained with PPO in either mjlab (MuJoCo-Warp) or Isaac Lab (PhysX), and a
standalone CPU MuJoCo harness for sim2sim evaluation. We add the R1 to each of them. Every stage reproduces
bit-for-bit from public clones at pinned commits. We evaluate the policies across engines under the training
randomization and 16 stress scenarios, on a tennis forehand and a dance taken from video. Three findings matter for
anyone building a similar pipeline. (1) Our first sim2sim "failure" was not a physics gap but a harness that had no
floor and the wrong timestep; a one-at-a-time ablation shows these two settings account for all of it, while
contact dimension, solver iterations, friction and contact priority are negligible. (2) Command delay is the dominant
failure mode, and it appears identically in policies trained in both engines: at 40 ms, survival is 2/20 and 0/20.
Training with randomized delay restores it to 15/20 in both, but in PhysX it costs robustness when there is no delay
(29/30 → 17/30). (3) Joint-space tracking error overstates the gap for video-derived motion: on the dance, 27% of
reference frames pin the left shoulder at a joint limit, where the policy reaches a similar hand position through a
different shoulder configuration. Retargeting reports these frames as limit-compliant.

## 1. Introduction

Learning humanoid skills from human video promises an unlimited supply of demonstrations. The recipe has settled into
a pipeline: recover the human's 3D motion from video, retarget it onto the robot's kinematics, train a policy in
simulation to track the retargeted reference, and check the policy in a second simulator before hardware
[VideoMimic; H2O; ASAP; BeyondMimic]. Each stage now has a strong open-source implementation. In practice, however,
the stages are developed for a few robots (most often the Unitree G1 and H1), and connecting them for a new robot
means writing code at every boundary: robot models, IK configurations, joint orderings, unit and frame conventions,
actuator limits, and an evaluation harness.

This paper reports doing that for the Unitree R1, end to end, on a single workstation, and reports what we found when
we tested the result. The contributions are:

1. **A reproducible video-to-policy pipeline for the R1.** We add R1 support to GMR (an R1 model and IK configurations
   for all eight GMR input formats), to mjlab and to Isaac Lab (robot definitions and tracking tasks, including a
   randomized command-delay variant). We release every stage with intermediate data and five trained policies.
   Re-running each stage from fresh clones reproduces the recorded outputs exactly (Section 5).
2. **A cross-engine robustness study.** Policies trained in MuJoCo-Warp and in PhysX are evaluated in a third
   setting, standalone CPU MuJoCo, under the training randomization and 16 stress scenarios (Section 6).
3. **Three practical findings**: a sim2sim harness must match the training model's floor and timestep before any gap
   means anything; command delay dominates transfer failure regardless of training engine; and joint-space error
   misleads on video-derived references that saturate joint limits (Section 7).

## 2. Related work

**Human motion from video.** GVHMR [Shen et al., 2024] estimates SMPL-X body motion [Pavlakos et al., 2019] in a
gravity-view coordinate frame, which removes much of the world-frame drift of earlier autoregressive methods.
It is the Stage 1 estimator here.

**Retargeting.** GMR [Araujo, Ze et al., 2025] retargets human motion to many humanoids in real time with a
two-stage IK, and shows that retargeting quality materially changes downstream tracking performance. Earlier
humanoid-imitation work retargeted by fitting robot shape and optimizing joint angles [H2O, He et al., 2024].
The R1 is not among GMR's supported robots; we add it.

**Motion tracking.** Physics-based tracking of large motion libraries was established in character animation
[PHC, Luo et al., 2023] and carried to real humanoids [H2O; OmniH2O; ASAP, He et al., 2025]. BeyondMimic
[Liao et al., 2025] gives a single tracking recipe (observations, rewards, randomization) that reaches highly dynamic
motions on hardware with shared hyperparameters. VideoMimic [Allshire et al., 2025] closes the loop from casual
video to real humanoid policies, including scene reconstruction. We use the BeyondMimic recipe as implemented in
mjlab [Zakka et al., 2026] and in its original Isaac Lab code (`whole_body_tracking`).

**Sim2sim and transfer.** Humanoid-Gym [Gu, Wang and Chen, 2024] made train-in-one-simulator, verify-in-another an
explicit step before hardware. PolySim [Lei et al., 2025] goes further and trains across several engines at once.
Domain randomization [Tobin et al., 2017; Peng et al., 2018] is the standard defence against the gaps such checks
expose, and actuator and latency modelling has long been known to matter for legged transfer [Hwangbo et al., 2019].
ASAP learns a residual action model from real data to correct what randomization does not. Our earlier working paper
on the R1 [Li, 2026, *What a sim2sim check can and cannot tell you*] found that a walking policy passing a sim2sim
check fell within seconds under a real-time harness, because the check contained no notion of timing. The present
study reaches a related conclusion for imitation policies by a different route.

## 3. The robot

The Unitree R1 is a 1.2 m, 28.9 kg humanoid with 24 driven joints in the model we use: 6 per leg, 2 waist, and 5 per
arm. We start from Unitree's MuJoCo description (`unitree_mujoco`, BSD-3) and keep one MJCF with 43 meshes for all
stages. PD gains and torque limits come from Unitree's deployment description. Earlier in the project, omitting the
torque limits produced runaway divergence, so they are enforced everywhere. Table 1 lists the actuator parameters
shared by training and evaluation.

*Table 1 — actuators (from `sim2sim_r1_tennis_v4_standalone.py`, identical in the mjlab robot definition).*

| Joint group | Kp | Kd | Torque limit (N·m) |
|---|---|---|---|
| hip pitch / roll / yaw | 150 | 8 | 88 |
| knee | 200 | 10 | 139 |
| ankle pitch / roll | 60 | 4 | 50 |
| waist roll / yaw | 80 | 5 | 50 / 88 |
| shoulder pitch / roll | 60 | 4 | 25 |
| shoulder yaw, elbow, wrist roll | 40 | 3 | 25 |

## 4. The pipeline

### 4.1 Stage 1: video → SMPL-X (GVHMR)
We run GVHMR unmodified at commit `ee960bb`, with the static-camera option. On the RTX 5080 (Blackwell), the stock
install fails in five separate ways (torch pins, no pytorch3d wheel, split CUDA headers, an undeclared dependency,
a missing `nvcc`). SETUP.md gives the working recipe. Processing time is 40.6 s for the 10.4 s tennis clip and
100.9 s for the 22.1 s dance clip, almost all of it preprocessing (detection, 2D pose, features).

### 4.2 Stage 2: SMPL-X → R1 (GMR)
We add the R1 to GMR as a model, its meshes, IK configurations and parameter entries; one script installs these into a
stock GMR clone. IK configurations for all eight GMR input formats are generated from one ratio rule that reproduces
the two hand-written configurations byte-for-byte before writing any others. The video path uses `smplx_to_r1.json`.
GVHMR output has a slow vertical drift (about 15 cm over the tennis clip), which we remove with a linear ground pass
fitted to the lower envelope of foot height. IK runs at 305–449 frames per second on CPU. The output is root position,
root orientation and 24 joint angles at 30 fps.

Plausibility on tennis: no joint-limit violations, peak joint speed 22.5 rad/s, above a 20 rad/s servo budget on 4
frames; 9% of frames in ground contact. Section 7.3 shows that "no violations" hides a real problem.

### 4.3 Stage 3: trajectory → tracking policy
The trajectory is replayed kinematically in mjlab to produce a 50 Hz reference with full-body positions, orientations
and velocities. We then train a tracking policy with the BeyondMimic recipe: PPO [Schulman et al., 2017], 4,096
parallel robots, 50 Hz control over 200 Hz physics (5 ms × 4). The policy observes a 135-dimensional vector: the
reference joint positions and velocities (48), the anchor (torso) position and orientation relative to the reference
(3 + 6), base linear and angular velocity (6), joint positions and velocities (48), and the previous action (24). It
outputs joint position targets as offsets from a default pose, scaled per joint.

Training randomization covers observation noise, initial pose and velocity, velocity pushes every 1–3 s, torso
center-of-mass offset, joint encoder bias, and foot friction in [0.3, 1.2]. The **delay** variants add a command delay
drawn uniformly from 0–50 ms (0–10 physics steps) per robot and redrawn every second with staggered phase. It is
implemented with the engine's own delay buffer in mjlab and with an equivalent action term in Isaac Lab.

The same task is implemented in both engines. The mjlab task registers the R1 as a new robot. The Isaac Lab version
adds the R1 to `whole_body_tracking`, using a URDF generated from the same MJCF, and lets the motion loader reorder
joints and bodies by name, because PhysX orders the 24 joints differently from MuJoCo.

### 4.4 Stage 4: sim2sim harness
The harness runs exported ONNX policies in standalone CPU MuJoCo (`mj_step`). It builds the physics model from the raw
MJCF with MuJoCo's spec API and reproduces the training model's settings explicitly: floor, foot contact parameters,
timestep, solver iterations and implicit-fast integrator. It reimplements the observation, the training
randomization, and the training termination rules: an anchor-orientation, anchor-height or end-effector-height
deviation beyond 0.25 m ends the episode. For Isaac-trained policies, a name-based mapping translates between PhysX
and MuJoCo joint order. A validation mode checks that the harness's first observation equals the one Isaac Lab
computes, and the maximum difference is 0.00000.

Stress scenarios add, on top of the training randomization: a constant command delay of 1–3 control steps (20–60 ms),
pushes at 2–3× training magnitude, ±20% PD gain, a 2 kg torso payload, foot friction fixed at 0.15, and removal of
base position and/or base linear velocity from the observation. Each scenario runs 20–30 seeds. A seed **survives** if
it completes the whole clip without triggering a termination rule.

## 5. Reproducibility

All code, the intermediate data of each stage, five trained policies with their exact configurations, and all raw
results are released (`Video-to-R1 Pipeline/`). Upstream systems are pinned: GVHMR `ee960bb`, GMR `bb1bbe4`,
mjlab `c2e1e06`, Isaac Lab `b0542fe2d`, whole_body_tracking `cd65172`. On 2026-10-08 we re-ran every stage from fresh
clones using only the released files and compared against the outputs used in this paper:

*Table 2 — reproduction check.*

| Step | Result |
|---|---|
| Stage 1, GVHMR on tennis | every output array identical (max abs. difference 0.0) |
| Stage 2, retarget to R1 | root position, rotation and all 24 joints identical |
| trajectory → CSV | byte-identical |
| CSV → 50 Hz reference | all 7 arrays identical |
| Stage 3, both R1 tasks | train and export (5-iteration smoke test; full training is not bit-deterministic on GPU) |
| Stage 4, delay-trained tennis policy | survival and error identical, seed for seed |
| Stage 4, Isaac-trained dance policy | observation check 0.00000; error identical |

## 6. Results

Two motions: **tennis**, a forehand from GVHMR's example clip (10.4 s), and **dance**, a stock video (22.1 s used).
Five policies:

*Table 3 — policies.*

| Policy | Engine | Training | In-engine evaluation |
|---|---|---|---|
| tennis | mjlab | 3,000 iterations (59 min), continued to 12,998 (+3 h 19 min) | 96.9% → 98.4% success, 64 episodes |
| tennis + delay | mjlab | fine-tuned from 12,998 for 3,000 iterations (55 min) | — |
| dance | mjlab | 3,000 iterations | 100% success, 64 episodes |
| dance | Isaac Lab | 3,000 iterations (≈100 min) | — |
| dance + delay | Isaac Lab | 3,000 iterations, from scratch | — |

Joint error below is the L2 norm of the 24-joint error vector, averaged over the episode. This is the same quantity as
mjlab's `error_joint_pos`. Dividing by √24 gives the per-joint RMS.

### 6.1 The first sim2sim failure was the harness

The first harness loaded the robot MJCF directly. The policy diverged: mean joint error 4.6–8.9 rad across harness
versions, against 0.78 inside mjlab. Running the policy as the raw PyTorch checkpoint on CUDA instead of ONNX on CPU
changed nothing (4.70 rad). Our first diagnosis was a discrepancy between the MuJoCo-Warp and native MuJoCo solvers.
That was wrong. Ten thousand more training iterations nearly halved the in-engine body-position error (MPKPE
0.30 → 0.16 m) but improved the harness by only 4%, a sign that the harness, not the policy, was at fault. The raw MJCF has no ground plane, and
the harness stepped physics at 2 ms × 4 (125 Hz) instead of training's 5 ms × 4. We then rebuilt the training model
in the harness and restored each raw setting one at a time:

*Table 4 — one-at-a-time ablation, tennis policy, nominal rollout (joint error).*

| Raw setting kept | Joint error |
|---|---|
| none (exact training model) | 0.577 |
| contact dimension | 0.576 |
| solver iterations | 0.577 |
| foot friction | 0.582 |
| contact priority | 0.577 |
| all four above together | 0.582 |
| **timestep** | **1.719** |
| **floor** | **6.969** |

Only the floor and the timestep matter. Once they match, the policy tracks the whole clip.

### 6.2 Robustness under randomization and stress

*Table 5 — survival (seeds surviving the full clip). Blank = not run.*

| Scenario | tennis | tennis + delay | dance (mjlab) | dance (Isaac) | dance (Isaac) + delay |
|---|---|---|---|---|---|
| nominal | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| training randomization | 30/30 | 30/30 | 29/30 | 29/30 | 17/30 |
| + 20 ms delay | 19/20 | 20/20 | | | 18/20 |
| + 40 ms delay | 2/20 | 15/20 | | 0/20 | 15/20 |
| + 60 ms delay | | 8/20 | | | 11/20 |
| + 2× pushes | 17/20 | | | | |
| + 3× pushes | 11/20 | 11/20 | | | |
| + foot friction 0.15 | 5/20 | 10/20 | | | |
| PD gain −20% / +20% | 20/20 / 20/20 | 19/20 / — | | | |
| + 2 kg payload | 20/20 | 20/20 | | | |
| no base position | 20/20 | | | | |
| no base linear velocity | 5/20 | 4/20 | | | |
| no base position or linear velocity | 14/20 | | | | |
| joint error, nominal | 0.577 | 0.597 | 2.24 | 2.01 | 2.19 |

Under the randomization they were trained with, all policies trained without delay survive 29–30 of 30 seeds,
whichever engine trained them. Parameter perturbations within plausible hardware variation (gains ±20%, 2 kg) are
harmless. Three perturbations are not: command delay, low foot friction, and the loss of base linear velocity.

### 6.3 Command delay

Delay is the sharpest failure. 20 ms is tolerated, but at 40 ms only 2 of 20 tennis rollouts survive. Most fail on
end-effector height, meaning a foot or hand leaves its reference height by more than 0.25 m. The PhysX-trained dance
policy fails in the same way, and worse: 0/20. Because the two policies come from different engines, the weakness is
not a quirk of either engine.

Training with randomized delay of 0–50 ms changes 40 ms survival from 2/20 to 15/20 (mjlab, tennis) and from 0/20 to
15/20 (Isaac, dance). The mjlab policy pays nothing for this under the training randomization (30/30) and becomes
slightly less precise (joint error 0.577 → 0.597). The Isaac policy pays heavily: 29/30 → 17/30 with no added delay.
We do not yet know why. Two candidates: the mjlab policy was fine-tuned from a converged checkpoint while the Isaac
policy was trained from scratch, and 3,000 iterations may not be enough for the harder problem.

### 6.4 Seed study — PENDING

*Running at the time of writing.* Two questions: (a) does the 40 ms gain come from delay or just from 3,000 more
iterations? We continue the same tennis checkpoint for 3,000 iterations **without** delay as a control. (b) How much
does the result vary across training seeds? Three seeds per arm (control: 42, 43, 44; delay: 42, 43, 44), each
evaluated on nominal, training randomization, 20/40/60 ms delay, 3× pushes and friction 0.15.

| Scenario | control (s42 / s43 / s44) | delay (s42 / s43 / s44) |
|---|---|---|
| PENDING | | |

### 6.5 Base linear velocity

Removing base position from the observation is harmless (20/20). Removing base linear velocity leaves 5/20, and 4/20
after delay training. Removing both gives 14/20, which is better than removing velocity alone. This is consistent with
the policy using the two signals together, but we have not tested it further. A real R1 has no direct measurement of
base linear velocity. Before hardware, either a state estimator must supply it or the policy must be trained without
it. mjlab's no-state-estimation task variant, included in the release, does the latter and has not yet been evaluated.

## 7. Discussion

### 7.1 Check the harness before blaming the physics
A sim2sim check compares two simulators, but in practice it compares two *models*. Our first harness differed from
training in two settings a reader of the MJCF would not see: the floor comes from the training framework's terrain,
not from the robot file, and the timestep comes from the task configuration. The ablation in Table 4 is cheap, and it
turns a vague "physics gap" into two named, fixable causes. We suggest running it before reporting any sim2sim gap.

### 7.2 Delay is the gap that sim2sim checks leave out
A standard sim2sim check steps both simulators synchronously, so it cannot reveal sensitivity to delay unless delay is
injected on purpose. Here 40 ms was enough to break policies from both engines. In our earlier walking study, the
same robot's walking policy tolerated about 20 ms. Real control loops routinely have tens of milliseconds of latency
between policy and actuator. The first measurement to make on hardware is therefore the actual command latency, and
the delay range used in training should be centred on it.

### 7.3 Joint-space error misleads on video-derived references
The dance policy's joint error (2.24) is nearly four times the tennis policy's (0.58), yet it completes the clip in
both engines. This is not a harness artifact: mjlab's own training metric ends at 2.04. Per joint, the error sits
almost entirely in the left shoulder (pitch RMS 1.49 rad, yaw 1.31 rad, with signed means of +1.03 and +0.89 rad).
In 27% of dance reference frames, the left shoulder is pinned at a joint limit. In those frames the left-shoulder
error is 2.06 rad RMS, and in the others it is 0.45 rad. The left hand, however, is placed almost as well in both sets
(19 vs 16 cm, world frame, so including root drift). The policy therefore reaches the pose through a different
shoulder configuration than the one the IK chose when it clamped at the limit. Retargeting reports these frames as
having no limit violations, because the IK clamps at the limit. A violation count of zero should not be read as "this
reference is comfortably feasible", and joint-space error on such references mixes real tracking failure with
harmless redundancy. Task-space error, or a check of how often the reference sits at a limit, is more informative.
The dance policy also had a quarter of the tennis policy's training iterations, which explains part of the gap.

## 8. Limitations

- **No hardware.** All results are in simulation, with idealized actuators (PD with torque limits, no motor dynamics).
  The sim2real items listed here (delay, base velocity, friction margin) have not been tested on the physical R1.
- **Two motions, mostly one seed** (Section 6.4 addresses the seed question for the delay result).
- The harness holds delay constant for the whole episode. Training redraws it every second and starts at zero, so the
  40–60 ms scenarios are stricter than training.
- Isaac Lab training ran on native Windows, because GPU PhysX hangs under WSL2 on this GPU. The harness ran under WSL2.
  The two engines' policies are compared on the same harness, but their training runs differ in more than the engine
  (from scratch vs fine-tuned).
- The unexplained cost of delay training in PhysX (29/30 → 17/30).

## 9. Conclusion

A video of a person can be turned into a humanoid policy for a robot that no published imitation pipeline supports,
using only open tools, on one workstation, and every stage can be reproduced exactly. What the resulting policies
lacked was not physics fidelity but robustness to things a synchronous sim2sim check does not contain: command delay,
an unobservable base velocity, and low friction. Two of the three findings concern the check rather than the policy:
a harness must match the training model's floor and timestep, and joint-space error must be read against how often
the reference sits at a joint limit.

---

## References

- Allshire, A., Choi, H., Zhang, J., McAllister, D., et al. *Visual Imitation Enables Contextual Humanoid Control.* CoRL 2025. arXiv:2505.03729.
- Araujo, J. P., Ze, Y., Xu, P., Wu, J., Liu, C. K. *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking.* arXiv:2510.02252, 2025.
- Gu, X., Wang, Y.-J., Chen, J. *Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer.* arXiv:2404.05695, 2024.
- He, T., Luo, Z., Xiao, W., Zhang, C., Kitani, K., Liu, C., Shi, G. *Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation (H2O).* IROS 2024. arXiv:2403.04436.
- He, T., et al. *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning.* CoRL 2024. arXiv:2406.08858.
- He, T., Gao, J., Xiao, W., Zhang, Y., Wang, Z., Wang, J., Luo, Z., et al. *ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills.* RSS 2025. arXiv:2502.01143.
- Hwangbo, J., et al. *Learning agile and dynamic motor skills for legged robots.* Science Robotics, 2019.
- Lei, Z., Zhou, Z., Yin, S., et al. *PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization.* arXiv:2510.01708, 2025.
- Li, S. *What a sim2sim check can and cannot tell you.* Working paper, 2026 (this project).
- Liao, Q., Truong, T. E., Huang, X., Tevet, G., Sreenath, K., Liu, C. K. *BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion.* arXiv:2508.08241, 2025.
- Luo, Z., et al. *Perpetual Humanoid Control for Real-time Simulated Avatars (PHC).* ICCV 2023. arXiv:2305.06456.
- Pavlakos, G., et al. *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image (SMPL-X).* CVPR 2019.
- Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.* ICRA 2018.
- Schulman, J., et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- Shen, Z., Pi, H., Xia, Y., Cen, Z., Peng, S., Hu, Z., Bao, H., Hu, R., Zhou, X. *World-Grounded Human Motion Recovery via Gravity-View Coordinates (GVHMR).* SIGGRAPH Asia 2024. arXiv:2409.06662.
- Tobin, J., et al. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS 2017.
- Todorov, E., Erez, T., Tassa, Y. *MuJoCo: A physics engine for model-based control.* IROS 2012.
- Zakka, K., et al. *mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning.* arXiv:2601.22074, 2026.
- NVIDIA. *Isaac Lab* (software), https://github.com/isaac-sim/IsaacLab.
- HybridRobotics. *whole_body_tracking* (BeyondMimic motion tracking code), https://github.com/HybridRobotics/whole_body_tracking.
