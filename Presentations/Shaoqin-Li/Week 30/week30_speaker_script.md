# Embodied AI Week 30 — speaker script

For the 26-slide deck `embodied ai week 30.pptx`. Suggested speaking time: approximately 19 minutes, including the three short videos; allow questions afterward. Completed experiments are dated 2026-10-14. Calibration and the game are proposals. Use this script with the reviewed slide wording, not the earlier draft's claims about a controlled camera comparison or missing humanoid punch/jump examples.

## Slide 1 — GVHMR on a moving camera

**Time: 20 seconds.**

This week I tested the video-to-R1 pipeline on videos with moving cameras. I wanted to see how the existing method behaves before trying to improve it. I will show what failed, what I fixed, and the resulting robot motions. Then I will introduce two next steps: using our lab robot's data to calibrate simulation, and connecting separate skills into a game.

## Slide 2 — What I set out to answer

**Time: 40 seconds.**

I have three questions. First, how much freedom do I have when choosing an input video? So far I have selected clear, full-body clips with one main person, and my earlier presentation used a fixed camera. Those choices make the pipeline easier to use, but they also limit the videos I can use. Second, can existing measurements from our real R1 tell me which simulator settings are realistic? Third, can I move from copying one motion to connecting several motions that I control interactively?

## Slide 3 — The moving-camera test

**Time: 15 seconds.**

I started with GVHMR's existing moving-camera path. I did not train a new human-motion model. This part is an evaluation of the existing pipeline, including the compatibility problems I had to resolve to run it in my software environment.

## Slide 4 — What GVHMR asks of a video

**Time: 50 seconds.**

These are my practical video-selection rules. I prefer the whole body, including the feet, to remain visible because leg motion and foot contact matter when I retarget the result to the robot. GVHMR may infer a hidden joint, but that is different from observing it, and I have not measured reconstruction accuracy on cropped clips. I also select one clear subject. The demo follows one person track, so extra people can complicate tracking; I am not claiming that any second person makes the method fail. A fixed camera avoids the camera-estimation stage. A moving camera is supported, and that is the path I tested here.

## Slide 5 — The three clips

**Time: 45 seconds.**

I downloaded and screened eighteen candidate clips, checking people count, body visibility and camera motion, and then inspected them visually. I kept a fixed-camera home dance, a boxing clip where the camera follows the person, and a handheld street-dance clip. These tests were run on October 14. The fixed-camera dance is a reference example from the same pipeline. It is not a controlled camera experiment, because the person, motion and scene also change between clips. I credit the original Pexels videos in the source notes.

## Slide 6 — What broke first

**Time: 45 seconds.**

The fixed-camera clip ran, but both moving-camera clips hit problems in the camera-estimation path. First, pycolmap sometimes classified nearly identical frame pairs as a watermark case and returned no camera pose. The following line tried to use a matrix that did not exist. Second, newer PyTorch refused to reload the saved camera-trajectory data with its default loading behavior. The error text on this slide comes from the recorded setup notes. It is not a new terminal traceback captured for this presentation.

## Slide 7 — How it got fixed

**Time: 55 seconds.**

The patch makes two parts of the existing path compatible with the environment. It disables watermark detection in the two-view solver, and it checks whether a pose was returned before using its matrix. If no pose is available, that frame pair falls back to no camera motion. The second edit allows loading the trajectory file that the pipeline itself produced. With those edits, both moving-camera clips ran through the pipeline. These changes do not improve the learned human-pose model. The no-motion fallback also does not prove that the camera was actually stationary; it is a limitation to examine when evaluating reconstruction quality.

## Slide 8 — Fixed camera: video next to the R1

**Time: 40 seconds, including the 8.7-second video.**

**Cue:** Play the video once. Point to the human on the left and the simulated robot on the right.

This is the fixed-camera reference example. The video goes through human-motion recovery, retargeting and policy training, then the trained policy runs in plain MuJoCo. I am showing the final delay-trained policy under nominal conditions. The side-by-side view helps me inspect the sequence and timing. It does not supply a ground-truth three-dimensional human pose.

## Slide 9 — Moving camera: boxing

**Time: 45 seconds, including the 15-second video.**

**Cue:** Play the video once. Draw attention to the moving viewpoint and the wide stance.

Here the camera follows the boxer. The pipeline still produced a motion that the R1 policy could perform in simulation. This was the least robust of the three under strong pushes and low friction. In the retargeted reference, each ankle roll reaches its limit in more than half the frames. That could contribute to the difficulty, but I have not changed the ankle constraint and repeated the experiment, so I cannot claim it is the proven cause.

## Slide 10 — Moving camera: street dance

**Time: 40 seconds, including the 15-second video.**

**Cue:** Play the video once. Point out handheld camera motion while the robot completes the sequence.

The third clip uses a handheld camera. It also completed the pipeline and produced a policy that transfers to plain MuJoCo. This extends the examples beyond a fixed-camera tennis serve. It does not establish that the method works for every handheld video, severe occlusion, multiple interacting people or camera cuts. Those would require additional tests.

## Slide 11 — Did the robot survive in a second simulator?

**Time: 70 seconds.**

The table compares the policy after the first training stage with the policy after an additional delay-training stage. Each motion has its own policy. Under the usual training randomization, twenty-nine or thirty of the thirty trials survived. At forty milliseconds of added delay, the fixed-camera dance changed from zero to twenty surviving trials, boxing from zero to nineteen, and street dance from one to nineteen, each out of twenty. The sixty-millisecond tests improved too, but not every trial survived. The boxing policy remained vulnerable to strong pushes and a slippery floor. These are robot-survival measurements, not measurements of human-pose reconstruction accuracy.

## Slide 12 — What this does not show

**Time: 70 seconds.**

There are several limits I want to keep clear. The delay stage also adds three thousand training iterations. This table alone cannot tell me whether the improvement came from delay exposure or simply more training. The earlier tennis study used matched extra-training controls and three training seeds to separate those effects. Here I have only one training seed per clip. Also, different motions with different camera conditions do not isolate the camera effect. I do not have ground-truth human poses for the stock clips. Finally, I found that the earlier twenty-five-frame-per-second dance had been treated as thirty frames per second, so its robot motion ran one-point-two times too fast. These new clips use their source frame rates.

## Slide 13 — Making the simulator more real

**Time: 15 seconds.**

My second topic is a proposal for using data from our real R1. This week I am presenting the approach. I have not run a new experiment on the physical robot, and I will not do that while the gantry is unavailable.

## Slide 14 — Why I need real data

**Time: 45 seconds.**

Delay is the clearest reason to measure rather than guess. I tested chosen delay settings in simulation, but I have not measured the real robot's command-to-motion delay. Our lab robot is the R1 EDU Light with non-dexterous hands, and its SDK and logs are available. Prof. Sun advised waiting for the gantry before testing in person. I can still inspect existing logs offline. I also checked the official SDK example: it takes control and sends commands, so it is not something to run unchanged just to read sensor data.

## Slide 15 — The approach I propose

**Time: 65 seconds.**

First I would inventory the existing logs: which joint states, IMU signals, commands and timestamps were recorded, and whether they share a synchronized clock. Then I would fit the behavior the data can actually identify, starting with delay and motor response. I would keep some recorded motions out of the fitting process and use those to check the prediction. If the model predicts those held-out motions better, I can use measured parameter ranges in simulation and randomization. Existing logs may not contain enough commands or excitation to identify every parameter. Estimated torque is not a calibrated torque measurement, and an IMU does not directly give drift-free base velocity. Any new collection stays deferred until testing is permitted.

## Slide 16 — A bag of R1 skills

**Time: 15 seconds.**

My next series is a library of separately trained skills. Instead of ending at a single motion demonstration, I want to connect the skills and control the robot in a MuJoCo game. The first complete milestone is a fluent walk, run and jump sequence.

## Slide 17 — The idea

**Time: 45 seconds.**

Each item in the bag contains a trained policy and the information needed to run it: its reference motion, joint order, control settings and entry and exit conditions. The game sends my movement and action choices to a controller that selects the appropriate skill. The controller must keep the robot's state continuous during a switch. The school scene is the setting for the challenge, but the main technical result is the robot moving between points with connected skills. Kick, punch and object handling remain later extensions.

## Slide 18 — What already exists

**Time: 45 seconds.**

I searched for work I could reuse before designing a new framework. RoboJuDo is a useful reference because it separates the input controller, policy and simulator and supports switching between locomotion and motion-imitation policies. Its documented integrations do not establish an R1 drop-in for this project. ASE provides a reference for reusable skills and higher-level control. DeepMimic already includes humanoid jump, punch, walk and run motion examples, and its repository now points to MimicKit. The gap I need to address is integrating suitable R1 policies and testing the full walk-run-jump chain, not inventing the existence of these actions. One 2026 paper, Switch, deals directly with switching between skills, and the next slides present it.

## Slide 19 — Featured paper: Switch

**Time: 20 seconds.**

I picked Switch because it addresses the part of my next project that I expect to be hardest: connecting skills fluently. This is an April 2026 preprint by Yuen-Fui Lau and collaborators. I am presenting their work as a reference for my R1 project. These are their results, not experiments I ran.

## Slide 20 — Introduction & Summary

**Time: 45 seconds.**

My idea starts with a bag of motions. But three successful demonstrations do not automatically give me one successful chain. Switch combines a skill graph, a transition-trained tracker and an online scheduler. I find that useful because it treats the handoff as something to design and test. For my project, I would judge the connected rollout rather than only show walking, running and jumping in separate videos.

## Slide 21 — Core Challenges

**Time: 45 seconds.**

A switch changes what the controller is asked to do while the robot is already moving. The next policy receives the actual pose, velocity and foot contact left by the previous action. It cannot assume a reset to a convenient training pose. The paper identifies mismatched states and unfamiliar transitions as key difficulties. In my game, pressing jump during a run should lead to a feasible takeoff, not an instant change into the first frame of a jump clip.

## Slide 22 — Framework

**Time: 60 seconds.**

I read this figure like a route map. The graph connects similar motion states; training includes transition buffers; the scheduler selects the next reference online. The paper's hardware uses its nearest-neighbor scheduler variant. The useful idea for me is that selecting a destination skill is different from choosing a path into it. I would need to check whether my R1 can execute those handoffs, rather than assume that a visually smooth reference is physically stable.

## Slide 23 — Results

**Time: 60 seconds.**

This is a summary of the authors' MuJoCo table, with fifty evaluation trials per difficulty. The difficulties require one, two or three switches. The graph-trained variants reach one hundred percent success in these tests. That does not prove universal reliability, and fifty evaluation trials are not fifty independent training runs. The buffers and contact reward cannot raise a success column already at one hundred percent, so I should not credit every component with that gain. I would want the same discipline in my own results: define success, show the baseline, and inspect the failures.

## Slide 24 — Core Values: what I would use

**Time: 45 seconds.**

My takeaway is to make transitions part of the experiment. I would compare direct switching with a transition-aware controller, keep the robot's state continuous, and measure whether the complete walk-run-jump chain succeeds. Switch uses a unified tracker, which differs from my initial independent-policy plan. Its G1 work does not establish the R1 result I want, and I have not verified an author code release. It gives me a method to study, not a completed game to download.

## Slide 25 — Why fluent is harder than it sounds

**Time: 60 seconds.**

Three independent policies can each look good and still fail when connected. A walk clip does not automatically become a controller that follows my chosen heading and speed. At a switch, the next policy receives the position, velocity and contact state left by the previous one. It must handle that state instead of restarting from its training pose. Running into a jump also needs an appropriate takeoff phase, then a stable landing and return to locomotion. I will judge fluency from the actual connected rollout: no reset, no teleport, no fall, and no large movement discontinuity. That is the milestone; simply loading three policy files is not enough.

## Slide 26 — Next

**Time: 45 seconds.**

I would develop the milestone in stages. First I would establish controllable walking and running and test their transitions. Then I would add a jump that can start from locomotion, land and continue moving. Those stages together complete the first milestone; I am not dropping the jump requirement. After the chain works in a simple scene, I would add the school layout and test reaching points using the same controls. In parallel, I can review existing lab logs offline. Physical measurements stay deferred until the gantry arrives and Prof. Sun permits testing.


## Slide 27 References

Every source used or cited in this deck is listed here with its link. GVHMR is first, with its code, project page and paper, because it is the tool that turns each video into human motion. Say that the Pexels clips are credited to their creators and free to use. No need to read the links aloud; they are clickable in PowerPoint.

## Q&A preparation

- **Does GVHMR require a fixed camera?** No. It has a moving-camera path. My fixed-camera preference reduced preprocessing complexity; it was not proof of a hard algorithm limit.
- **Does a cropped leg make recovery impossible?** I have not tested that claim. Human-pose models can infer missing joints; whether the inferred motion is accurate enough for our robot needs measurement.
- **Have you improved GVHMR itself?** I fixed compatibility problems in camera preprocessing and data loading. I have not trained or validated a new human-motion-recovery algorithm.
- **Is replacing a missing camera pose with no motion always correct?** No. It avoids the crash, but can introduce camera-estimation error. A separate accuracy evaluation is needed.
- **Does the fixed clip prove the effect of moving the camera?** No. It is a reference example, not a matched experiment. A stronger test would hold the action and scene constant and vary camera conditions, with suitable pose/camera ground truth.
- **Why use survival rather than reconstruction error?** Survival tests the downstream robot controller. I do not have true three-dimensional human poses for the stock videos, so I cannot report reconstruction error against them.
- **Did one policy generalize to all three clips?** No. I trained a separate policy for each clip.
- **Could extra training explain the delay gains?** For the new clips, it is a confound. The earlier tennis study used three seeds and matched no-delay extra-training controls to isolate delay training.
- **Is boxing's ankle saturation the cause of failure?** It is a plausible explanation based on the reference, not a controlled causal finding.
- **Have you tested the real R1?** Not for this work. SDK/logs are available, but Prof. Sun advised waiting for the gantry before physical testing. Existing recordings can be inspected offline.
- **Can any log identify motor friction and delay?** No. Identifiability depends on the recorded commands, excitation, sampling and timing. I would report which parameters the available data actually supports.
- **Does real data turn the experiment into sim2real?** It calibrates the simulator. It does not establish successful policy deployment on the physical R1.
- **Are walk, run and jump already trained?** No new training or connected game result is claimed in this deck. They are the next project milestone.
- **What does fluent mean?** A user-controlled connected rollout that walks, runs, takes off, lands and resumes locomotion without state resets, teleportation or falls. I would report transition success and movement discontinuities rather than only show a selected successful video.
- **Why not blend two policy outputs and call it done?** The result may violate balance or contact requirements. Blending is a candidate mechanism; the connected rollout is the test.
- **Why not use RoboJuDo unchanged?** It supplies useful deployment structure, but the R1 model, policies, control settings and transition behavior must be integrated and verified.
- **Is there prior work with humanoid punch/jump?** Yes. DeepMimic includes those motion examples. That does not supply a validated R1-ready controller for this project.
- **Why is grasping later?** It needs hand/object modeling and contact objectives beyond the present body-motion pipeline. The first milestone explicitly focuses on walk, run and jump.

- **Why this paper?** It directly studies the fluent handoffs required by the first game milestone.
- **Is it the same architecture as separate bags?** No. Switch trains a unified tracker on graph-augmented transitions. Independent policies remain one candidate baseline for this project.
- **What does reported success mean?** The paper defines failure using a 0.5 m root-relative body-position-error threshold. This is an evaluation definition, not a guarantee against every failure mode.
- **Did they test our exact chain on R1?** No R1 walk/run/jump result is established by this paper. Do not present the authors' G1 demonstration as our result.
- **Does the hardware run the full graph-search planner?** The paper says its hardware deployment uses the nearest-neighbor variant.
- **What would I reproduce first?** A transition comparison in simulation, with fixed task definitions and explicit reset/failure accounting. This is a proposed experiment; no training was started for the presentation update.

## Evidence and sources

Experimental numbers and clip details come from the saved workstation evidence, not newly run experiments:

- `claude_brain/projects/embodied_ai_r1/video_to_r1_pipeline/results/new_clips_2026-10-14/README.md` and its `sweeps/` files.
- `claude_brain/projects/embodied_ai_r1/video_to_r1_pipeline/SETUP.md` and `EVIDENCE.md`.
- Local Week 30 assets: `assets/gvhmr_patch.diff`; crash-summary text: `week30_shots/captures/gvhmr_crash_text.txt` (a record from setup notes, not a fresh traceback).
- [Public experiment folder](https://github.com/SunCCNY/Embodied-AI/tree/main/Presentations/Shaoqin-Li/Video-to-R1%20Pipeline/results/new_clips_2026-10-14).
- [GVHMR official repository](https://github.com/zju3dv/GVHMR), [demo source](https://github.com/zju3dv/GVHMR/blob/main/tools/demo/demo.py), and [tracker source](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/preproc/tracker.py).
- [Unitree official R1 SDK example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/r1/low_level/r1_ankle_swing_example.cpp).
- [RoboJuDo release documentation](https://github.com/HansZ8/RoboJuDo/tree/release) and [policy interfaces](https://github.com/HansZ8/RoboJuDo/blob/release/docs/policy.md).
- [Switch paper](https://arxiv.org/abs/2604.14834), [full text](https://arxiv.org/html/2604.14834v1): Figures 1-3 and Table II reproduced from the PDF with attribution; local copy `paper_switch/switch.pdf`.
- [ASE](https://github.com/nv-tlabs/ASE), [DeepMimic motion directory](https://github.com/xbpeng/DeepMimic/tree/master/data/motions), and [MimicKit](https://github.com/xbpeng/MimicKit).

Original-video credit links:

- [A Woman Dancing At Home — Pexels 6003772](https://www.pexels.com/video/a-woman-dancing-at-home-6003772/).
- [Man Shadow Boxing — RDNE Stock project, Pexels 7187515](https://www.pexels.com/video/man-shadow-boxing-7187515/).
- [Person Dancing — Pexels 4540327](https://www.pexels.com/video/person-dancing-4540327/).

## Presenter checks

- Present in PowerPoint with the embedded videos. The PDF contains static poster frames, not playable video.
- Open each embedded video in the actual presentation application before the meeting; document review does not verify playback on that machine.
- If time is short, play each clip once and summarize the 40 ms row. Keep the limitations and distinguish results from proposals.
- Rehearse the final script after deck corrections; timings are suggestions, not a measured rehearsal duration.
