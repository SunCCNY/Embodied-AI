# Week 32 speaker script
Shaoqin Li · October 31, 2026 · About 25–30 minutes, with optional detail in the companion. Simulation only.

## Slide 1 — Week 32 | Connecting running and jumping

I will start with the project goal and what I had last week, because the meaning of this week’s results depends on that context. I am building a set of movement skills for the R1. Last week I had separate pieces. This week I worked on a continuous run, jump, landing and return to running. All the work is in simulation. The workstation handoff contains measurements and training logs, but no new rendered rollout video.

Evidence: Week 31 script; Week 32 P04/P08/P09 reports

## Slide 2 — The project goal is a reusable bag of R1 movement skills

My long-term idea is a bag of skills that I can use in one simulated scene. The R1 should follow movement commands and perform actions while staying balanced. That makes transitions important: I need to know not only whether it can jump, but whether it can run into the jump and keep moving afterward. Nothing in this presentation is a physical-robot demonstration.

Evidence: Week 31 slides 1–2 and 22–23

## Slide 3 — Week 31 established three separate pieces

Week 31 had three parts. First, I trained a speed-command locomotion policy. Second, I obtained a jumping-jacks tracker through the human-video pipeline. Third, I tried switching between two other motion policies. Those results showed useful individual skills, but they did not prove that one robot state could flow through running, a forward jump, landing and running again.

Evidence: Week 31 slides 5–8, 10–12, 18–22

## Slide 4 — These images show the Week 31 starting point

These are two historical frames from the previous presentation. The left is the delay-trained locomotion policy, and the right is the jumping-jacks comparison between a human video and the simulated R1. I am showing them only to make the starting point concrete. They are not images of the new command-jump or imitation policies evaluated this week.

The first slides explain the Week 31 background; all Week 32 counts refer to simulation. Historical pictures are explicitly labeled and do not substitute for current rollout media.

Evidence: Week 31 poster_walk_run_delay.png; poster_jumping_jacks.png

## Slide 5 — Week 31 left three specific gaps

The important distinction is between the names of the skills and the measured behavior. A policy going fast can still be walking. Jumping jacks can leave the ground without solving a running jump. And one successful rollout does not tell me how the skill handles different timing or delays. Those were the gaps I turned into explicit tests.

Evidence: Week 31 slides 12, 14–16, 20, 22

## Slide 6 — Week 32 targeted one continuous running-jump sequence

This diagram is the goal, not a rollout image. The policy should establish a run, take off into a larger jump, travel forward in flight, land, and return to running. The simulator is not reset between phases. I also need to distinguish the large commanded jump from the short flight phases that occur naturally while running.

Evidence: P04 fallback specification; P08 contract

## Slide 7 — I split the week into a prerequisite and two routes

There are two routes in the results, not one policy that changed its purpose halfway through. I first needed a true-running controller. One route extended that controller with a jump command. In parallel, the other route prepared a running-jump motion reference for a separate imitation tracker. Success in the command route does not make the imitation route a success.

Evidence: P04 README/JUMP_STAGE; P02; P08/P09 plans

## Slide 8 — Four terms make the rest of the talk easier to follow

A policy is the actual learned controller. A reference is a target motion. Training changes the controller to pursue its rewards or reference, while evaluation freezes the controller and applies a fixed test. This is why I separate the learning curves from the final success counts. The terms are related but not interchangeable.

Evidence: P04/P09 task and evaluation contracts

## Slide 9 — I kept the two policy families separate

These are two approaches to the same milestone. The command policy learns what to do when I request a jump. The tracking policy follows a retargeted motion. Their observation spaces and actuator models differ, so I do not compare them as matched experiments.

Evidence: P08 manifest; P09 export parity/config manifests

## Slide 10 — I froze the evaluation before choosing a model

I loaded pinned checkpoints rather than a latest-file guess. I kept the original detector and thresholds during evaluation. For imitation, I used the same twenty initial-condition seeds to compare candidates; those are evaluation episodes, not twenty independent training runs.

Evidence: P08 run_jump_metrics; P09 s6_eval.py

## Slide 11 — Before jumping, I trained and verified a true running gait

This is a step missing from my earlier summary. I did not jump directly from Week 31’s fast walk to the final jump result. I first fine-tuned that locomotion policy to produce a genuine running gait and checked it under a flight-based contract. The selected second version passed all twenty steady-running trials at a two-meter-per-second command, with about sixteen-point-seven percent airborne time. This was a running result, not yet a running-jump result.

Evidence: P04 raw/runftv2_model_14498_vx2.json; JUMP_STAGE §1

## Slide 12 — I then gave that running controller a jump request

For the command route I appended one jump-countdown input to the running controller. During a short requested window, the rewards encouraged the larger flight and height; outside the window, the normal running task remained. This makes the requested jump distinct from the small flight phases in an ordinary run. I tested the full before-and-after sequence in one physics rollout.

Evidence: P04 JUMP_STAGE §2; jump task self-tests

## Slide 13 — My first commanded jump was too flat, so I fixed the target

The first jump policy could make a long, flat leap but missed the height requirement. I found that its reward measured pelvis rise from the time of the request, while the test measured whole-body rise after takeoff. Those are different quantities. I changed the reward definition and weight, trained further and reported the fixed final checkpoint. It passed twenty of twenty nominal trials, but the stronger jump also made landing forces larger.

Evidence: P04 raw/jump_model_17497_jump_vx2.json; jumpapex_model_20496_jump_vx2.json

## Slide 14 — I reproduced the nominal commanded jump

The original command policy passed all twenty nominal episodes. Two commands per episode gave forty successful events. Random command timing was weaker: sixteen of twenty passed, below the eighteen-of-twenty acceptance target. Friction edge cells passed.

Evidence: P08 S1/rework_v2 raw

## Slide 15 — I found a clear action-delay boundary

At twenty-five milliseconds, the composite run-jump test failed even though the policy did not fall. At fifty milliseconds, all twenty episodes fell. Survival and the complete skill test are different outcomes.

Evidence: P08 S1/rework_v2 B0/D25/D50

## Slide 16 — I reduced landing peaks, but missed the target

The landing candidate reduced the mean peak from 8.77 to 6.27 body weights and the maximum from 10.00 to 8.01. But the target was a maximum no greater than seven. It missed that target and was not promoted. Torque saturation also increased.

Evidence: P08 S3/raw and S5/bankB B0

## Slide 17 — I gained delay-tolerant running and lost the jump

Delay training changed the behavior, but not in the way I needed. The policy could run under the tested delays without falling, while the jump disappeared. I stopped this branch rather than choosing an earlier checkpoint after seeing the failure.

Evidence: P08 S4/raw; S5 decision

## Slide 18 — For imitation, I changed the source of the motion

The imitation route continues the idea of teaching motions from human data, but this week the source was different. Week 31 reconstructed movement from ordinary video. This running-jump branch used LAFAN1 recorded human motion, then retargeted and repaired it for the R1. I call the two chosen references c2 and c3 in the logs. These are just candidate names, not separate controller architectures. The alternative c3 keeps its own source-anchored test.

Evidence: Week 31 slides 18–21; P02 README; P09 A5 admission

## Slide 19 — Then I built the imitation test around a repaired reference

The imitation path needed a reference the robot could at least represent. The original pilot failed during the approach, before the intended jump. I checked the stored motion and native environment, then tested changes to learning rather than declaring the robot incapable.

Evidence: P09 D0; S6 admission

## Slide 20 — A complete imitation pass required four things together

The logs call this the L1 gate. I use full test in the slides. It is a conjunction: finishing without a fall or unintended body contact, enough physical flight, adequate recovery speed, and sufficiently accurate joint tracking. For the original c2 reference, physical flight must last at least 0.24 seconds. Mean joint RMS must stay at or below 0.15 radians, roughly nine degrees, with the higher-percentile bound also met. A row needs sixteen full passes out of twenty. The command route uses a different test, so its 0.12-second jump threshold must not be substituted here.

Evidence: P09 unchanged s6_eval.py; P03/P09 gate text

## Slide 21 — I tested three explanations for the early tracking failure

The original pilot ended before reaching the intended jump. I did not treat that as proof that the jump was physically impossible. I tested three learning explanations separately. A1 widened and then restored a tracking tolerance. A2 changed which reference phases training episodes began from. A3 increased an existing speed-tracking reward. Each was a fresh single-seed training attempt, with its one intended scientific change plus read-only diagnostics.

Evidence: P09 A1–A3 manifests; D0 diagnosis

## Slide 22 — I changed where training episodes started

I split the starts into four equal components. This was a training-only change. The evaluation still began at frame zero with the original termination threshold. A2 was the first attempt to complete the clip, but this is still a single-seed observation.

Evidence: P09 A2 manifest; p09_cfgs_v2_copy.py

## Slide 23 — Five attempts finished; none passed the full test gate

A1 relaxed the training termination threshold and then tightened it. A2 changed start-phase sampling. A3 increased velocity reward weight. A4 combined the A2 configuration with the velocity change and extra training. A5 changed the reference under the original sampler. Only A2 and A4 completed the clip; every full L1 count remained zero.

Evidence: P09 A1–A5 raw JSONs

## Slide 24 — I monitored the learning curves, not only the final score

These are the actual training curves, not the final evaluation scores. A2 and resumed A4 developed longer episodes. A1 became shorter again as its tolerance tightened. A3 and the c3 variant remained short. The x-axis starts at the first saved diagnostic for each row; A4 already inherited A2 training, so these are not matched fresh starts. Training uses varied reference starts, pushes and randomization, so episode duration is not a full skill pass.

Evidence: P09 A1–A5/train_diag_every250.json

## Slide 25 — I also tracked error during training

The left graph is episode-mean training joint RMS, and the right graph is anchor velocity error at the last step. I do not use these as L1 passes or compare them directly with the strict evaluation values. A4 is a resumed policy and A5 changes reference. A1 did not log the new joint RMS metric, so it is omitted rather than filled in. Each curve comes from its saved native-training diagnostics.

Evidence: P09 A2–A5/train_diag_every250.json

## Slide 26 — A1 episodes shortened as I restored strict tolerance

A1 started with a wider end-effector termination tolerance, then tightened it to the original setting. The duration curve fell sharply around the return to strict tolerance. This describes one training run; it does not prove a unique cause. The final strict frame-zero evaluation still failed all twenty episodes. The data were already logged; I did not rerun training to make this graph.

Evidence: P09 A1/train_diag_every250.json

## Slide 27 — I saw progress in recovery and joint tracking

A4 recovered at the required speed in all twenty episodes. Eighteen met the joint-error gate, compared with zero for A2. The no-fall count stayed eighteen. Flight remained zero. Sixteen episodes met all non-flight gates together; I do not count them as full passes.

Evidence: P09 A2/A4 raw JSONs

## Slide 28 — Flight duration remained the binding common failure

A2 flew for 0.200 to 0.225 seconds. A4 flew for 0.195 to 0.215 seconds. The R1 reference geometry gives about 0.22 seconds of airborne time, while the physical gate is 0.24. That suggests a tension between close tracking and the flight requirement. Geometric reference flight and physical force-based flight are different measurements, so this does not prove impossibility.

Evidence: P09 A2/A4 raw; S6 reference admission

## Slide 29 — I kept the extra-training confound visible

The result is encouraging as engineering progress, but it is not an isolated test of the velocity reward. A4 had more total updates and a resumed training lineage. I need matched-budget, multi-seed tests before making a causal claim.

Evidence: P09 A2/A4 manifests and raw

## Slide 30 — I tried one other reference and kept one slot unused

The alternate c3 reference still failed early under the original configuration. I did not train c3 with the successful start mixture in this sweep. I also did not manufacture a slower reference just to fill the sixth slot.

Evidence: P09 A5 raw/admission; A6 README

## Slide 31 — The failed attempts also narrowed the next question

The negative results help locate what still needs testing. Simply relaxing termination and returning to strict settings was not enough. Simply increasing the speed reward was not enough. A different reference under the original configuration also failed early. The phase-mixture attempts made the largest observed progress on c2, but more seeds and matched budgets are needed before I turn that observation into a general causal claim.

Evidence: P09 A1/A3/A5 raw; final report

## Slide 32 — I stopped where the evidence stopped

The most important boundary is that all of this is simulation. I have not validated these policies on hardware. I also do not have a new rollout video from this sweep. The plotted evidence comes from the saved numeric evaluations.

Evidence: P08/P09 final reports; data handling rules

## Slide 33 — I would discuss three next tests with Prof. Sun

My first question is whether we should prioritize repairing the reference’s flight or transferring the start mixture to c3. I would then compare matched training budgets with more seeds. I am presenting these as choices for discussion, not completed work or permission to run them.

Evidence: P09 final report: untested options

## Slide 34 — My Week 32 result: progress, but no full imitation pass

I can now separate three things clearly: a command policy that jumps nominally, a tracker that completes the clip, and the full skill contract that remains unmet. That separation is the useful result of the week.

Evidence: P08 S5; P09 final/raw

## Slide 35 — Evidence and reproducibility

The supporting files preserve the source of every result slide. I can reproduce the charts without new training or simulation. The presentation notes and box-by-box companion contain the definitions that would be too dense for the main slides.

Evidence: P08/P09 saved artifacts; snapshot recorded in README

## Q&A preparation

**Did the tracker pass?** No. A4 is 0/20 full L1. Sixteen meet all non-flight gates together; no full pass is inferred.

**Did velocity weighting cause the gain?** Not established. A4 has extra training, resumed state and a new seed.

**Why no new video?** This sweep produced numeric evaluations but no rendered rollout. No unrelated old video is substituted.

**Does the 0.22-second reference make 0.24 seconds impossible?** No such proof. Geometric reference flight and physical force-based flight differ; this is a feasibility warning.

**Which policy stays final?** model_20496 is the original command-policy fallback. A4 is only the best failed c2 tracking candidate, not a replacement.