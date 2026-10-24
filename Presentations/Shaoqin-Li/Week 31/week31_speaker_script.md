# Embodied AI Week 31 — speaker script

For the 24-slide deck `embodied ai week 31.pptx`. Suggested speaking time: about 20 minutes, including the five short videos. Leave time for Prof. Sun's answer on slide 23. All tests ran 2026-10-20 to 2026-10-23, in simulation only. Every number comes from the results READMEs in the Video-to-R1 Pipeline folder (club repo commit 8fb692e). Each result slide ends with a "Why it matters" line; read it out, because it is the one sentence to remember.

## Slide 1 — Week 31: walk, run and jump

**Time: 20 seconds.**

Last week I proposed a bag of R1 skills that I can chain in a MuJoCo game. You set the first milestone: walk, run and jump, connected smoothly. This week I worked on exactly that. Everything you will see is simulation; nothing ran on the real R1, because the safety gantry has not arrived.

## Slide 2 — What I set out to answer

**Time: 45 seconds.**

Three questions. One: can one trained motion hand over to another, because the game needs switching. Two: can the R1 walk and run when I give it a speed, and does that survive command delay. Three: can a jump come out of my video pipeline. Together they are the first milestone. Three words I will use all the time, at the bottom: a policy is the trained controller that turns what the robot senses into motor commands; command delay is a motor command that arrives late; and sim2sim means testing the same policy in a second, separate simulator. I'll go through the questions in this order.

## Slide 3 — The paper went to L4DC

**Time: 60 seconds.**

The paper is submitted to L4DC; the numbers on this slide are the paper's. Its question: how much command delay can a learned motion tracker take? Five motions from video, three seeds each, checked in a second simulator. With 40 milliseconds of delay, survival drops to 15 percent; training with a random delay of up to 50 milliseconds lifts it to 95. A control with the same extra training but no delay did not improve. In the figure, orange falls off around 30 to 40 milliseconds and blue holds to about 60. Why it matters: practising with delay is the fix I reuse this week, and the walk and the jumping jacks behave the same way, so they can strengthen a later version of the paper.

## Slide 4 — Part 1: switching between two skills

**Time: 10 seconds.**

Part one. Before training new skills, I wanted to know whether switching is even the hard part.

## Slide 5 — How I tested a switch

**Time: 60 seconds.**

Here is the test. Policy A, tennis or dance, plays its motion. I stop it at the end, or 30, 50 or 70 percent of the way through. Then policy B takes over from exactly that state; I only move its motion so it starts where the robot is standing. 20 runs per case, and a run counts only if the robot gets through the switch and all of motion B. I tried three ways to hand over: a direct switch, a one-second blend, and going through the default standing pose first.

## Slide 6 — Tennis to dance, direct switch

**Time: 45 seconds.**

This is one direct switch. Watch the label: tennis is in charge, then dance takes over in one step. At the switch the robot is far from the pose dance starts from, yet it recovers and dances to the end. Why it matters: a switch can work with no extra training at all.

## Slide 7 — Did the switch survive?

**Time: 70 seconds.**

Here are all the numbers; the pink cells are the failures. The direct switch survived almost everywhere, except one case: tennis cut at 70 percent, when the arm is up in the serve, then zero out of 20. The one-second blend raised that case to 19 of 20, but another case fell to 11. And the idea I expected to be safest, going through the standing pose, was the worst: one out of 20 from dance to tennis. Why it matters: in this test, where the switch happens matters. This is two motions and a few cut points, so it is a first look, not a rule.

## Slide 8 — What the switch test told me

**Time: 60 seconds.**

Four things. First, neither motion starts or ends at the standing pose, so I propose clips for the skill bag that start and end standing. Second, a direct switch usually works, but not when the robot is in an extreme pose. Third, the one-second ramp was inconsistent: it helped one case and hurt another, probably because the policies were never trained on blended targets. Fourth, and most important: in one test, holding the standing pose with only the motor springs tipped the robot over in about three seconds. So the stand between skills has to be a policy that balances. That led straight to part two.

## Slide 9 — Part 2: walk and run on command

**Time: 10 seconds.**

Part two: walking and running, and standing still when the command is zero.

## Slide 10 — Where the walking policy came from

**Time: 65 seconds.**

I did not write this task myself. Unitree publishes an open-source training repo with an R1 task that follows a speed command. I set it up in its own environment, because it needs older library versions than my tracking work, and trained it on the 5080: 4,096 simulated environments in parallel, 10,001 iterations, just under five hours. The task raises the top speed from 1 to 2 meters per second halfway through, so one run gave me both a walk and a faster version. The terminal shows the real log: 4,096 environments, and at the last iteration none of them falls. Honestly, it took three launches: the first died when I closed the session, the second trained one simulated environment because I dropped a setting.

## Slide 11 — Walk and faster walk

**Time: 60 seconds.**

On the left, the walk: it stands, walks at half a meter per second, then one meter per second, then stops. On the right, the later checkpoint goes up to two meters per second and stops again. Near the end it seems to turn toward the camera. That is slow heading drift, about one and a half degrees per second, because nothing tells it to hold a heading. It is not a spin. Why it matters: one policy covers standing, walking and walking fast, steered only by the speed I give it.

## Slide 12 — How well it follows the speed command

**Time: 70 seconds.**

The numbers. These are 64 simulated runs per speed in the training simulator. Within the tested command ranges, the largest speed gap is 0.07 meters per second. The walk checkpoint was not trained above one meter per second; at two, 37 of 64 runs fall. The later checkpoint reaches 1.98. On the right, three plain facts. It is not really running: both feet are in the air only 2.7 percent of the time, so I call it a fast walk. It stops without falling. And at zero command it stands still: 192 simulated runs, 64 for each checkpoint, no falls, less than five millimeters of drift. That is the balancing stand the switch test asked for. Backwards is the least accurate. Why it matters: within the commands it was trained to follow, this is close speed control for a game.

## Slide 13 — A bug in Unitree's deploy file

**Time: 65 seconds.**

I wanted to check the policy using the input wiring in Unitree's C++ deploy code. In the parity check, the exported ONNX actor and trained Torch actor differed by at most 0.000002; that compares actor outputs only, not the full deploy inputs or hardware. The check is documented in the walk/run evaluation README, section 5. Then I found a mismatch in Unitree's shipped deploy file: it scales every joint command by 0.25, but training used a different scale per joint group, from 0.15 for the legs to 0.41 for the arms. In this CPU simulation, with the shipped file, the policy falls in every walking run, usually within two seconds and always within three. Why it matters: the shipped settings failed in simulation, before any real test. My corrected copy is ready for a later real test.

## Slide 14 — Standing was the weak skill under delay

**Time: 65 seconds.**

In this CPU simulation, the weakest skill under command delay was standing. At 40 milliseconds the base policy fell in four of ten standing runs. In the terminal, the top block is the base policy at zero speed and the bottom block is the delay-trained one. So I applied the paper's recipe: 3,000 more iterations with a random delay up to 50 milliseconds, and a control with the same 3,000 iterations and no delay. The control is a matched comparison: same extra training, no delay. One thing I changed on the way: this version of the training library gave each joint group its own random delay, so I set the simulation up so all 24 joints share one delay, and checked it. Why it matters: this shows how the stand handled the tested delays in this CPU simulation; actual delay on the R1 is not measured here.

## Slide 15 — Delay training moved the limit past 60 ms

**Time: 60 seconds.**

Here is the result. Each point is 50 runs, five speeds times ten. Green is the original, orange is the control with 3,000 extra iterations, blue is the delay-trained policy. Green and orange are nearly identical and fall apart between 40 and 60 milliseconds, so extra training alone did not help here. Blue survives everything up to 60 milliseconds and 46 of 50 at 80; the only failures are standing at 80. Why it matters: this run's matched control supports delay training as the reason, the same pattern as the paper. It is one seed per arm, so more seeds are needed.

## Slide 16 — The delay-trained walk

**Time: 55 seconds.**

This is the delay-trained policy, going from standing up to two meters per second and back, with no falls. It is not free: at a two meter per second command it reaches 1.85, about seven and a half percent below, and backwards it reaches minus 0.39 for minus 0.5, about 22 percent below. Up to one meter per second it matches the others. Why it matters: that trade buys a 60 millisecond margin at stand, so this is now my best simulated candidate for the stand in the middle of the skill bag.

## Slide 17 — Part 3: a jump from a video

**Time: 10 seconds.**

Part three: a jump, from a video, through the same pipeline as before.

## Slide 18 — Picking a clip

**Time: 55 seconds.**

Picking the clip mattered more than anything else. I downloaded eleven clips and checked each. Most failed what GVHMR needs, as we saw last week: body cut off, a moving camera, more than one person, not flat ground, or slow motion. Jumping jacks was the best fit, and it starts and ends standing, which the switch test said we need. I added jump rope as a second, harder one. I did not find a clean single vertical jump.

## Slide 19 — Jumping jacks: video next to the R1

**Time: 50 seconds.**

Here it is: the original video on the left, the R1 on the right in plain MuJoCo, the second simulator, frame by frame. In the simulation it really leaves the ground: both feet are in the air 20 percent of the time. Why it matters: the bag now has a simulated jump with a flight phase that starts and ends standing.

## Slide 20 — Jumping jacks under delay

**Time: 55 seconds.**

This is the second simulator, with one fixed delay in each run plus the training's random changes. The plain policy falls apart at 40 milliseconds, two of twenty, and the delay-trained one holds 20 of 20 up to 50 and 19 at 60. Its weak point is pushes: with three times stronger pushes, 8 of 20 survive without delay training and 11 of 20 with it, because a push during flight cannot be corrected until the feet are down again. The average joint-angle error without random changes was about 34 degrees (0.59 rad). Why it matters: it improved in this run, like walking. But it is one seed, and the delay arm also got 3,000 extra iterations, so the cause is not isolated.

## Slide 21 — Jump rope: the robot follows a wrong reference

**Time: 60 seconds.**

Jump rope did not work. The man keeps his arms down and turns the rope with his wrists; the robot holds its arms out and barely hops. Is that the policy's fault? I drew the reference motion by itself, with no physics and no policy, at the bottom, and the arms are already out. So the policy is faithfully tracking a wrong target. The mistake is before training: in GVHMR, which has to read small fast wrist motion with handles in the hands, or in the retargeting, or both. I have not isolated it yet. Why it matters: check the copied motion first, before blaming training.

## Slide 22 — What this does not show

**Time: 60 seconds.**

What this does not show. All simulation. One seed per arm this week. The switch test is two motions with policies trained without delay. The key hand-off, from a motion into the walking stand, is not tested yet. The run is really a fast walk. For the jumps, extra training and delay are mixed; the walking control and the paper separate them. And the walking check has no pushes or rough ground.

## Slide 23 — Next

**Time: 60 seconds.**

Next, in order. One, connect the pieces: switch from a motion into the delay-trained stand and back, then walk, jumping jacks, walk, in one MuJoCo scene. That is the first piece of the game. Two, a better jump: find which stage breaks jump rope, and get a clean vertical jump, possibly recorded on my phone. Three, a question for you, Professor: before the gantry arrives, may I record the robot's own sensor stream while it stays still, with no motion commands? What that recording can tell us, and whether it is safe enough, are open questions I would like your view on.

## Slide 24 — References

**Time: 10 seconds.**

References, with links, for anyone who wants to follow up. The Pexels clips are credited to their creators and free to use.

## Q&A preparation

**Why use Unitree's walking task instead of a walk from video?**
A video walk is one fixed clip. The game needs a walk I can steer: any speed, stop, start. Unitree's task is trained for exactly that, and it already has C++ deploy code for the real R1. The video pipeline stays the source for the actions (jumps, later kicks and punches).

**Why is it not a real run?**
The task's rewards and its 0.6 s step timing were made for walking. Faster commands made it walk faster rather than run: both feet are in the air only 2.7 % of the time. A real run needs a different reward or a running reference, for example a running clip through the pipeline.

**Is 60 ms of delay realistic for the R1?**
I don't know yet. The delay range in training should come from a measured latency, and how to measure it on our R1 is still an open question. The paper makes the same point.

**Why did going through the standing pose fail?**
Two reasons. Holding the default pose with plain motor springs is not stable (it tips in about 3 s), and the motion policies never saw a long hold or a slow blend in training, so they react badly to it.

**How long does each skill take to train?**
Walk/run: 4 h 50 min for 10,001 iterations on the RTX 5080. The +3,000-iteration control took 1 h 52 min alone; the delay arm took 4 h 09 min while sharing the GPU with the jump runs. Jump clips: 2 h 12 min plain + 2 h 45 min with delay, both clips in parallel.

**Did you tell Unitree about the deploy bug?**
Not yet. I can open an issue on unitree_rl_mjlab with the per-joint scales and the test that shows the fall.

**Only one seed this week. Can we trust the delay result?**
For walking there is a matched control and the effect is large (0 of 50 vs 50 of 50 at 60 ms). The paper's seed study (5 motions x 3 seeds per arm) showed the same effect every time. More seeds are cheap to add.

**How will jump rope be fixed?**
First find the stage at fault: draw GVHMR's human body over the video. If GVHMR's arms are already out, it's the pose estimate; if not, it's the retargeting. A clean vertical-jump clip is the faster route to a jump skill.

## Evidence and sources

All numbers: the results READMEs in `Presentations/Shaoqin-Li/Video-to-R1 Pipeline/results/` of the club repo (commit 8fb692e): `hub_switch_probe_2026-10-20`, `walk_run_eval_2026-10-22`, `walk_run_delay_2026-10-23`, `jump_clips_2026-10-23`, and the slide brief `week31_slide_assets/README.md`. Charts, terminal panels and poster frames are the workstation's `week31_slide_assets` files, used unchanged; the terminal panels are real output of the commands shown.

One wording change from the brief, checked against the raw data: slide 13 says the robot fell "usually within 2 s, always within 3 s" instead of "within about 1.5 s". `walk_run_eval_2026-10-22/sim2sim_cpu/run_model10000.json`, shipped scale, walking commands: 40 of 40 runs fell at 0 ms delay, fall time 1.48 to 2.90 s (median 1.73 s); over all delays, 120 of 120 fell, median 1.54 s. The README's "~1.5 s" is that median.

## Presenter checks

- Open the pptx in PowerPoint (not a browser preview) so the five videos play: slides 6, 11 (two), 16 and 19.
- Click each video once before the meeting; the first play can lag while PowerPoint loads it.
- If a video does not play, the same file is in the results folder of the Video-to-R1 Pipeline.
- Slide 23 has the question for Prof. Sun; leave time for his answer.
