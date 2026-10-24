# Week 31 slide brief -- FOR THE ROG

Written by the workstation, 2026-10-24. Shaoqin asked for **a package the ROG builds the Week 31 slides from**. The ROG
builds the deck; the workstation only supplies content, numbers and media (role split, set 2026-10-09).
Meeting: **Saturday 2026-10-24, 9:30 am**, with Prof. Sun.

## What to build

The same deliverables as Week 30 (`Presentations/Shaoqin-Li/Week 30/` in the CUNY repo), into `Week 31/`:
`Embodied AI Week 31.pptx` (speaker script in the slide notes), `week31_speaker_script.md`,
`week31_box_by_box.html`, `README.md`.

Style and voice: copy the Week 30 deck (dark navy section slides with a gold bar, light slides with rounded mint/sand/stone
cards, Calibri) and the accepted experiment-deck recipe: first person and past tense, no equations, real terminal
output only, embedded robot video, an honesty slide, render to PNG and look at every slide before delivering.

The slide list below is a **proposal with all the wording and numbers checked against the results READMEs**. Re-layout,
merge or cut freely; do not change the numbers without checking the source README. About 20 minutes at 24 slides.

## Storyline (confirmed by Shaoqin 2026-10-20)

paper submitted -> switching probe -> walk/run video -> jump status -> question for Prof. Sun about logging.

Paper status, in Shaoqin's words (2026-10-23): *"submitted, but I can always add more for future submission"*.
So: submitted to L4DC 2027; this week's walk and jumping-jacks results can strengthen a later version.

## Where the files are

Everything is in the CUNY repo under `Presentations/Shaoqin-Li/Video-to-R1 Pipeline/` (paths below are relative to that
folder). Download a file with
`gh api "repos/SunCCNY/Embodied-AI/contents/Presentations/Shaoqin-Li/Video-to-R1%20Pipeline/<path>" -H "Accept: application/vnd.github.raw" > <file>`
(URL-encode spaces as `%20`).

| folder | what |
|---|---|
| `results/week31_slide_assets/` | **new for these slides**: 2 charts, 3 terminal panels, 6 video poster frames, `make_assets.py` |
| `results/hub_switch_probe_2026-10-20/` | Part 1: switching test README, 2 videos, frame strip |
| `results/walk_run_eval_2026-10-22/` | Part 2: walk/run README, 2 videos, ONNX, deploy_v1 fix |
| `results/walk_run_delay_2026-10-23/` | Part 2: delay fine-tune + control README, 1 video, sim2sim tables |
| `results/jump_clips_2026-10-23/` | Part 3: jump README, 2 side-by-side videos, stills, sweeps |
| `paper/l4dc2027/figures/fig_delay_margin.png` | the paper's main figure |

The READMEs in those folders are the source of truth for every number. Text copies are in the brain mirror under
`projects/embodied_ai_r1/video_to_r1_pipeline/results/`.

**About the new assets**
- Charts: survival vs. command delay. Colors match the paper figure (orange = no delay in training, blue = delay-trained;
  aqua = the original walking policy). Palette checked with the dataviz validator; aqua is low-contrast, so every line
  carries a direct label. Regenerate with `make_assets.py` if you restyle (needs the workstation for the terminal panels).
- Terminal panels are real output of the commands shown in them, run on the workstation on 2026-10-23. Do not edit their
  text.
- Poster frames are single frames from the videos, for PowerPoint's poster image.

## Headline numbers

- Switching (20 runs each): direct switch 20/20 almost everywhere, but tennis cut at 70 % -> dance **0/20**; going
  through the default stand **1/20** (dance -> tennis); the PD-held default stand tips over in about 3 s.
- Walk/run (Unitree's `unitree_rl_mjlab`, 4,096 robots, 10,001 iterations, 4 h 50 min): tracks -0.5 to 2.0 m/s within
  about 5 %, no falls; 2.0 m/s is a **fast walk** (both feet off the ground 2.7 % of the time); stands at zero command
  (192 robots, 0 falls, < 5 mm drift).
- Bug: Unitree's shipped R1 deploy file uses action scale 0.25 for every joint; training used 0.15-0.4125 per group. With
  the shipped file the robot falls in every walking run within ~1.5 s. Fixed copy = deploy v1.
- Delay (CPU check, survived of 50): at 60 ms base **0**, control **0**, delay-trained **50**; delay-trained 46/50 at 80 ms.
  Cost: 2.0 m/s command -> 1.85 m/s.
- Jumping jacks (of 20): plain 2 at 40 ms, delay-trained 20 at 40 and 50 ms, 19 at 60; pushes x3 11/20.
- Jump rope fails: the retargeted reference already has the arms out; GVHMR vs GMR not yet isolated.


## Slide by slide

### Slide 1 -- Title  (0.3 min)

- **On the slide:**
  - WEEK 31
  - Walk, run and jump: the first R1 skills
  - plus: switching between two skills, and the paper went to L4DC
  - tests run 2026-10-20 to 2026-10-23 -- simulation only
- **Say:** Last week I proposed a bag of R1 skills that I can chain in a MuJoCo game. Prof. Sun set the first milestone: walk, run and jump, connected smoothly. This week I worked on exactly that. Everything you will see is simulation; nothing ran on the real R1, because the safety gantry has not arrived.

### Slide 2 -- What I set out to answer  (0.8 min)

- **Title:** What I set out to answer
- **Subtitle:** Three questions this week
- **On the slide:**
  - 1. Can one trained motion hand over to another?
  - The game needs the R1 to switch between skills.
    Before training anything new, I tested a switch between two policies I already had: tennis and dance.
  - 2. Can the R1 walk and run on command?
  - I trained a policy that follows a speed command and stands still at zero.
    Then I tested how much command delay it survives, the same test as in my paper.
  - 3. Can a jump come from a video?
  - I sent two jump clips through my video-to-R1 pipeline without changing it, and checked the result in a second simulator.
- **Say:** Three questions. One: can one trained motion hand over to another, because the game needs switching. Two: can the R1 walk and run when I give it a speed, and does that survive command delay. Three: can a jump come out of my video pipeline. I'll go through them in this order.

### Slide 3 -- The paper went to L4DC  (1.0 min)

- **Title:** The paper went to L4DC
- **Subtitle:** Submitted; this week's results can go into a later, longer version
- **On the slide:**
  - How Much Command Delay Can a Learned Humanoid Motion Tracker Tolerate?
  - Five R1 motions from video, three training seeds each, tested in a second simulator.
    At 40 ms of command delay, survival fell to 15 % on average. With a random 0-50 ms delay in training it was 95 %.
    The same amount of extra training without delay did not help, so the gain comes from the delay itself.
    This week's walk and jumping-jacks results follow the same pattern.
  - Figure from the paper: survival vs. command delay for each motion. Orange = no delay in training, blue = random 0-50 ms delay in training; the shaded band is the training range.
- **Image:** `paper/l4dc2027/figures/fig_delay_margin.png`
- **Say:** The paper is submitted to L4DC. Its question: how much command delay can a learned motion tracker take? Five motions from video, three seeds each, checked in a second simulator. With 40 milliseconds of delay, survival drops to 15 percent; training with a random delay of up to 50 milliseconds lifts it to 95. A control arm with the same extra training but no delay did not improve. The figure shows all five motions: orange falls off a cliff around 30 to 40 milliseconds, blue holds to about 60. What you will see later today, the walk and the jumping jacks, behave the same way, so they can strengthen a later version of the paper.

### Slide 4 -- Switching between two skills  (0.2 min)

- **Title:** PART 1
- **Subtitle:** Switching between two skills
- **On the slide:**
  - Two policies I already had: tennis and dance
  - no new training -- a quick check before building anything
- **Say:** Part one. Before training new skills, I wanted to know whether switching is even the hard part.

### Slide 5 -- How I tested a switch  (1.0 min)

- **Title:** How I tested a switch
- **Subtitle:** Same robot, same physics; only the policy in charge changes
- **On the slide:**
  - 1. Run A
  - Policy A (tennis or dance) plays its motion.
  - 2. Cut
  - Stop A at the end of its motion, or 30 %, 50 % or 70 % of the way through.
  - 3. Hand over
  - Policy B takes over from exactly that robot state. Its motion is moved to where the robot is standing.
  - 4. Count
  - 20 runs per case. A run counts if the robot gets through the switch and all of B without falling.
  - Three ways to hand over
  - Direct
  - B's first frame at once.
  - 1 s ramp
  - Blend the target from where the robot is to B's first frame over one second.
  - Via the stand
  - Go to the default standing pose first, then start B.
- **Say:** Here is the test. Policy A, tennis or dance, plays its motion. I stop it at the end, or 30, 50 or 70 percent of the way through. Then policy B takes over from exactly that state; I only move its motion so it starts where the robot is standing. 20 runs per case, and a run counts only if the robot gets through the switch and all of motion B. I tried three ways to hand over: a direct switch, a one-second blend, and going through the default standing pose first.

### Slide 6 -- Tennis to dance, direct switch  (0.8 min)

- **Title:** Tennis to dance, direct switch
- **Subtitle:** One of the 20 runs, cut at the end of tennis
- **On the slide:**
  - The label in the corner shows which policy is in charge. 32 s, plain MuJoCo.
- **Video:** `results/hub_switch_probe_2026-10-20/videos/tennis_to_dance_direct.mp4` (poster frame: `results/week31_slide_assets/poster_probe_tennis_to_dance.png`)
- **Say:** This is one direct switch. Watch the label: tennis is in charge, then dance takes over in one step. At the switch the joints are two to three radians away from where dance wants them, yet the robot recovers and dances to the end.

### Slide 7 -- Did the switch survive?  (1.2 min)

- **Title:** Did the switch survive?
- **Subtitle:** Runs out of 20. Each policy alone survives 20 of 20.
- **On the slide:**
  - from -> to
  - how I handed over
  - at A's end
  - cut 30 %
  - cut 50 %
  - cut 70 %
  - tennis -> dance
  - direct
  - 20
  - 18
  - 18
  - 0
  - tennis -> dance
  - 1 s ramp
  - 19
  - 20
  - 20
  - 19
  - tennis -> dance
  - via the stand
  - 18
  - 20
  - 19
  - 19
  - dance -> tennis
  - direct
  - 20
  - 20
  - 20
  - 20
  - dance -> tennis
  - 1 s ramp
  - 20
  - 20
  - 11
  - 18
  - dance -> tennis
  - via the stand
  - 1
  - 8
  - 7
  - 2
  - The direct switch failed in one place: tennis cut at 70 %, with the arm up in the serve, 0 of 20. The blend fixed that case but lost elsewhere. Going through the standing pose was the worst way, down to 1 of 20.
- **Say:** Here are all the numbers. The direct switch survived almost everywhere, except one case: tennis cut at 70 percent, when the arm is up in the serve, then zero out of 20. The one-second blend fixed that case but dropped to 11 in another. And the idea I expected to be safest, going through the standing pose, was the worst: one out of 20 from dance to tennis.

### Slide 8 -- What the switch test told me  (1.0 min)

- **Title:** What the switch test told me
- **Subtitle:** Four findings that shaped the rest of the week
- **On the slide:**
  - The motions do not start or end standing. Joints are 0.6 to 1.9 rad away from the default stand at the first and last frame. If every skill should start and end at the stand, the clips must start and end standing.
  - A direct switch mostly works, but not always. It failed when the arm was up mid-serve (a 4.5 rad gap).
  - Smoothing the hand-over made it worse. The policies never saw ramps or pauses in training, so a blended target is new to them.
  - Holding the stand with plain motor springs tips the robot over in about 3 s. The stand in the middle of the skill bag has to be a real balancing policy. That is Part 2.
- **Say:** Four things. First, neither motion starts or ends at the standing pose, so clips for the skill bag should start and end standing. Second, a direct switch usually works, but not when the robot is in an extreme pose. Third, smoothing does not help, because the policies were never trained on blended targets. Fourth, and most important: just holding the standing pose with the motor springs tips the robot over in about three seconds. So the stand between skills has to be a policy that balances. That led straight to part two.

### Slide 9 -- Walk and run on command  (0.2 min)

- **Title:** PART 2
- **Subtitle:** Walk and run on command
- **On the slide:**
  - A policy that follows a speed command, and stands at zero
  - trained on the RTX 5080 workstation
- **Say:** Part two: walking and running, and standing still when the command is zero.

### Slide 10 -- Where the walking policy came from  (1.1 min)

- **Title:** Where the walking policy came from
- **Subtitle:** Unitree's own open-source training code, trained by me
- **On the slide:**
  - I did not write a walking task from scratch. Unitree's open-source repo unitree_rl_mjlab has an R1 task that follows a speed command. I set it up in its own environment and trained it: 4,096 robots in parallel, 10,001 iterations, 4 h 50 min.
    Halfway through, the task raises the top forward speed from 1 m/s to 2 m/s. So one run gave me a walk (iteration 5,000) and a faster version (iteration 10,000).
    Two earlier launches failed: one stopped when I closed the session, one trained a single robot because I left out the 4,096 setting. I now check the robot count in the log right after launch.
- **Image:** `results/week31_slide_assets/term_walkrun_train.png`
- **Say:** I did not write this task myself. Unitree publishes an open-source training repo with an R1 task that follows a speed command. I set it up in its own environment, because it needs older library versions than my tracking work, and trained it on the 5080: 4,096 robots, 10,001 iterations, just under five hours. The task raises the top speed from 1 to 2 meters per second halfway through, so one run gave me both a walk and a faster version. The terminal shows the real log: 4,096 environments, and at the last iteration nobody falls. Honestly, it took three launches: the first died when I closed the session, the second trained one robot because I dropped a flag.

### Slide 11 -- Walk and faster walk  (1.0 min)

- **Title:** Walk and faster walk
- **Subtitle:** Rendered in the training simulator, speed command changes every 4 s
- **On the slide:**
  - Walk (iteration 5,000)
    stand -> 0.5 -> 1.0 m/s -> stop, 16 s
  - Faster (iteration 10,000)
    stand -> 0.5 -> 1.0 -> 1.5 -> 2.0 m/s -> stop, 23 s
- **Video:** `results/walk_run_eval_2026-10-22/walk_model5000.mp4` (poster frame: `results/week31_slide_assets/poster_walk_model5000.png`)
- **Video:** `results/walk_run_eval_2026-10-22/run_model10000.mp4` (poster frame: `results/week31_slide_assets/poster_run_model10000.png`)
- **Say:** On the left, the walk: it stands, walks at half a meter per second, then one meter per second, then stops. On the right, the later checkpoint goes up to two meters per second and stops again. Near the end it seems to turn toward the camera. That is slow heading drift, about one and a half degrees per second, because nothing tells it to hold a heading. It is not a spin.

### Slide 12 -- How well it follows the speed command  (1.2 min)

- **Title:** How well it follows the speed command
- **Subtitle:** 64 robots per speed, average over the last 16 s of 20 s
- **On the slide:**
  - command (m/s)
  - walk (it. 5,000)
  - faster (it. 10,000)
  - -0.5 (backwards)
  - --
  - -0.43
  - 0 (stand)
  - 0.00
  - 0.00
  - 0.5
  - 0.47
  - 0.45
  - 1.0
  - 0.96
  - 0.96
  - 1.5
  - 1.44
  - 1.51
  - 2.0
  - 37 of 64 fell
  - 1.98
  - 2.0 m/s is a fast walk, not a run.
    Both feet are off the ground only 2.7 % of the time. In a human run it is about 30 % or more.
    It stops without falling.
    From 2.0 m/s it needs 0.86 m to stop; from 1.0 m/s, 0.40 m.
    At zero it stands still.
    192 robots, none fell, they drifted less than 5 mm in 16 s. This is the stand the switch test asked for.
- **Say:** The numbers. Both checkpoints follow the command within about five percent. The walk was never trained past one meter per second, so at two, 37 of 64 robots fall; the later checkpoint reaches 1.98. Two honest points on the right. It is not really running: both feet are in the air only 2.7 percent of the time, a human run is around 30. So I call it a fast walk. And at zero command it stands still, 192 robots, no falls, less than five millimeters of drift. That is the balancing stand the switch test asked for.

### Slide 13 -- A bug in Unitree's deploy file  (1.1 min)

- **Title:** A bug in Unitree's deploy file
- **Subtitle:** Found when I fed the policy the way the real robot's code does
- **On the slide:**
  - I wrote a second check that runs the policy in plain CPU MuJoCo and builds its inputs exactly like Unitree's C++ deploy code. Against the training simulator, the policy outputs match to 0.000002.
    With the deploy file Unitree ships, the robot fell in every walking run within about 1.5 s.
    The file scales every joint command by 0.25. Training scaled each joint group differently.
    I made a corrected copy (v1). For a first test on the real robot I would also cap the speed at 1 m/s.
  - joint group
  - shipped
  - trained
  - legs and waist
  - 0.25
  - 0.15
  - ankles
  - 0.25
  - 0.3125
  - shoulder pitch, roll
  - 0.25
  - 0.375
  - shoulder yaw, elbow, wrist
  - 0.25
  - 0.4125
  - How much one unit of policy output moves the joint target, per joint group.
- **Say:** Before trusting this on a real robot, I wanted the policy to run the way the robot runs it. I wrote a CPU check in plain MuJoCo that builds the inputs exactly like Unitree's C++ deploy code, and confirmed it matches the training simulator. Then I found a bug in Unitree's shipped deploy file: it scales every joint command by 0.25, but training used a different scale per joint group, from 0.15 for the legs to 0.41 for the arms. With the shipped file the robot falls in every walking run within a second and a half. My corrected copy is ready for when we are allowed to try the real robot.

### Slide 14 -- Standing was the weak skill under delay  (1.1 min)

- **Title:** Standing was the weak skill under delay
- **Subtitle:** So I applied the paper's recipe to the walking policy
- **On the slide:**
  - Standing (command 0), 10 runs per delay. Columns: delay in ms, runs survived, average speed.
  - With 40 ms of command delay, the walking policy fell while standing in 4 of 10 runs: a slow wobble that grows. Walking held up better than standing.
    Delay arm: 3,000 more iterations with a random 0-50 ms delay, as in the paper.
    Control arm: the same 3,000 iterations without delay, so I can tell what the delay did and what extra training did.
    One fix on the way: the training library gave each joint group its own random delay. I made all 24 joints share one delay and checked it.
- **Image:** `results/week31_slide_assets/term_stand_delay.png`
- **Say:** In that CPU check, the weakest skill under delay was, surprisingly, standing. At 40 milliseconds the base policy fell in four of ten standing runs. In the terminal, top block, the base policy at zero speed; bottom block, the delay-trained one. So I applied the paper's recipe: 3,000 more iterations with a random delay up to 50 milliseconds, and a control with the same 3,000 iterations and no delay. One thing I had to fix on the way: this version of the training library gave each joint group its own random delay, which a real robot would not do, so I made all 24 joints share one.

### Slide 15 -- Delay training moved the limit past 60 ms  (1.0 min)

- **Title:** Delay training moved the limit past 60 ms
- **Subtitle:** Robots that survived out of 50 (5 speeds x 10 runs), CPU check simulator
- **On the slide:**
  - Base and control fail between 40 and 60 ms.
    The 3,000 extra plain iterations did not move the limit.
    Delay-trained: 50 of 50 up to 60 ms, 46 at 80 ms.
    Its only failures are standing at 80 ms, 6 of 10 survived.
- **Image:** `results/week31_slide_assets/chart_walkrun_delay.png`
- **Say:** Here is the result. Each point is 50 runs, five speeds times ten. Green is the original, orange is the control with 3,000 extra iterations, blue is the delay-trained policy. Green and orange are nearly identical and fall apart between 40 and 60 milliseconds, so extra training alone does nothing. Blue survives everything up to 60 milliseconds and 46 of 50 at 80; the only failures are standing at 80. Same pattern as the paper, now on a walking policy.

### Slide 16 -- The delay-trained walk  (0.9 min)

- **Title:** The delay-trained walk
- **Subtitle:** Speed command 0 -> 0.5 -> 1.0 -> 1.5 -> 2.0 -> 0 m/s, 4 s each
- **On the slide:**
  - What it costs
    At a 2.0 m/s command it reaches 1.85 m/s, 7 % slow.
    Backwards at -0.5 m/s it reaches -0.39.
    From 0 to 1.0 m/s it matches the others.
    This is now my best candidate for the stand in the middle of the skill bag.
- **Video:** `results/walk_run_delay_2026-10-23/walk_run_delay_model12999.mp4` (poster frame: `results/week31_slide_assets/poster_walk_run_delay.png`)
- **Say:** This is the delay-trained policy, going from standing up to two meters per second and back, no falls. It is not free: at a two meter per second command it reaches 1.85, about seven percent slow, and backwards is a bit slower too. Up to one meter per second it matches the others. Because it stands with a 60 millisecond margin, this is now my candidate for the stand in the middle of the skill bag.

### Slide 17 -- A jump from a video  (0.2 min)

- **Title:** PART 3
- **Subtitle:** A jump from a video
- **On the slide:**
  - Two stock clips through the unchanged pipeline
  - video -> GVHMR -> GMR -> mjlab training -> plain MuJoCo check
- **Say:** Part three: a jump, from a video, through the same pipeline as before.

### Slide 18 -- Picking a clip  (0.9 min)

- **Title:** Picking a clip
- **Subtitle:** I downloaded 11 Pexels clips and checked each one
- **On the slide:**
  - rejected
  - why
  - two "person jumping" clips
  - body cut off in every frame, camera moving
  - two jump-rope clips
  - framed from the waist up
  - gym clip
  - two people
  - trampoline
  - not flat ground
  - parkour leap
  - slow motion, so the timing is wrong
  - bed jumping
  - several people
  - Chosen
    Jumping jacks: one person, whole body in frame, fixed camera, real flight, and it starts and ends standing.
    Jump rope: fixed camera, small fast hops.
    I found no clean clip of a single vertical jump.
- **Say:** Picking the clip mattered more than anything else. I downloaded eleven clips and checked each. Most failed GVHMR's needs from last week: body cut off, moving camera, more than one person, not flat ground, or slow motion. Jumping jacks was the best fit, and it starts and ends standing, which the switch test said we need. I added jump rope as a second, harder one. I did not find a clean single vertical jump.

### Slide 19 -- Jumping jacks: video next to the R1  (0.9 min)

- **Title:** Jumping jacks: video next to the R1
- **Subtitle:** 10.4 s, about 10 jumping jacks (20 jumps)
- **On the slide:**
  - left: original video (Kindel Media, Pexels)   right: the delay-trained R1 policy in plain MuJoCo, frame-synced
- **Video:** `results/jump_clips_2026-10-23/videos/jumping_jacks_side_by_side.mp4` (poster frame: `results/week31_slide_assets/poster_jumping_jacks.png`)
- **Say:** Here it is: the original video on the left, the R1 on the right in plain MuJoCo, the second simulator, frame by frame. It really leaves the ground: both feet are in the air 20 percent of the time.

### Slide 20 -- Jumping jacks under delay  (0.9 min)

- **Title:** Jumping jacks under delay
- **Subtitle:** Runs that survived out of 20, plain MuJoCo with the training randomization
- **On the slide:**
  - Same pattern as the paper.
    Plain policy: 2 of 20 at 40 ms. Delay-trained: 20 of 20 up to 50 ms, 19 at 60 ms.
    Tracking: 0.59 rad average joint error
    with no randomization, close to the paper's motions.
    Weak point: pushes.
    Three times stronger pushes: 11 of 20. A push in the air cannot be corrected until the robot lands.
- **Image:** `results/week31_slide_assets/chart_jj_delay.png`
- **Say:** Under delay the jump behaves like the paper's motions: the plain policy falls apart at 40 milliseconds, two of twenty, and the delay-trained one holds 20 of 20 up to 50. Its weak point is pushes: only 11 of 20 survive three times stronger pushes, because a push during flight cannot be corrected until the feet are down again.

### Slide 21 -- Jump rope: the robot follows a wrong reference  (1.0 min)

- **Title:** Jump rope: the robot follows a wrong reference
- **Subtitle:** The problem enters before training
- **On the slide:**
  - video and R1 policy, same moment
  - the reference motion by itself: no physics, no policy
  - The person hops about 26 times. The robot barely leaves the ground, 0.5 % of the time, with its arms out to the sides and the torso leaning.
    I drew the reference motion by itself, with no physics and no policy. It already has the arms out. The policy is tracking a wrong target.
    So the error comes from GVHMR, from GMR, or from both. I have not yet found which one.
    For the skill bag I use jumping jacks.
- **Image:** `results/week31_slide_assets/poster_jump_rope.png`
- **Image:** `results/jump_clips_2026-10-23/stills/jump_rope_reference_frames_120_300_480.png`
- **Say:** Jump rope did not work. The man keeps his arms down and turns the rope with his wrists; the robot holds its arms out and barely hops. Is that the policy's fault? I drew the reference motion by itself, no physics, no policy, bottom left, and the arms are already out. So the policy is faithfully tracking a wrong target. The mistake is before training, in GVHMR, which has to read small fast wrist motion with handles in the hands, or in the retargeting, or both. I have not isolated it yet.

### Slide 22 -- What this does not show  (1.0 min)

- **Title:** What this does not show
- **Subtitle:** The honest slide
- **On the slide:**
  - Everything is simulation. Nothing ran on a real R1.
  - One training seed per arm for walk and run, and one per jump clip.
  - The switch test used two motions and policies trained without delay. "Survived" uses the training rules for a fall, which allow large joint error.
  - I have not yet tested switching from a motion policy into the walking policy's stand.
  - The "run" is a fast walk: both feet are in the air only 2.7 % of the time.
  - For the jump clips, the delay stage also adds 3,000 iterations. Only the walk control and the paper's seed study separate the delay from extra training.
  - The CPU check for walking has no pushes, sensor noise or rough ground.
- **Say:** What this does not show. All simulation. One seed per arm this week. The switch test is two motions with no-delay policies. The key hand-off, from a motion into the walking stand, is not tested yet. The run is really a fast walk. For the jumps, extra training and delay are mixed; the walking control and the paper separate them. And the walking check has no pushes or rough ground.

### Slide 23 -- Next  (1.0 min)

- **Title:** Next
- **Subtitle:** My plan, in order
- **On the slide:**
  - 1. Connect them
  - Switch from a motion into the delay-trained stand and back.
    Then walk -> jumping jacks -> walk in one MuJoCo scene, the first piece of the game.
  - 2. A better jump
  - Draw GVHMR's body over the jump-rope video to find which stage goes wrong.
    Find or record on my phone a clean vertical jump that starts and ends standing.
  - 3. Question for Prof. Sun
  - Before the gantry arrives, may I record the R1's own sensor stream (rt/lowstate) while it stays still, with no motion commands?
    The corrected deploy file is ready for later, speed capped at 1 m/s.
- **Say:** Next, in order. One, connect the pieces: switch from a motion into the delay-trained stand and back, then walk, jumping jacks, walk, in one MuJoCo scene. That is the first piece of the game. Two, a better jump: find which stage breaks jump rope, and get a clean vertical jump, possibly recorded on my phone. Three, a question for you, Professor: before the gantry arrives, may I record the robot's own sensor stream while it stays still, with no motion commands? That would give me real latency and sensor data to set the delay range from.

### Slide 24 -- References  (0.2 min)

- **Title:** References
- **Subtitle:** Everything used or cited in this deck, with links
- **On the slide:**
  - unitree_rl_mjlab  (R1 walk/run task and C++ deploy)
    Unitree Robotics, open source.
    Code: github.com/unitreerobotics/unitree_rl_mjlab
  - mjlab  (GPU training framework)
    MuJoCo-Warp, Isaac-Lab-style API; the tracking task follows BeyondMimic.
    Code: github.com/mujocolab/mjlab
  - GVHMR  (video to SMPL-X human motion)
    Shen et al., World-Grounded Human Motion Recovery via Gravity-View Coordinates. SIGGRAPH Asia 2024.
    Code: github.com/zju3dv/GVHMR
    Paper: arxiv.org/abs/2409.06662
  - GMR  (retargeting SMPL-X to the robot)
    Araujo et al., Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking.
    Code: github.com/YanjieZe/GMR
    Paper: arxiv.org/abs/2510.02252
  - My paper (L4DC 2027 submission)
    Li and Sun, How Much Command Delay Can a Learned Humanoid Motion Tracker Tolerate? Delay Randomization and Sim-to-Sim Transfer. Under review.
  - Video-to-R1 pipeline  (my code and all results in this deck)
    Results folders dated 2026-10-20 to 2026-10-23.
    Repo: github.com/SunCCNY/Embodied-AI/tree/main/Presentations/Shaoqin-Li/Video-to-R1%20Pipeline
  - Video clips  (Pexels, free to use, credit given)
    Elderly Man Doing Jumping Jacks Outside (Kindel Media); A Man Jumping Over The Skip Rope.
    Jumping jacks: www.pexels.com/video/elderly-man-doing-jumping-jacks-outside-7299359/
    Jump rope: www.pexels.com/video/a-man-jumping-over-the-skip-rope-4812842/
  - MuJoCo  (the second simulator)
    Google DeepMind physics engine.
    Code: github.com/google-deepmind/mujoco
- **Say:** References, with links, for anyone who wants to follow up.

## Q&A preparation

**Why use Unitree's walking task instead of a walk from video?**

A video walk is one fixed clip. The game needs a walk I can steer: any speed, stop, start. Unitree's task is trained for exactly that, and it already has C++ deploy code for the real R1. The video pipeline stays the source for the actions (jumps, later kicks and punches).

**Why is it not a real run?**

The task's rewards and its 0.6 s step timing were made for walking. Faster commands made it walk faster rather than run: both feet are in the air only 2.7 % of the time. A real run needs a different reward or a running reference, for example a running clip through the pipeline.

**Is 60 ms of delay realistic for the R1?**

I don't know yet. That is why I want to log the robot's own data: the delay range in training should be set from the measured latency. The paper makes the same point.

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

## Terms for the box-by-box companion

Plain-language meanings to reuse in `week31_box_by_box.html`:

- **policy**: the trained neural network that turns what the robot senses into motor targets, 50 times a second
- **simulator**: the physics program. Training runs in mjlab (GPU MuJoCo); the check runs in plain CPU MuJoCo, a second, independent setup
- **second simulator**: plain CPU MuJoCo, built separately from the training setup, so a policy that only works because of a training-simulator quirk shows up here
- **command delay**: the time between the policy deciding and the motors receiving the command. Real robots always have some; the check holds it constant for the whole run
- **delay**: short for command delay: how late the motor command arrives
- **iteration**: one round of training: all robots act for a short while, then the network is updated once
- **seed**: the random starting point of a training run. Different seeds give slightly different policies
- **control**: a comparison run trained the same extra amount but without the delay, so any difference comes from the delay
- **survive**: the robot finished the run without meeting the training's fall rules (body too low, tilted, or too far off the motion)
- **default stand**: the R1's standard standing joint angles, the same in all my tasks
- **motor springs**: PD control: each motor pulls toward a target angle like a spring with damping. With no policy on top, nothing keeps the robot balanced
- **GVHMR**: a published model that estimates a 3D human body motion from ordinary video
- **GMR**: a published retargeting tool that maps a human motion onto a robot's joints
- **mjlab**: the GPU training framework (MuJoCo-Warp) I train the R1 with
- **MuJoCo**: a physics engine for robots, from Google DeepMind
- **unitree_rl_mjlab**: Unitree's open-source training repo; it has the R1 speed-command task and C++ deploy code for the real robot
- **deploy**: the C++ program and settings file that run a trained policy on the real robot
- **rad**: radians; 1 rad is about 57 degrees
- **reference**: the target motion the policy is trained to follow, made from the video
- **rt/lowstate**: the R1's own data stream: joint angles, motor currents and the body sensor (IMU), published by the robot
- **gantry**: a safety frame that holds the robot by a harness, so it cannot fall during tests
- **L4DC**: Learning for Dynamics and Control, a conference on learning and control; submission deadline 2026-11-13
- **pushes**: random shoves during the run; x3 means three times the strength used in training
- **training randomization**: the random changes used in training (friction, mass, motor strength, sensor offsets), applied in the check too

## References (slide 24)

- unitree_rl_mjlab (R1 walk/run task and C++ deploy), Unitree Robotics: https://github.com/unitreerobotics/unitree_rl_mjlab
- mjlab (GPU training framework; the tracking task follows BeyondMimic): https://github.com/mujocolab/mjlab
- GVHMR, Shen et al., World-Grounded Human Motion Recovery via Gravity-View Coordinates, SIGGRAPH Asia 2024: https://github.com/zju3dv/GVHMR , https://arxiv.org/abs/2409.06662
- GMR, Araujo et al., Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking: https://github.com/YanjieZe/GMR , https://arxiv.org/abs/2510.02252
- My paper: Li and Sun, How Much Command Delay Can a Learned Humanoid Motion Tracker Tolerate? Delay Randomization and Sim-to-Sim Transfer. L4DC 2027 submission, under review.
- Video-to-R1 pipeline (my code and all results): https://github.com/SunCCNY/Embodied-AI/tree/main/Presentations/Shaoqin-Li/Video-to-R1%20Pipeline
- Pexels clips (free to use, credit given): Elderly Man Doing Jumping Jacks Outside (Kindel Media) https://www.pexels.com/video/elderly-man-doing-jumping-jacks-outside-7299359/ ; A Man Jumping Over The Skip Rope https://www.pexels.com/video/a-man-jumping-over-the-skip-rope-4812842/
- MuJoCo, Google DeepMind: https://github.com/google-deepmind/mujoco

## Presenter checks

- Open the pptx in PowerPoint (not a browser preview) so the five videos play: slides 6, 11 (two), 16, 19.
- Click each video once before the meeting; the first play can lag while PowerPoint loads it.
- If a video does not play, the same file is in the results folder of the Video-to-R1 Pipeline (README links below).
- Slide 23 has the question for Prof. Sun; leave time for his answer.

## Not decided / not done (do not present as done)

- Hand-off from a motion policy into the walking policy's stand: not tested.
- Jump rope: which stage (GVHMR or GMR) is wrong: not found yet.
- Unitree not yet told about the deploy-file bug.
- Real R1: nothing run; the logging question is for Prof. Sun at this meeting.
- One training seed per arm this week (walk/run) and per jump clip.
