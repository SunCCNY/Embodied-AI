# Making a Robot Copy a Human

**An introduction to the GMR + Unitree R1 motion-retargeting project, from zero.**

You are assumed to know nothing about this project, this robot, or this research area. By the end of
this document you will know what the field is trying to do, what the two or three papers that matter
actually say, why this project exists, and — command by command — how to reproduce every result in
it on an ordinary laptop with no GPU.

There is no mathematics in this document. Where a paper's contribution is a piece of mathematics, it
is described in words, because the point here is to understand *what is being done and why*, not to
re-derive it.

| Part | | |
|---|---|---|
| **One** | The papers | what the field is doing, in plain language |
| **Two** | How this project started | why a small lab with one laptop ended up here |
| **Three** | The tutorial | every command, with the output it actually produced |

---

# PART ONE — THE PAPERS

## The problem all of these papers share

A humanoid robot has motors — the Unitree R1 has 24 of them — and to make it move you must decide,
many times a second, what angle each motor should hold. Deciding that by hand for anything more
interesting than standing still is impractical.

Humans, meanwhile, already know how to move. **Motion capture** records a person: a suit of sensors,
or a set of cameras, writes down where the person's hips, knees, elbows and hands were, a hundred or
more times a second. That data is cheap, plentiful, and full of exactly the behaviour we would like
a robot to have.

The catch is that a robot is not a person. Different limb lengths, different joints, different
number of ways to bend. You cannot copy a human's joint angles onto a robot, because the robot does
not have the same joints.

**Retargeting is the translation step.** For each frame of the human's motion, it asks: *what set of
angles for this robot's motors puts its body closest to where the human's body was?* Do that for
every frame and you have turned a recording of a person into a sequence of poses for a robot.

Every paper below is about that translation, or about what the translation leaves out.

---

## Paper 1 — GMR: "Retargeting Matters"

> **Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking**
> João Pedro Araújo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu — Stanford University
> arXiv:2510.02252 · **ICRA 2026** · open source, runs on CPU
> *This is the tool this project uses.*

### 1. Introduction & Summary

Most of the attention in humanoid learning goes to the policy — the neural network that actually
balances the robot and tracks a motion. Retargeting is treated as plumbing: a preprocessing step you
run once, get through, and stop thinking about.

This paper's argument is that the plumbing is load-bearing. The authors show that the *quality* of
the retargeted motion you train on changes how well the resulting policy works. Feed a policy a
reference full of small impossibilities — a foot that slides while it is supposed to be planted, a
limb that passes through the body, a pose the robot cannot physically reach — and the policy spends
its capacity learning to paper over them. The failures show up worst on exactly the motions you most
want: long ones, and fast ones.

Having made that case, they release **GMR**, a retargeting system built to produce cleaner
references, and they show it beats the other open-source options and comes close to the quality of
the expensive commercial motion data that robot manufacturers keep to themselves.

The one-sentence version: **the reference motion is not preprocessing, it is part of the method, and
treating it carelessly costs you policy performance.**

### 2. Core Challenges

**A robot is not a scaled-down person.** The obvious approach — shrink the human and copy the angles
— fails because proportions do not scale uniformly. A robot's arms may be short relative to its legs
in a way no human's are. Something has to give, and deciding *what* gives is the entire problem.

**Every choice is a trade-off between body parts.** You cannot generally put the hands, the feet,
the hips and the head all exactly where the human's were, because the robot's geometry will not
allow it. So retargeting has to be told what matters most. Put the feet exactly right and the hands
drift; insist on the hands and the robot's balance goes.

**It has to work for many different robots.** A method that only works for one machine is a
configuration, not a contribution. GMR's design problem is to make the robot-specific part small and
declarative — something you write down, rather than code you rewrite.

**It has to be fast enough to be useful.** If retargeting a motion takes hours, it is a batch
process. If it runs faster than the motion plays, it can sit inside a live teleoperation loop, and
that opens a completely different set of applications.

### 3. Framework — how it works, without the mathematics

GMR works **one frame at a time**. For a single frame of the human's motion it asks a solver: given
where this person's body parts are, find the robot joint angles that get the robot's corresponding
parts as close as possible — while respecting the robot's actual joint limits, so it never proposes
a pose the machine could not hold.

Three ideas do the real work.

**A correspondence list.** Somebody has to say *this human body part drives that robot body part* —
the human's left knee drives the robot's left knee, and so on. In GMR this is a plain text file, one
per (motion format, robot) pair. Adding a new robot means writing one of these files rather than
writing new code. **This file is where almost all of the robot-specific knowledge lives.**

**Weights, because you cannot have everything.** Each correspondence carries a number saying how
much it matters. The feet and the torso are weighted heavily; a wrist is weighted lightly. When the
robot's geometry forces a compromise, the weights decide who wins. This is how the trade-off in
challenge two is actually resolved — not by cleverness, but by a set of priorities somebody chose.

**Two passes per frame.** The first pass cares mostly about *orientation* — getting the shape of the
pose right, the angle of the torso, the swing of the limbs. The second cares mostly about
*position* — where in the world that pose sits. Splitting it this way gets a better answer than
asking for both at once.

Underneath, GMR uses an existing open-source solver rather than inventing one. GMR's contribution is
the framing: the correspondence files, the weights, the two passes, and the fact that it runs fast
on an ordinary CPU.

### 4. Results

The authors test on a large public motion-capture dataset, train tracking policies on the output,
and compare against the other open-source retargeters and against a commercial dataset that a robot
manufacturer produced in-house.

| compared against | result |
|---|---|
| other open-source retargeting tools | GMR's references consistently produce better-tracking policies |
| a manufacturer's closed commercial dataset | GMR gets close, on both tracking accuracy and faithfulness to the original motion |

The finding that matters more than the ranking is the *shape* of the difference: reference quality
matters most on **long and dynamic** motions. On a short, gentle clip almost anything works. The gap
opens exactly where the interesting behaviour is.

### 5. Core Values

**It makes retargeting a first-class research object.** Before this paper you could reasonably treat
retargeting as a solved detail. After it, "which retargeter did you use?" is a question a reviewer
can legitimately ask about a policy result.

**It is small, open and CPU-only.** No GPU cluster, no proprietary data. On the laptop used for this
project it retargets faster than the motion plays. That is what makes it usable by a student lab,
and it is the direct reason this project could do the work at all.

**Its robot-specific part is a text file** — which is what makes adding a new robot a weekend of
work rather than a rewrite, and is exactly the door this project walked through.

**A caveat worth carrying:** its own headline experiment is run on a motion dataset for which no
Unitree R1 correspondence file exists. Reproducing their comparison on this robot is not currently
possible without writing that file first.

---

## Paper 2 — KDMR: what retargeting is never asked

> **KDMR: Kinodynamic Motion Retargeting for Humanoid Locomotion**
> Xiaoyu Zhang, Steven Haener, Varun Madabushi, Maegan Tucker — Georgia Institute of Technology
> arXiv:2603.09956 · March 2026 · tested on the Unitree G1 · code not released

### 1. Introduction & Summary

Everything in the pipeline above rests on a quiet assumption: that a motion which *looks* right is a
motion the robot could actually perform. KDMR refuses that assumption and checks it.

Retargeting, as described above, is a **geometry** problem: put the parts where the parts were. But
walking is not a geometry problem. Nothing in the retargeter knows about weight, contact, or
friction. So it will cheerfully produce a robot gliding a centimetre above the floor, or with a foot
sliding sideways while planted, or with a leg passing through the ground — because none of those are
geometric errors. They are physics errors, and the retargeter was never asked about physics.

KDMR adds a second stage that does ask. After the usual pass, it re-solves the whole motion under
the constraint that a real body with real mass and real contact could have produced it.

One sentence: **the usual method answers "what pose"; KDMR adds the missing answer to "and could the
robot hold it".**

### 2. Core Challenges

The paper is organised around failure modes, and all of them are things you can see in your own
output within minutes:

- **Floating** — the sole never reaches the ground, because nothing said it had to.
- **Ground penetration** — the same error with the opposite sign; the floor is not part of a pure
  geometry problem, so the foot goes through it.
- **Foot sliding** — the foot is nominally planted but drifts sideways between frames. To a viewer
  it reads instantly as fake; to a policy trying to imitate it, it is an instruction to break
  friction.
- **Transitions no motor could make** — two poses that are each reachable, but not one after the
  other at that speed.
- **The lost heel-to-toe roll** — humans roll through the foot during a step. Treating the foot as
  one rigid target flattens that, and the gait looks robotic even when every joint angle is fine.

Underneath all five is one fact: **these artifacts are invisible to a geometric objective.** You
cannot tune your way out of them by adjusting weights, because the thing being optimised does not
contain the relevant physics at all.

An analogy that holds: a flipbook animator can draw a person mid-leap on every page and the sequence
reads as a leap. A stunt coordinator has to answer where the force comes from, which foot is loaded,
and whether the landing is survivable. Ordinary retargeting is the flipbook.

### 3. Framework

Two stages. The first is a conventional geometric pass, used deliberately as a starting guess rather
than as the answer. The second re-solves the entire motion at once — not frame by frame — under
constraints that encode physical reality: a planted foot may not slide, a foot in contact must push
*down* on the floor rather than being pulled up by it, forces must stay inside what friction can
provide, and the whole trajectory must be consistent with the robot's mass.

The practical price is what those constraints need as input: the method uses recorded ground
reaction forces, which come from instrumented treadmills or force plates, and its contact reasoning
assumes steady walking.

### 4. Results

The output keeps the human's gait style but now has real contact events and no floating. A policy
trained to imitate the corrected motion learns faster and reaches a higher final score than the same
policy trained on the uncorrected one — which is the same claim GMR makes, arrived at from the other
direction.

### 5. Core Values

**It names the artifact and gives it a method.** "The robot floats" stops being an observation and
becomes a known failure mode with a published fix.

**It is optimisation, not learning** — no GPU, no training run. That makes it reachable for a small
lab, and it does not overlap with the reinforcement-learning work happening elsewhere.

**Its limitation is an opening.** It needs force plates. Inferring the same information from
kinematics alone is named by the authors as future work — and a cheap, non-learned check that needs
nothing but the motion file is exactly that gap, sized for a student project.

---

## Two more, briefly

**TWIST — Teleoperated Whole-Body Imitation System** (arXiv:2505.02833, same Stanford group). A
person wearing motion capture drives a humanoid in real time. GMR was originally released as the
retargeting component of this system, which is why it is built to run faster than the motion plays
rather than as an offline batch tool. Worth knowing as the reason GMR is fast.

**NMR — "Make Tracking Easy"** (arXiv:2603.22201). Its useful observation for us: the per-frame
solve has **more than one valid answer**, and which one you land on depends on where the solver
started. Run the same motion twice from different starting points and you can get visibly different
joint solutions, including sudden jumps between frames. They fix this with a learned refinement
stage. **This project ran into the same effect by accident** — see Part Three, step 11.

## Where the field is, as of September 2026

Two things are worth knowing before you read anything else.

**Nearly every recent retargeting paper now ships a plausibility check of some kind.** Since about
mid-2026, papers routinely include a stage that verifies whether the motion is executable before
handing it downstream. The idea this project is working on is no longer unusual — but those checks
are all learned or optimisation-based and sit *inside* the pipeline. A cheap check that runs
*before* you spend an hour of compute is still a different thing.

**Almost nothing published uses the Unitree R1.** As of 2026-09-02, an arXiv full-text search for
"Unitree R1" returns **zero results**. Every paper above targets the G1, H1 or H1-2. The R1 is a
cheaper, smaller machine that a teaching lab can actually afford, and nobody has published on it.
That is the gap this project sits in.

---

# PART TWO — HOW THIS PROJECT STARTED

## The lab, and the machine

A research group of four students at The City College of New York, under Prof. Yi Sun, with one
physical humanoid: a **Unitree R1 EDU** — 1.2 m tall, about 25 kg, 24 driven joints, and roughly a
fifth of the price of the research humanoids the papers above use. The group's goals are to publish
with it, and to build teaching material that shows students how robots are trained.

The development machine is an ordinary laptop: Windows with Ubuntu inside it, four CPU cores, and a
GPU with 4 GB of memory. **That constraint shapes everything that follows.**

## What came before: a policy that worked, then didn't

The first line of work was walking. A controller was trained in a simulator, learned to walk, and
worked — in that simulator. Moved into the manufacturer's own simulator, the same controller fell
over in about two and a half seconds.

Months went into finding out why. The eventual answer was **not** what everyone expects. It was not
that the physics differed too much: the same controller walks for twenty seconds on the exact
deployment model when it is run cleanly. The gap was in the *plumbing* — the real-time message-passing
harness between the controller and the simulator, which delivers decisions late. A walking robot
tolerates far less lateness than a standing one.

Along the way, five controlled training experiments were run to try to fix the transfer by training
harder. All five improved the numbers you measure during training. **None of them transferred.** The
conclusion was uncomfortable but clear: on four CPU cores and 4 GB of GPU memory, more training was
not the lever. The laptop supports *diagnosis* very well and *training* poorly.

## The decision that led here

If training is the bottleneck, then the sensible move is to find work where **training is not the
bottleneck**. Two directions were chosen:

1. **Human video → robot motion** — retargeting. It is geometry and inverse kinematics. It runs on a
   CPU, in real time, with no training run at all.
2. **Writing up the sim2real methodology itself** — the diagnosis work, which the laptop is good at.

Retargeting had a second advantage: a labmate was already doing imitation learning and reinforcement
learning, and another was doing manipulation. Retargeting sits *upstream* of both — it produces the
reference motion that an imitation policy would later learn to track — so it complements their work
instead of competing with it.

## Why GMR, and what was missing

GMR was evaluated in March 2026 and was the obvious tool: open source, permissively licensed, runs on
a CPU, fast enough to be real-time, and designed so that adding a robot means writing a
correspondence file rather than modifying the code.

**It did not support the R1.** It supported eleven robots — several Unitree models among them — and
the R1 was not one. A clone straight from GitHub answers "no" to every question about this robot.

So the first contribution was simply to make it work: a robot model in the format GMR expects, two
correspondence files (one for each motion-capture format the project can use), the parameter entries
that register the robot, and the mesh files that give it a shape. That was done at the end of July
2026, along with a control run on the G1 so that any R1 oddity could be compared against a robot the
tool already supported.

**The first result** is a video of the R1 throwing punches, retargeted from a 35-second boxing
capture, next to the same motion on the G1.

## The question that turned a demo into research

The video is convincing. The guard comes up, the weight shifts, the punches extend and retract, the
feet stay under the body. Nothing looks wrong.

So the next question was the one KDMR asks: **could the robot actually have done that?**

Answering it needed a tool that nobody had written for this setup — a small program that takes a
retargeted motion and checks four things a real machine would have to satisfy: do the feet ever
touch the ground, does the body rise and fall like a real one does, is any joint driven past its
mechanical stop, and could a motor move that fast.

Run it on the boxing motion and the answer is **no**: the foot goes through the floor, and on a
handful of frames a joint is asked to move faster than the servo could deliver.

That is where the project is now, and it is a better position than the demo was. The video is a
demo. The measurement is a finding — and the tool that produced it is cheap enough to run *before*
you spend an hour of compute training a policy to imitate a motion no robot could perform.

Part Three shows you how to reproduce all of it.

---

# PART THREE — THE TUTORIAL

Everything below was run on Ubuntu 24.04 (inside WSL2 on Windows 11), with no GPU used at any point.
**Every `Expected` block is output that was actually produced by the command above it.** Where a
command's output is not shown, it is because it was not captured — not because it was tidied.

You need the package this document came in. Its files are referred to as being in
`~/r1_gmr` below; put it wherever you like and adjust the paths.

| Step | | Time |
|---|---|---|
| 0 | Make an environment | 2 min |
| 1 | Clone GMR | ~20 min, mostly download |
| 2 | Prove the R1 is missing | 1 min |
| 3 | Install the R1 | 2 min |
| 4 | Check the install | 1 min |
| 5 | Retarget a boxing capture | 1 min |
| 6 | Watch it | 5 min |
| 7 | Open the motion file | 2 min |
| 8 | Measure whether it is possible | 2 min |
| 9 | Get the wrong answer on purpose | 5 min |
| 10 | The window experiment | 10 min |
| 11 | See what is actually R1-specific | 5 min |

---

## Step 0 — An environment of its own

GMR needs MuJoCo 3.6 and an IK library called `mink`. If you have other robotics work on this
machine, it will want different versions. Keep them apart:

```
conda create -n gmr python=3.10 -y
conda activate gmr
```

**Your prompt now starts with `(gmr)`.** Every command from here on assumes it. This is the single
most common reason a command that worked yesterday fails today.

## Step 1 — Clone GMR

```
cd ~
git clone --recurse-submodules https://github.com/YanjieZe/GMR.git
cd GMR
pip install -e .
```

On Linux, one extra library:

```
conda install -c conda-forge libstdcxx-ng -y
```

About twenty minutes, nearly all of it download — the repository with its robot assets is roughly
3.6 GB. **`--recurse-submodules` is not optional**: GMR pulls a second repository of robot models as
a submodule, and without it you get a clone that imports fine and then fails on a missing file. If
you forget:

```
git submodule update --init --recursive
```

> **One warning prints on every single GMR command, forever:**
> `xrobotoolkit_sdk not found, skip for now.`
> It is a VR streaming dependency that nothing here uses. **Ignore it. It is not an error.**

## Step 2 — Prove the R1 is missing

Before adding anything, see the problem for yourself:

```
python -c "from general_motion_retargeting import IK_CONFIG_DICT; \
    print({k: 'unitree_r1' in v for k, v in IK_CONFIG_DICT.items()})"
```

**Expected — every single one `False`:**
```
{'smplx': False, 'bvh_lafan1': False, 'bvh_nokov': False, 'bvh_xsens': False,
 'fbx': False, 'fbx_offline': False, 'xrobot': False, 'xsens_mvn': False}
```

That is a correct, fully working GMR install that cannot retarget onto a Unitree R1 at all. Each key
is a motion-capture format; each `False` says "no correspondence file exists for this robot in this
format".

✅ **Checkpoint: all `False`. If you see any `True`, the R1 is already installed here.**

## Step 3 — Install the R1

Four things are needed: the robot model, its 43 mesh files, two correspondence files, and five lines
registering it in GMR's parameter table. The package's installer does all four and prints each one.

```
cd ~/r1_gmr
python install_r1_into_gmr.py --meshes support/assets
```

**Expected:**
```
GMR clone: /home/YOURNAME/GMR
  r1_mocap.xml             -> assets/unitree_r1/r1_mocap.xml
  r1_scene.xml             -> assets/unitree_r1/r1_scene.xml
  bvh_xsens_to_r1.json     -> general_motion_retargeting/ik_configs/bvh_xsens_to_r1.json
  smplx_to_r1.json         -> general_motion_retargeting/ik_configs/smplx_to_r1.json
  43 meshes copied from support/assets
  params.py: 5 entries added (backup: params.py.before_r1)

  R1 accepted from: smplx, bvh_xsens
  OK -- the R1 is installed in GMR
```

Run it a second time and the four file lines gain `(already there)` and the parameter line reads
`0 entries added -- already patched`. It is safe to repeat.

If your clone is not at `~/GMR`, add `--gmr /path/to/GMR`.

✅ **Checkpoint: `OK -- the R1 is installed in GMR`.**

Now ask the same question as step 2, in a friendlier form:

```
python - <<'EOF'
from general_motion_retargeting import IK_CONFIG_DICT
for fmt, robots in IK_CONFIG_DICT.items():
    print("  %-12s %s" % (fmt, "R1 OK" if "unitree_r1" in robots else "-"))
EOF
```

**Expected:**
```
  smplx        R1 OK
  bvh_lafan1   -
  bvh_nokov    -
  bvh_xsens    R1 OK
  fbx          -
  fbx_offline  -
  xrobot       -
  xsens_mvn    -
```

**Two formats work: SMPL-X and Xsens BVH.** `bvh_lafan1` is still `-`, and that matters: LAFAN1 is
the dataset most GMR tutorials use, and it is the dataset GMR's own paper benchmarks on. Nobody has
written that correspondence file for the R1. If you follow an online tutorial, this is where it will
refuse you, and the refusal is correct.

## Step 4 — Check the install

```
cd ~/GMR
python ~/r1_gmr/check_setup.py
```

**Expected:**
```
==================================================================
  Lab 4 setup check
==================================================================
  [ok]  python 3.10+                           3.10
  [ok]  mujoco                                 3.6.0
  [ok]  mink (the IK solver)
  [ok]  general_motion_retargeting
  [ok]  R1 robot model                         r1_scene.xml
  [ok]  R1 motion mappings                     smplx, bvh_xsens
  [ok]  the boxing capture                     2949 KB
  [ok]  mujoco_menagerie submodule
  [ok]  video writer (imageio)
==================================================================

  READY.  Nothing else to install.
```

Anything that fails prints the exact command that fixes it.

> The header says *Lab 4* because this checker is shared with the teaching material built from this
> same work — the workshop lab that walks students through steps 5 to 9. It is the same check either
> way.

## Step 5 — Retarget a boxing capture

GMR ships exactly one motion capture, and it is the one used throughout this project: **4,249 frames
at 120 Hz — 35 seconds of boxing**, recorded with an Xsens suit.

Run it on a slice in the middle:

```
cd ~/GMR
python scripts/xsens_bvh_to_robot_headless.py \
    --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
    --robot unitree_r1 --start 600 --end 1800 --save_path out/mine.pkl
```

**Expected — the last two lines:**
```
[headless] IK solve rate: 197.2 FPS (6.1s for 1200 frames), motion is 120 fps -> real-time capable
[headless] motion saved to out/mine.pkl
```

**Read that rate.** The solver is producing poses faster than the motion plays. Retargeting is not
an offline batch process — it could run live, off a streaming motion-capture suit, with margin.

Your number will differ, and it is sensitive to what else the machine is doing: the same command run
while a video was rendering in another terminal measured **35.5 FPS**. Anything above 120 makes the
same point.

**Why `--start 600 --end 1800`?** Because frames 0–360 are not the capture. You will find out what
they are in step 9, the hard way, on purpose.

✅ **Checkpoint: `out/mine.pkl` exists and your rate is above 100 FPS on an idle machine.**

## Step 6 — Watch it

There are three ways to see the motion, and on a Windows/WSL2 machine only two of them work.

**6a — the videos in the package (always works).** `media/r1_gmr_boxing.mp4` is the R1 throwing the
punches; `media/r1_vs_g1_boxing.mp4` is the same capture retargeted onto both the R1 and the Unitree
G1, side by side. The G1 is the robot every paper in Part One uses, so it is the control: if
something looks odd on the R1, the question is always "does the G1 do it too?"

**Watch one of them two or three times and commit to a judgement before step 8:** does that look
like a real robot boxing?

It is genuinely convincing. The guard comes up, the weight shifts, the punches extend and retract,
the feet stay planted under the body. Nothing about it looks wrong.

**6b — render your own (works, but slowly).** The same script that retargets can write a video:

```
python scripts/xsens_bvh_to_robot_headless.py \
    --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
    --robot unitree_r1 --start 600 --end 900 \
    --save_path out/clip.pkl --video_path out/clip.mp4
```

**Expected:**
```
[headless] IK solve rate: 27.2 FPS (11.0s for 300 frames), motion is 120 fps -> slower than real time
[headless] video saved to out/clip.mp4
[headless] motion saved to out/clip.pkl
```

**Look at the wall clock, not just that line.** The solver reports 11 seconds of work, but this
command took about **seven minutes** to finish on the machine used here — 300 frames of software
rendering under WSL2, at well under one frame per second. Almost all of the time went into drawing
pixels that the solver had already computed.

That contrast is the useful part: **the bottleneck is not the mathematics, it is the rendering.**
Worth remembering the next time something feels slow — measure before optimising the part you
assume is expensive. It is also why every script in this project has a headless path.

**6c — the interactive viewer (may not work at all).**

```
python scripts/vis_robot_motion.py --robot unitree_r1 --motion_file out/mine.pkl
```

> ⚠️ **Under WSL2 this fails with `ERROR: gladLoadGL error` on many machines, and that is not your
> mistake.** It was reproduced here in every configuration tried — forced X11, software rendering,
> and with the GPU genuinely engaged. Plain graphics windows open fine; it is MuJoCo's interactive
> viewer specifically. Native Linux and macOS are unaffected. **If you hit it, use 6a and 6b** —
> this is also the reason the headless script in this package exists at all.

## Step 7 — Open the motion file

Before measuring the motion, look at what retargeting actually produced:

```
python - <<'EOF'
import pickle, numpy as np
d = pickle.load(open("out/mine.pkl", "rb"))
for k, v in d.items():
    print("%-12s %-14s %s" % (k, str(np.asarray(v).shape), np.asarray(v).dtype))
EOF
```

**Expected:**
```
root_pos     (1200, 3)      float64
root_rot     (1200, 4)      float64
dof_pos      (1200, 24)     float64
fps          ()             int64
robot        ()             <U10
```

That is the whole thing. For each of 1,200 frames:

| | |
|---|---|
| `root_pos` | where the pelvis is — three numbers, metres |
| `root_rot` | which way the pelvis faces — four numbers |
| `dof_pos` | the angle of each of the 24 joints |

**Notice what is not there.** No velocities. No forces. No torques. No contact. No mass. **Nothing
about whether any of this is physically possible** — it is a list of poses and the times to hold
them.

That absence is the finding you are about to measure. It is also exactly what KDMR is about.

## Step 8 — Measure whether it is possible

The checker asks four questions a real machine would have to satisfy.

```
python ~/r1_gmr/exp4_check_motion.py out/mine.pkl
```

**Expected:**
```
  1. FEET -- does the robot ever touch the ground?
       lowest contact point:  min  -1.15 cm   mean  +0.50 cm   max  +1.80 cm
       -> touches down on 762/1200 frames (64%)

  2. PELVIS -- does the body rise and fall like a real one?
       height 0.648 .. 0.733 m   (range 8.49 cm)
       -> moves, as a real body would

  3. LIMITS -- is any joint pushed past its mechanical stop?
       -> none. every joint stays inside its range.

  4. SPEED -- could a motor actually move this fast?
       peak joint speed 27.3 rad/s   (limit set to 20)
       -> exceeded on 5 frames

  VERDICT: this motion is NOT physically realisable.
     - feet penetrate the floor by 1.15 cm
     - joint speed implausible on 5 frames
```

**Two of the four checks pass.** The body rises and falls 8.5 cm, like a real one. No joint is driven
past its mechanical stop. Whoever produced this did not do it carelessly.

But **the foot goes 1.15 cm below the floor**, and on five frames a joint is asked to move at
27.3 rad/s — past what the servo could deliver.

> ### Why this happens, and why it is not a bug
> Retargeting was given exactly one problem: **put the robot's limbs where the human's were.** It
> solves that, and solves it well, which is precisely why the video is convincing.
>
> Nothing in the pipeline was ever asked about gravity, contact, torque, or whether the robot could
> stay upright. **So nothing in the output respects them.** The floor is not part of the problem it
> was given, so the foot goes through it.
>
> A tool that solves the problem you gave it is not broken when it fails a problem you never
> mentioned. That is the whole argument of Part One's second paper.

## Step 9 — Get the wrong answer on purpose

This is the most useful five minutes here. Retarget the *beginning* of the capture — the part step 5
told you to skip:

```
python scripts/xsens_bvh_to_robot_headless.py \
    --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
    --robot unitree_r1 --start 0 --end 360 --save_path out/tpose.pkl
```
```
python ~/r1_gmr/exp4_check_motion.py out/tpose.pkl
```

**Expected:**
```
  1. FEET -- does the robot ever touch the ground?
       lowest contact point:  min  +4.53 cm   mean  +4.58 cm   max  +4.63 cm
       -> FLOATING the whole time.  0 of 360 frames make contact.

  2. PELVIS -- does the body rise and fall like a real one?
       height 0.787 .. 0.787 m   (range 0.06 cm)
       -> FROZEN. A moving human does not hold this constant.

  4. SPEED -- could a motor actually move this fast?
       peak joint speed 0.6 rad/s   (limit set to 20)
       -> within a plausible servo budget

  !! WARNING -- THIS LOOKS LIKE A CALIBRATION POSE, NOT A MOTION.
```

**That is a far more dramatic result than the real one.** *The robot never touches the ground and its
pelvis never moves* — a much better story than "the foot sinks a centimetre."

It is also completely wrong. Every Xsens recording opens with the performer standing still in a
calibration T-pose. Every number above is true of that pose and false of the boxing.

**The giveaway is check 4** — the one nobody was looking at. 0.6 rad/s is impossibly slow for
someone throwing punches. An implausible number in a check you were not focused on is what exposes a
mistake in the one you were.

> **This exact error was made while building this project, and it was written up as a headline
> finding before anyone caught it.** The tool now refuses to give a verdict when the input looks like
> a calibration pose — which is why the warning exists at all.
>
> **A measurement tool that cannot say "I do not trust this input" will eventually hand you a
> confident wrong answer.**

## Step 10 — The window experiment

Step 9's lesson was that measuring the wrong frames gives the wrong answer. This step asks a harder
question: **does the choice of frames change the motion itself, or only what you measured?**

Retarget the whole capture in one run, with no start or end:

```
python scripts/xsens_bvh_to_robot_headless.py \
    --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
    --robot unitree_r1 --save_path out/full_capture.pkl
```
```
python ~/r1_gmr/exp4_check_motion.py out/full_capture.pkl
```

**Expected:**
```
  1. FEET -- does the robot ever touch the ground?
       lowest contact point:  min  +3.22 cm   mean  +4.03 cm   max  +5.57 cm
       -> FLOATING the whole time.  0 of 4249 frames make contact.

  2. PELVIS -- does the body rise and fall like a real one?
       height 0.600 .. 0.786 m   (range 18.60 cm)
       -> moves, as a real body would

  4. SPEED -- could a motor actually move this fast?
       peak joint speed 14.4 rad/s   (limit set to 20)
       -> within a plausible servo budget
```

Compare that with step 8. **The same source file, the same tool, the same robot** — and one run says
the foot penetrates the floor by 1.15 cm on 64% of frames, while the other says the robot floats
4 cm and never touches the ground at all.

The boxing frames are inside both runs. So which is true?

```
python ~/r1_gmr/measurements/compare_windows.py \
    --full out/full_capture.pkl --window out/mine.pkl --start 600
```

**Expected:**
```
full capture   :  4249 frames, root z 0.5998 .. 0.7858
windowed run   :  1200 frames, root z 0.6481 .. 0.7330
same frames in the full run  : root z 0.7110 .. 0.7854

root height difference (full - windowed):
   mean +0.0590 m   sd 0.004326 m   min +0.0512   max +0.0642

joint angles, same frames: max |difference| = 5.053e-01 rad  (mean 6.038e-02)
```

Read those two numbers carefully.

**The robot sits 5.9 cm higher** in one run than the other, for the very same frames — and the
difference barely varies (a few millimetres of spread), which is the signature of a constant
vertical offset rather than a different motion.

**But the joint angles are not the same either.** They differ by up to half a radian — about 29
degrees at some joint, on some frame. That is not an offset. That is a genuinely different solution
to the same problem.

> ### What this means
> **Both earlier numbers were correct.** They are honest measurements of two different outputs, and
> the pipeline produced two different outputs from the same input file.
>
> Two things are probably responsible, and this project has not yet separated them:
>
> 1. **The floor is inferred from the clip.** GMR works out where the ground is from the motion it
>    is given. Hand it a clip that opens with someone standing still, and it makes a different
>    decision than if you hand it the middle of a fight.
> 2. **The solver starts from wherever it last was.** Each frame is solved starting from the
>    previous frame's answer. Arriving at frame 600 having worked through 599 earlier frames leaves
>    the solver somewhere different than starting at frame 600 cold — and, as Part One's fourth
>    paper points out, this kind of problem has more than one valid answer.
>
> **Separating those two causes is the most valuable open experiment in this project**, and it needs
> no new hardware: disable the floor inference, re-run, and see how much of the gap survives.

✅ **Checkpoint: you can state why two correct measurements disagreed.**

## Step 11 — What is actually R1-specific

The last step is the honest accounting of what was added to make any of this work.

```
python ~/r1_gmr/measurements/measure_config_and_links.py
```

This compares the R1 correspondence file against the one GMR ships for the Unitree G1 — the file it
was adapted from — and then measures the robot.

**Expected — the first part:**
```
  ik_match_table1 -- 14 bodies each
    bodies renamed for the R1 : ['left_wrist_roll_link', 'right_wrist_roll_link']
    bodies they replace       : ['left_wrist_yaw_link', 'right_wrist_yaw_link']
    shared bodies retuned     : 0

  human_scale_table:
    Hips             G1 0.9    -> R1 0.84      changed
    LeftKnee         G1 0.9    -> R1 0.818     changed
    LeftShoulder     G1 0.75   -> R1 0.578     changed
    ...
```

**That is the whole R1-specific contribution to the mapping**, and it is smaller than it sounds:

1. **One joint substitution.** The G1 has a wrist that yaws; the R1 has one that rolls. Two lines.
2. **Three limb-group size factors** — torso, legs, arms — changed because the R1 is a smaller
   machine with different proportions.

**Everything else is GMR's.** All 14 body correspondences, every weight that decides which body part
wins when the geometry cannot satisfy them all, the two-pass structure, the assumed human height.
The honest description is *"I adapted and retuned GMR's G1 configuration for the R1"*, never *"I
derived a new mapping"*.

**And the size factors were tuned by hand, not calculated.** The second half of the output measures
the robot's actual bones against the recorded performer's:

```
  segment          R1  performer    ratio     config
  --------------------------------------------------
  thigh        0.291m     0.433m    0.673      0.818
  shank        0.310m     0.422m    0.735      0.818
  upper arm    0.221m     0.313m    0.704      0.578
  forearm      0.117m     0.256m    0.456      0.578
```

The `ratio` column is what the geometry implies. The `config` column is what is actually used. They
do not agree — so those numbers were arrived at by trying values until the motion looked right.

That is normal practice and not a criticism. But it is a **second open experiment sitting in plain
sight**: set the factors from the measured ratios, re-run step 8, and see whether the floor
penetration changes. It is ten minutes of work and it is a result either way.

---

# WHERE THIS LEAVES YOU

You have installed a research retargeting tool, added a robot it did not support, turned 35 seconds
of human boxing into robot joint angles, watched it, and then established — with a measurement, not
an opinion — that the motion could not physically have happened.

You have also met the two failure modes that matter more than the result:

- **Measuring the wrong thing confidently** (step 9): four correct numbers, all about the wrong three
  seconds.
- **Two correct measurements that disagree** (step 10): because the tool's output depends on an input
  choice nobody thought was a parameter.

## What is open, if you want to work on this

1. **Separate the two causes in step 10.** Half a day, no new hardware, and it is a genuine finding.
2. **Derive the size factors from the measured ratios** (step 11) and see what changes.
3. **Write the missing LAFAN1 correspondence file.** It would open a large public dataset to this
   robot and make GMR's own published benchmark reproducible on it — which currently it is not.
4. **Add a physics stage**, as KDMR does, but using only what is in the motion file. KDMR needs force
   plates; the authors name inference-from-kinematics as future work.
5. **Publish anything at all about this robot.** As of September 2026 there is nothing.

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `ModuleNotFoundError: general_motion_retargeting` | you forgot `conda activate gmr` — by far the most common |
| `ModuleNotFoundError: mink` | `pip install -e .` was not run, or ran in the wrong environment |
| every format prints `-` in step 3 | the installer did not run, or ran against a different clone — pass `--gmr` |
| `unitree_r1` rejected by `--robot` | you are using a LAFAN1 script; use `xsens_bvh_to_robot_headless.py` |
| missing `.xml` under `mujoco_menagerie/` | cloned without `--recurse-submodules` |
| `xrobotoolkit_sdk not found` | not an error. It prints on every command, forever |
| `ERROR: gladLoadGL error` | the interactive viewer cannot open under WSL2 — use the headless script |
| rendering is a hundred times slower than solving | expected. It is the pixels, not the mathematics |
| `VERDICT: withheld` | you measured the calibration T-pose — see step 9 |
| your numbers differ from this document | say so. On the same versions they should not, and a difference is a finding |
