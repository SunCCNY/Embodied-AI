# Part 2 — Lab: Training a Humanoid Robot in Simulation

**Four experiments on a Unitree R1 humanoid, in MuJoCo. Any laptop, no GPU required.**

Shaoqin Li · Applied and Embodied AI Lab · Electrical Engineering Department · The City College of
New York

---

## What this is

Four self-contained experiments that take you from *no software at all* to a humanoid robot that
stands, learns to stand by itself, gets attacked until it breaks, and finally copies a human boxer
from motion capture.

You build everything yourself. No prepared folder of code is handed to you: you install the tools,
download the robot from the manufacturer's public repository, and create every file by pasting it
out of the manual. There are exactly four small downloads in the whole sequence, and each one is a
thing that **cannot** be typed — a trained neural network, or a file that belongs to another
project. Every one of them is explained where it appears.

Everything runs on an ordinary laptop with **no GPU**.

## The four experiments

| | Folder | What you do | Roughly |
|---|---|---|---|
| 1 | [`Experiment-1-Make-a-Humanoid-Stand`](Experiment-1-Make-a-Humanoid-Stand) | Make the robot stand with no learning at all — two hand-tuned numbers | 2–3 h |
| 2 | [`Experiment-2-Teach-It-To-Stand`](Experiment-2-Teach-It-To-Stand) | Replace those two numbers with a neural network you train yourself | 1.5 h + training |
| 3 | [`Experiment-3-Where-It-Breaks`](Experiment-3-Where-It-Breaks) | Attack the trained policy three ways and find what actually breaks it | 1 h |
| 4 | [`Experiment-4-Make-the-Robot-Copy-a-Human`](Experiment-4-Make-the-Robot-Copy-a-Human) | Turn a human motion capture into robot motion — then measure whether a robot could do it | 1 h |

**Do them in order.** Each one reuses the folder and the software environment built by the one
before it. Experiment 4 is the exception: it uses a second, separate tool, and its setup must be
done at home before the session (its folder explains why).

## Before you start

- **A laptop with about 5 GB free.** Windows, macOS or Linux.
- **On Windows you need Ubuntu through WSL2.** Experiment 1 installs it for you, step by step. After
  that, everybody types the same commands regardless of operating system.
- **No GPU, no admin rights beyond installing your own software, no cloud account.**
- Set aside a full evening for Experiment 1. It is the longest, because it starts from nothing.

## How each folder works

Every experiment folder contains the same three kinds of thing:

| | |
|---|---|
| **`… .pdf`** | the manual — read from this one |
| **`… .md`** | the *same* document as plain text. **Copy code from here, not from the PDF** — copying Python out of a PDF loses the indentation, and in Python the indentation is part of the code |
| **`… .tar.gz`** | the small download for that experiment, when it has one |

To read a `.md` file comfortably, click it on GitHub — it renders as a formatted page. To copy code
out of it, use the **Raw** button, or open the downloaded file in a text editor.

## About the numbers in these manuals

Every block labelled **Expected** was produced by actually running the command shown, on a machine
built by following these exact steps. Nothing is illustrative and nothing was tidied up afterwards.

That matters, because it means **a difference is information**. If your output does not match, the
first assumption should not be that you did something wrong — tell your instructor, because it may
be a genuine difference between your machine and ours, which is itself worth knowing.

Verified on Ubuntu 24.04 under WSL2 — MuJoCo 3.12.0, NumPy 2.4.6, PyTorch 2.14.0 (CPU),
stable-baselines3 2.9.0; Experiment 4 additionally on Python 3.10 with MuJoCo 3.6.0.

## What you should be able to say at the end

1. What makes a robot stand up, without using a formula.
2. Why a learned controller survives a shove that a hand-tuned one cannot — and why that is a
   difference in kind, not in tuning.
3. **Why a rising training curve does not mean your policy works.**
4. Why a motion that looks perfect can be physically impossible, and how to catch that in ten
   seconds instead of after an hour of training.
5. How to tell a robot failing from a *simulator* failing.

## If you get stuck

Every manual ends with a troubleshooting table covering the errors we actually hit while writing it,
including the ones caused by the operating system rather than by you. Work through that first, then
ask.
