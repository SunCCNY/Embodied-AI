# Part 2 — MuJoCo lab experiments

Four lab manuals built around a Unitree R1 humanoid in MuJoCo. Each one is self-contained: the
student starts from an empty folder, installs what is needed, and types every file out of the
manual. Work through them in order — each lab uses what the previous one built.

| Manual | What it covers |
|---|---|
| Lab 1 - Make a Humanoid Stand | standing with no learning at all, built from an empty folder |
| Lab 2 - Teach It To Stand | training a policy, and why a rising training curve does not mean it works |
| Lab 3 - Where It Breaks | control delay, sensor loss, and swapping the simulator underneath a policy |
| Lab 4 - Make the Robot Copy a Human | motion capture retargeted onto the R1, and measuring whether the result is physically possible |

`Setup - Installing GMR (do this at home).pdf` is a separate install guide for Lab 4. GMR takes a
while to install and needs its own Python environment, so it is meant to be done before the session
rather than during it.

## Requirements

Ubuntu shell — WSL2 on Windows, or native on macOS and Linux. Labs 1 to 3 need only MuJoCo and
NumPy to run, plus PyTorch and stable-baselines3 to train in Lab 2. Lab 4 uses its own environment.
Every command is given in the manuals.

## About the numbers

Every block labelled `Expected` in these manuals is real output, produced by running the command
shown on a machine built by following the same steps. Verified on Ubuntu 24.04 under WSL2 with
MuJoCo 3.12.0, NumPy 2.4.6, PyTorch 2.14.0 CPU and stable-baselines3 2.9.0; Lab 4 on Python 3.10
with MuJoCo 3.6.0. Exact numbers can shift slightly between MuJoCo versions — where that matters,
the manual says so.

## The Python files here

`r1_walk_env.py`, `r1_walk_train.py`, `exp2_eval.py`, `export_policy.py` and `watch_policy.py` are
the reference versions of scripts the labs build up. They are here for convenience; the manuals do
not ask students to download them.

The slides this work was presented from are in `Presentations/Shaoqin-Li/`.
