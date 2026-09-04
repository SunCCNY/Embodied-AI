# Experiment 2 — Teach It To Stand

**Replace the two hand-tuned numbers with a neural network, and train it yourself.**

## What you do

In Experiment 1 you told the robot exactly what pose to hold. Here you tell it only what *good*
means, and let it work out the pose — fifty times a second, from scratch, by trial and error.

You build the training environment, read the reward that defines the task, launch a real training
run, and while it runs in the background you measure a finished policy that was trained the same
way. At the end you come back to your own run and see how far it got.

**Nothing you need depends on your training finishing.**

## Before you start

**Finish Experiment 1.** This one reuses that folder and that software environment, and it only
means something once you have personally failed to survive a 0.20 m/s push.

You will install two more packages — PyTorch and stable-baselines3 — and the manual gives the exact
command. On Linux you must ask for the CPU build, or you will download 4.8 GB of GPU code you cannot
use.

## What is in this folder

| File | |
|---|---|
| `Lab 2 - Teach It To Stand.pdf` | the manual, 44 pages |
| `Lab 2 - Teach It To Stand.md` | the same as text — **copy the code from here** |
| `lab2_policy.tar.gz` | 2 MB — a finished, trained policy |

**Why there is a download.** A trained network is a few hundred thousand numbers produced by an hour
of computation. It cannot be typed. This one lets you measure a working policy immediately instead
of waiting for your own to finish.

## What you will find

The trained robot survives a shove **twice as hard** as the hand-tuned one from Experiment 1 — and
for a structural reason, not because it is better tuned.

Then the real lesson. Somewhere in the middle of training there is a stretch where the policy
**cannot stand for two seconds**, while its training score is hitting an all-time best. The curve
you would normally watch says everything is fine. It is not.

> **A rising training curve does not mean your policy works.** The only way to know is to evaluate
> it.

## Start here

Open the PDF at Part 0. Launch the training early — Part 4 — so it accumulates while you read.
