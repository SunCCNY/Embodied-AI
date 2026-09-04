# Experiment 4 — Make the Robot Copy a Human

**Turn a real human motion capture into robot motion — then measure whether a robot could have done
it.**

## ⚠ Do the setup at home, days before the session

This is the only experiment with a large download (about 3.6 GB) and a second software environment.
**There is no time to install it during a session.**

Work through `Setup - Installing GMR (do this at home).pdf` first, run the checker at the end of it,
and bring the `READY` verdict with you.

## What you do

Take a 35-second motion capture of a person boxing and convert it onto the Unitree R1, using GMR, a
published research tool. GMR supports nineteen robots and **the R1 is not one of them** — the setup
guide has you prove that yourself, then install the four pieces that fix it, using the same robot
meshes you downloaded back in Experiment 1.

Then you watch the robot throw punches. It looks superb.

Then you measure it.

## Before you start

Experiments 1–3 are not strictly required to run the commands, but Part 6 only means something if
you have trained a policy yourself. **The setup must already be done.**

## What is in this folder

| File | |
|---|---|
| `Setup - Installing GMR (do this at home).pdf` | 11 pages — **do this first, at home** |
| `Setup - Installing GMR (do this at home).md` | the same as text |
| `Lab 4 - Make the Robot Copy a Human.pdf` | the manual, 15 pages |
| `Lab 4 - Make the Robot Copy a Human.md` | the same as text — **copy the code from here** |
| `r1_gmr_support.tar.gz` | 8 KB — the four files that add the R1 to GMR |
| `lab4_fallback.tar.gz` | 2.4 MB — **only if the install defeats you**: a finished video and motion file, so you can still do the measurement |

## What you will find

The motion is **not physically possible**. The foot passes through the floor, and on five frames a
joint is asked to move faster than its motor could. Two of the four checks pass, so it was not built
carelessly — it fails anyway.

That is not a bug in the tool. Retargeting is asked exactly one question — *where do the limbs go* —
and it answers it well. Nobody ever asked it about gravity, contact or torque, so nothing in its
output respects them.

Then you deliberately produce a **much more dramatic wrong answer**, and find the one number that
gives the mistake away. That mistake was made for real while building this experiment, and reported
as a finding before anyone caught it.

> A measurement tool that cannot say "I do not trust this input" will eventually hand you a confident
> wrong answer.

## Start here

Setup guide at home. Then the manual, Part 0.
