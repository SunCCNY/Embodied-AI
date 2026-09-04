# Experiment 3 — Where It Breaks

**Attack a working policy three ways, and find out which failures a real robot would actually have.**

## What you do

Experiments 1 and 2 both ended in success. This one asks the question that decides whether any of it
would survive contact with hardware: **what breaks it, and which of those breakages are real?**

Three attacks on two frozen policies — a standing one and a walking one you have not seen before:

1. **Delay.** Make the controller late, the way a real robot's network and computation make it late.
2. **Corrupt one sensor at a time.** Find the one the robot cannot do without.
3. **Swap the entire simulator.** Run the same policy on a second, independently-built model of the
   same machine.

Nothing trains. Nothing is random. You are measuring, and you write your predictions down first.

## Before you start

**Finish Experiments 1 and 2.** You reuse the robot file from Experiment 2 and the viewer from it.
**Nothing new to install** — if Experiment 1 ran, this runs.

You also build the second robot yourself, from the description Unitree publish, using a script that
prints every one of the six changes it makes. Nobody hands you a mystery model to compare against.

## What is in this folder

| File | |
|---|---|
| `Lab 3 - Where It Breaks.pdf` | the manual, 29 pages |
| `Lab 3 - Where It Breaks.md` | the same as text — **copy the code from here** |
| `lab3_policies.tar.gz` | 592 KB — two trained policies, standing and walking |

**Why there is a download.** Walking does not train from scratch in a lab session. The walking policy
took four warm-started runs and millions of steps; you are handed it so you can attack it.

## What you will find

Standing tolerates **six times** more control delay than walking. One sensor — the one that tells the
robot which way is down — is the one it cannot lose, and it is probably not the one you would guess.

And the third attack does not produce a fall at all. Two different simulators, same policy, and the
robots simply **drift apart** while both stay upright. Understanding why that is a drift and not a
fall is the whole experiment.

> A measurement without a control is an anecdote. Every part of this experiment has one, including a
> control that was itself broken the first time it was written.

## Start here

Open the PDF at Part 0. **Write your predictions down before running anything** — the tables land far
harder against a committed guess.
