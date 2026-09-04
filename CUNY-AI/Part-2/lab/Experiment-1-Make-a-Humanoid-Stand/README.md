# Experiment 1 — Make a Humanoid Stand

**No learning at all. Two numbers, tuned by hand, and a robot that stays upright.**

## What you do

Start from an empty folder and end with a simulated Unitree R1 standing on its own. You install the
tools, download the robot's shape from Unitree's public repository, write the file that describes it
to the simulator, and then find the two numbers — a spring strength and a damping — that hold it up.

Then you break it on purpose, twice, in two completely different ways, and learn to tell those two
failures apart. That distinction is the point of the experiment: **one of them is a robot falling
over, and the other is the simulator failing.** Confusing the two costs people weeks.

## Before you start

Nothing. This is the first experiment and it assumes no software, no account and no prior setup.
Part 0 of the manual installs everything, including Ubuntu if you are on Windows.

Set aside an evening. It is the longest of the four because it builds the foundation the other three
reuse.

## What is in this folder

| File | |
|---|---|
| `Lab 1 - Make a Humanoid Stand.pdf` | the manual, 59 pages — read from this |
| `Lab 1 - Make a Humanoid Stand.md` | the same document as text — **copy the code from here** |

**No download.** Everything in this experiment you build or fetch yourself, which is deliberate: at
the end you can rebuild the whole thing from an empty folder, and the last part of the manual asks
you to do exactly that.

## What you will find

A map of all 49 combinations of the two numbers, showing that the region where the robot stands has
both a floor and a ceiling — for two unrelated reasons. And a limit: a push of 0.20 m/s knocks the
robot down, and **no combination of the two numbers fixes it.** That wall is what Experiment 2 is
for.

## Start here

Open the PDF at Part 0 and work forward. Do not skip steps — a step you skip is a file you do not
have.
