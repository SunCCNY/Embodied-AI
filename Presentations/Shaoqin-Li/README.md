# Shaoqin Li — group meeting presentations

Slides from the weekly Applied and Embodied AI Laboratory group meetings, in the order they
were given, one folder per week. The through-line is a Unitree R1 humanoid in MuJoCo: getting it to stand, training
it to walk with reinforcement learning, chasing down why a policy that works in training fails
on deployment, and most recently retargeting human motion capture onto the robot.

| Week | Presented |
|---|---|
| [Week 05](Week%2005/) | 2026-02-27 |
| [Week 07](Week%2007/) | 2026-03-13 |
| [Week 09](Week%2009/) | 2026-03-27 |
| [Week 10](Week%2010/) | 2026-04-10 |
| [Week 14](Week%2014/) | 2026-05-08 |
| [Week 17](Week%2017/) | 2026-06-19 |
| [Week 18](Week%2018/) | 2026-06-26 |
| [Week 19](Week%2019/) | 2026-07-03 |
| [Week 20](Week%2020/) | 2026-07-15 |
| [Week 21](Week%2021/) | 2026-07-21 |
| [Week 22](Week%2022/) | 2026-08-22 |
| [Week 23](Week%2023/) | 2026-08-22 |
| [Week 24](Week%2024/) | 2026-08-28 |
| [Week 25](Week%2025/) | 2026-09-03 |
| [Week 26](Week%2026/) | 2026-09-11 |
| [Week 27](Week%2027/) | — |
| [Week 28](Week%2028/) | — |

Week numbers follow the lab meeting schedule, so gaps are weeks I did not present.
Week 25 is the GMR motion retargeting deck; Week 26 extends it to all eight GMR input formats
and compares the R1 against the G1 baseline. Week 27 closes the video pipeline end to end
(phone video → GVHMR → SMPL-X → GMR → R1 motion). Week 28 trains imitation-learning policies on
those motions (BeyondMimic via mjlab) and checks them sim2sim; its folder is a full package —
deck, speaker companion, the code added to mjlab, and the trained policies — see its README.

The lab manuals that came out of this work are in `CUNY-AI/Part-2/`.
