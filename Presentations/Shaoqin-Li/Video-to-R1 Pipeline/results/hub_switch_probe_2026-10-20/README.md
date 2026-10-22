# Hub-switch probe (2026-10-20, Week 31)

Question: can we chain separately trained tracking policies (a "skill bag") by switching from one to the next, or do
we need a transition/recovery policy? No new training. Policy A runs to a cut point, then the SAME physics state
continues into policy B's motion, re-anchored to where the robot stands. Policies: plain mjlab tennis and dance
(`policies/mjlab_tennis`, `policies/mjlab_dance`). Harness = `sim2sim_v5_dr.py` rules (training randomization,
training termination rules); the runner reproduces its numbers exactly (tennis 0.577 rad, dance 2.240 rad).
20 seeds per cell. Controls: tennis alone 20/20, dance alone 20/20.

## Result (switch + all of B survived, /20)
| A -> B | transition | at A's end | cut 30% | cut 50% | cut 70% |
|---|---|---|---|---|---|
| tennis -> dance | direct (B frame 0 at once) | 20 | 18 | 18 | **0** |
| tennis -> dance | 1 s ramp to B frame 0 | 19 | 20 | 20 | 19 |
| tennis -> dance | via default stand | 18 | 20 | 19 | 19 |
| dance -> tennis | direct | 20 | 20 | 20 | 20 |
| dance -> tennis | 1 s ramp | 20 | 20 | 11 | 18 |
| dance -> tennis | via default stand | **1** | 8 | 7 | 2 |

## What it says
1. **The motions do not start or end at the default stand.** Largest joint offset from default: tennis start 0.77 /
   end 0.61 rad, dance start 0.95 / end 1.93 rad. The "hub = default stand" idea needs clips trimmed or blended to it.
2. **A direct switch is surprisingly survivable** (jp gap 1.8-3 rad at the switch, first-second error 1.1-2.0 rad, then
   back to B-alone error) but not reliable: tennis cut at 70% (arm raised mid-serve, 4.5 rad gap) -> dance 0/20.
3. **Ramped references do not fix it.** Ramps and holds are out-of-distribution for a tracker; dance->tennis ramp falls
   to 11/20 at a 50% cut, and "via default stand" is the worst (1/20).
4. **The default stand is not a stable rest state.** PD-holding DEFAULT_POS with no policy tips forward within ~3 s
   (pelvis z 0.80 -> 0.09, one run). A tracking policy asked to hold it fails too. A hub needs a dedicated standing /
   balance policy -> the velocity-command policy (`unitree_rl_mjlab`, zero command = stand) is the natural candidate.

## Caveats
Two motions, plain (non-delay) policies, one pair of directions. "Survive" uses the training termination rules, which
tolerate large joint error (dance alone already averages 2.26 rad because its reference violates joint limits).
Nominal (no randomization) not run. Standing collapse is a single 3 s check.

## Reproduce
`./run_switch_matrix.sh train_DR <cut>` (cut 1.0 / 0.3 / 0.5 / 0.7) from the mjlab root;
`switch_video.py <A> <B> direct 1000 out.mp4` needs `MUJOCO_GL=osmesa` + `LD_LIBRARY_PATH` to `~/local_osmesa` (SETUP.md).
Videos: `videos/*_direct.mp4` (label shows which policy is driving).
