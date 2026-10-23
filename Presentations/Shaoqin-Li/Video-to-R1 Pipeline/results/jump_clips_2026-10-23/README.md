# Jump clips through the pipeline (2026-10-23)

The Week 31 skill bag needs a jump. Two stock jump clips went through the unchanged pipeline:
GVHMR → GMR (`--src_fps`) → mjlab tracking (3,000 plain + 3,000 with 0–50 ms delay) → plain-MuJoCo sim2sim.
The recipe and harness are the ones from [`../new_clips_2026-10-14/`](../new_clips_2026-10-14/README.md).

**Headline: jumping jacks work.** In plain MuJoCo the R1 does about 10 jumping jacks (20 jumps), frame-synchronized with the video. Both feet are off the ground
20 % of the time (measured from floor contacts in the nominal rollout)
(`videos/jumping_jacks_side_by_side.mp4`). The delay-trained policy survives 30/30 under training randomization
and 20/20 up to 50 ms of command delay. **Jump rope does not work yet.** The robot stays up but looks wrong. The reason is the
retargeted reference, not the policy (see below).

## The clips (Pexels license: free to use and modify, credit given; raw clips not re-hosted here)

| name | source | camera | fps | length used |
|---|---|---|---|---|
| jumping_jacks_fixed | ["Elderly Man Doing Jumping Jacks Outside", Kindel Media, Pexels 7299359](https://www.pexels.com/video/elderly-man-doing-jumping-jacks-outside-7299359/) | fixed | 30 | full 10.4 s (downscaled 2160×3840 → 1080×1920) |
| jump_rope_fixed | ["A Man Jumping Over The Skip Rope", Pexels 4812842](https://www.pexels.com/video/a-man-jumping-over-the-skip-rope-4812842/) | fixed | 25 | full 11.6 s (3840×2160 → 1920×1080) |

**How they were chosen.** 11 candidates were downloaded from Pexels searches for jumping, jumping jacks, jump rope, trampoline and parkour.
`scripts/triage_candidate_clips.py` was run on them, then each candidate was checked by eye. Rejected: the two "A Person Jumping" clips
(6726849/6726851; body cut in 100 % of frames, camera moving), two jump-rope clips framed from the waist up, a two-person gym clip,
a trampoline (not flat ground), a parkour leap (slow motion, so the timing is wrong) and a bed-jumping clip (several people).
The jumping-jacks clip was the best fit: one person in 100 % of frames, whole body in frame, static camera, real flight phases,
and it starts and ends in a normal standing pose. That last point matters for the hub, which switches skills from the stand.
No clean static clip of a single vertical or broad jump turned up in this search.

## Stage by stage

- **GVHMR** (`-s`, static camera): about 80 s per clip. The demo's final render step crashes because the SMPL (not SMPL-X) body
  model is missing. This happens after `hmr4d_results.pt` is saved, so it does no harm.
- **GMR**: `--src_fps 30` / `25`, default `--ground linear`. Pelvis height (`stills/reference_pelvis_height.png`): jumping jacks
  0.65–0.83 m at about 2 jumps/s, jump rope 0.71–0.85 m at about 2.2 hops/s. No joint passes its limit. Peak joint speed 10.3 / 14.1 rad/s.
- **Training**: 4,096 robots. Both clips trained in parallel with the walk/run delay fine-tune: stage A 2 h 12 min, stage B 2 h 45 min.

| clip | stage | mean reward | episode length (of 500) | training joint error (rad) |
|---|---|---|---|---|
| jumping jacks | A plain | 32.5 | 495 | 0.86 |
| jumping jacks | B delay | 32.2 | 491 | 0.71 |
| jump rope | A plain | 35.9 | 500 | 1.04 |
| jump rope | B delay | 36.5 | 492 | 0.80 |

## Sim2sim (plain CPU MuJoCo, `4_sim2sim/sim2sim_v5_dr.py`, `sweeps/`)

Survival, stage A (plain) → stage B (delay-trained). Mean joint error of the delay-trained policy in brackets.

| scenario | jumping jacks | jump rope |
|---|---|---|
| nominal | 1/1 → 1/1 (0.59 rad) | 1/1 → 1/1 (1.06 rad) |
| training randomization | 30/30 → 30/30 (0.64) | 27/30 → 30/30 (1.10) |
| + 20 ms delay | 19/20 → 20/20 | 12/20 → 20/20 |
| + 30 ms | 19/20 → 20/20 | 4/20 → 19/20 |
| + 40 ms | 2/20 → **20/20** | 0/20 → 16/20 |
| + 50 ms | 1/20 → **20/20** | 0/20 → 3/20 |
| + 60 ms | 0/20 → 19/20 | 0/20 → 1/20 |
| + 70 ms | 0/20 → 15/20 | 0/20 → 0/20 |
| + 80 ms | 0/20 → 4/20 | 0/20 → 0/20 |
| pushes ×3 | 8/20 → 11/20 | 4/20 → 10/20 |
| friction 0.15 | 11/20 → 14/20 | 8/20 → 12/20 |
| kp −20 % | 20/20 → 20/20 | 15/20 → 17/20 |
| payload 2 kg | 20/20 → 20/20 | 18/20 → 20/20 |
| no base linear velocity | 15/20 → 17/20 | 14/20 → 16/20 |

- Jumping jacks show the same delay pattern as the five paper motions: the plain policy loses at 40 ms, and the delay policy holds to 60 ms.
- Pushes are the weak point for both clips (about half survive), as for the boxing clip on 10-14. A jump has short flight
  phases, and a push during flight cannot be corrected until landing.
- Caveat (same as 10-14): stage B also adds 3,000 iterations. Only the tennis seed study and today's walk/run control
  ([`../walk_run_delay_2026-10-23/`](../walk_run_delay_2026-10-23/README.md)) isolate the delay itself. One seed per clip.

## Why jump rope looks wrong: the reference, not the policy

In `videos/jump_rope_side_by_side.mp4` the R1 barely leaves the ground (airborne 0.5 % of the time; the person hops about 26 times).
It holds its arms out to the sides and leans its torso. The person keeps their
upper arms down at the sides and turns the rope from the wrists. `scripts/render_reference_frames.py` draws the reference motion
itself, with no physics and no policy (`stills/jump_rope_reference_frames_120_300_480.png`). The reference already has the arms
raised sideways and the torso leaning. So the error enters before training: in GVHMR (hands holding handles, small fast wrist
motion), in GMR's IK, or in both. The policy tracks that wrong reference. That is also why this clip's joint error stays
around 1.1 rad and 50 ms breaks it. The SMPL-X output has not been looked at yet, so it is not known which stage is at fault. The
next step is to render the GVHMR body over the video. For the skill bag, use jumping jacks.

## Files

| path | what |
|---|---|
| `videos/<clip>_side_by_side.mp4` | original video next to the delay-trained R1 policy in plain MuJoCo (nominal), frame-synced, H.264, azimuth 270 |
| `stills/` | 3 frames per side-by-side, the jump-rope reference frames, and the reference pelvis-height plot |
| `sweeps/sweep_*.txt`, `sweeps/l4dc_sweep_*.txt` (+ json) | the 10-14 scenario set, and the fine delay grid 0–80 ms + stress tests |
| `policies/<clip>_{base,delay}.onnx` | stage A and stage B policies |
| `scripts/run_jump_clips.sh` | training + first sweep driver, exactly as run (from `~/mjlab`) |
| `scripts/triage_candidate_clips.py`, `triage_output.txt` | candidate triage |
| `scripts/measure_flight.py` | airborne fraction in the nominal rollout (`ONNX=... MOTION=... python measure_flight.py`, from `~/mjlab`) |
| `scripts/render_reference_frames.py` | draws the reference motion at chosen 50 Hz steps (no physics) |
| `../../data/jump_clips_2026-10-23/` | GVHMR outputs, GMR trajectories (`.pkl`, `.csv`), 50 Hz references (`.npz`) |

Commands (same as the main README, with the real frame rate):
```
python tools/demo/demo.py --video=jumping_jacks_fixed.mp4 -s --output_root <out>                    # GVHMR env
python 2_smplx_to_r1/gvhmr_to_robot_headless.py --gvhmr_pred_file <out>/jumping_jacks_fixed/hmr4d_results.pt \
    --body_models ~/GVHMR/inputs/checkpoints/body_models --src_fps 30 --save_path jumping_jacks_fixed.pkl   # GMR env
python 3_motion_to_policy/gmr_pkl_to_mjlab_csv.py --pkl jumping_jacks_fixed.pkl --csv jumping_jacks_fixed.csv
uv run src/mjlab/scripts/csv_to_npz.py --input-file jumping_jacks_fixed.csv --input-fps 30 --output-name r1_jumping_jacks_fixed --robot r1
```
