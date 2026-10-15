# Three new clips, 2026-10-14: an unseen-video test of the whole pipeline

Three free stock videos picked for this test (two of them shot with a **moving camera**) went through every stage,
unchanged: GVHMR → GMR → mjlab tracking policy → sim2sim in plain MuJoCo. Simulation only, like everything else in this
folder.

## The clips (Pexels license: free to use and modify, credit given; raw clips not re-hosted here)

| Clip | Source | Camera | Cut used | Video fps | Robot motion |
|---|---|---|---|---|---|
| woman_home_fixed | ["A Woman Dancing At Home", Pexels 6003772](https://www.pexels.com/video/a-woman-dancing-at-home-6003772/) | fixed | 0.0–8.7 s (a jump cut to her sitting at 9.0 s), downscaled to 1920×1080 | 30 | 8.68 s |
| rdne_boxing_moving | ["Man Shadow Boxing", RDNE Stock project, Pexels 7187515](https://www.pexels.com/video/man-shadow-boxing-7187515/) | moving (follows him) | 3.0–18.0 s | 24 | 14.96 s |
| street_hiphop_moving | ["Person Dancing", Pexels 4540327](https://www.pexels.com/video/person-dancing-4540327/) | moving (handheld) | 0.0–15.0 s | 25 | 14.96 s |

They were chosen from 18 downloaded candidates. `1_video_to_smplx/triage_candidate_clips.py` counts people per
frame, checks whether the body touches the frame edge, and estimates camera motion; every candidate was then checked by eye.
Most stock boxing footage is upper-body close-ups, which is why the boxing clip is a backlit silhouette.

## What it took

- **Stage 1 (GVHMR).** The fixed clip ran unchanged. Both moving-camera clips crashed until GVHMR was patched
  (`1_video_to_smplx/gvhmr_ee960bb_pycolmap4.patch`, see SETUP): pycolmap marks near-static frame pairs as a
  "watermark" and returns no camera pose, and torch ≥ 2.6 refuses to reload GVHMR's own camera trajectory.
- **Stage 2 (GMR).** Retargeted with the real frame rate (`--src_fps 30 / 24 / 25`). Without it, the 24 and 25 fps clips
  would have played 1.25× and 1.2× too fast (the bug that affected the earlier dance clip, see the main README).
  Reference frames pinned at a joint limit: woman 23% (right ankle roll), street 23% (right ankle roll), **boxing 70%
  (both ankle rolls, 54–59%: the wide boxing stance)**.
- **Stage 3 (training).** The recipe validated by the seed study: 3,000 iterations of `Mjlab-Tracking-Flat-Unitree-R1`
  from scratch, then 3,000 more with `-Delay` (0–50 ms random command delay). 4,096 robots per run, with the three clips
  trained in parallel on the RTX 5080: 2 h 18 min for stage A and 2 h 43 min for stage B. Script:
  `3_motion_to_policy/mjlab_overlay/run_new_clips_2026-10-14.sh`.
  End of stage A, in mjlab: episode length 500 / 494 / 493 (of 500), body-position error 3.8 / 7.5 / 6.4 cm
  (woman / boxing / street). The woman clip learned slowest (episode length 76 at iteration 750). Its reference packs a
  knee lift, a deep squat and a hop into the first 3.5 s.

## Sim2sim results (plain MuJoCo, `4_sim2sim/sim2sim_v5_dr.py`)

Survival, stage A policy → after delay training:

| Scenario | woman (fixed) | boxing (moving) | street (moving) |
|---|---|---|---|
| nominal | 1/1 → 1/1 | 1/1 → 1/1 | 1/1 → 1/1 |
| training randomization (30) | 30 → 30 | 29 → 30 | 29 → 29 |
| + 20 ms delay (20) | 20 → 20 | 17 → 19 | 17 → 20 |
| **+ 40 ms delay (20)** | **0 → 20** | **0 → 19** | **1 → 19** |
| **+ 60 ms delay (20)** | **0 → 16** | **0 → 13** | **0 → 14** |
| + 3× pushes (20) | 15 → 18 | 5 → 7 | 3 → 8 |
| + foot friction 0.15 (20) | 15 → 15 | 2 → 3 | 14 → 18 |
| joint error, nominal (rad, L2 over 24 joints) | 0.746 → 0.610 | 1.159 → 0.752 | 0.827 → 0.588 |

**What it shows**
1. **The pipeline works on unseen video, including a moving camera.** All three policies complete their clip in a
   second simulator under the training randomization (29–30 of 30).
2. **The delay result repeats on three more motions.** At 40 ms almost nothing survives without delay training (0–1/20)
   and almost everything survives with it (19–20/20). This matches the tennis seed study, which showed the gain comes from
   the delay itself rather than the extra iterations. (Here the two stages also differ by 3,000 iterations, so this table
   alone could not separate the two.)
3. **Boxing is the weakest**: largest tracking error, and it falls under strong pushes and on a slippery floor. This fits
   its reference holding both ankles at their roll limits most of the time.

## Files

| Path | Contents |
|---|---|
| `videos/<clip>_side_by_side.mp4` | **original video next to the R1 doing the same motion**, frame-synchronized (delay-trained policy, nominal) |
| `stills/<clip>_side_by_side_frame<N>.png` | 3 frames per clip from those videos |
| `sweeps/` | sweep tables (`.txt`) and raw results (`.json`), stage A (`base`) and delay (`delay`) |
| `../../policies/new_clips_2026-10-14/` | 6 ONNX policies, final delay checkpoints, env/agent configs |
| `../../data/new_clips_2026-10-14/` | GVHMR outputs, GMR trajectories (`.pkl`, `.csv`), 50 Hz references (`.npz`) |

Side-by-side videos: `4_sim2sim/make_side_by_side.py`, camera azimuth 270 (woman), 210 (boxing) and 300 (street), picked by eye
from 12-angle renders.

**Caveats**
- One training seed per clip.
- Third-party stock clips, so this is not the phone-video test (paper §8).
- No hardware.
- The pinned-ankle fraction is measured on the retargeted reference; its effect on boxing is a likely explanation, not a tested one.
