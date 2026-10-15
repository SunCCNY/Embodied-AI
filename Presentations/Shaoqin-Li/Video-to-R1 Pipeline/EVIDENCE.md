# Evidence map: paper claim → file

For whoever writes the paper or slides. Every picture, video and number behind `paper/draft.md`, with the claim it supports.
Everything is in this folder on GitHub:
**https://github.com/SunCCNY/Embodied-AI/tree/main/Presentations/Shaoqin-Li/Video-to-R1%20Pipeline**
(the claude_brain mirror has only the text files; download images and videos from GitHub, e.g.
`gh api "repos/SunCCNY/Embodied-AI/contents/Presentations/Shaoqin-Li/Video-to-R1%20Pipeline/results/figures/<file>" --jq .download_url`).

Status: simulation only, two third-party clips. No phone video and no hardware yet (paper §8). Do not caption anything as real-robot.

## NEW 2026-10-14: three unseen stock clips (this week's presentation)

Start with `results/new_clips_2026-10-14/README.md`. Woman at home (fixed camera), boxing silhouette (**moving camera**)
and street hip-hop (**moving camera**): Pexels clips, trained with the seed-study recipe.
- **Best slide material:** `results/new_clips_2026-10-14/videos/<clip>_side_by_side.mp4`, the original video next to the R1
  doing the same motion, frame-synchronized. Stills: `results/new_clips_2026-10-14/stills/`.
- **Headline numbers:** all three complete their clip in plain MuJoCo under training randomization (29–30/30). At 40 ms delay,
  0–1/20 survive without delay training and 19–20/20 with it. Boxing is weakest (largest error; falls under strong pushes
  and on a slippery floor).
- Credit on the slide: videos by Pexels contributors (links in that README).

## Paper-ready figures (`results/figures/`)

| File | Shows | Supports | Made by |
|---|---|---|---|
| `pipeline_strip.png` | the 5 stages: video → GVHMR → GMR → RL tracking → sim2sim | §4, overview figure | `poster_gsoe_fall2026` matplotlib |
| `video_frame_116.png` + `r1_frame_116.png` | tennis serve, ball toss: original video vs R1 policy in MuJoCo at the same moment | §1/§4, "the R1 reproduces video motion" | `4_sim2sim/render_poster_frames.py` |
| `video_frame_141.png` + `r1_frame_141.png` | tennis serve, follow-through (body lean, back leg up) | same | same |
| `fig_dance_video_vs_r1.png` | dance: lunge (tracks closely) and arms overhead (R1 bends its arms: reference pinned at shoulder limits) | §6, §7.3 | render script + Pillow compose |
| `dance_video_frame_{320,560}.png`, `dance_r1_frame_{320,560}.png` | the single panels of the figure above | same | same |
| `fig_ablation.png` | joint error when one raw-MJCF setting is restored: only "no floor" (6.97) and "timestep" (1.72) matter | §6.1, §7.1 (Table 4) | `results/make_figures.py` |
| `fig_stress.png` | survival per stress scenario, tennis policy vs delay-trained | §6.2 (Table 5) | `results/make_figures.py` |
| `fig_delay_seeds.png` | survival vs added delay, 3 seeds per arm: control (no delay) vs delay-trained | §6.3, §6.4 (Table 6): **gain is from delay, not extra training** | `results/make_figures.py` |
| `fig_delay_before_after.png` | 40 ms delay: no-delay policy falling vs delay-trained policy at the same moment, tennis (mjlab) and dance (Isaac) | §6.3 | `results/make_frame_figures.py` |
| `fig_dance_shoulder_limits.png` | left-shoulder pitch/roll/yaw over the dance: reference vs R1, joint limits, frames pinned at a limit shaded | §7.3 | `results/make_figures.py` from `results/diagnostics/dance_left_shoulder.npz` |
| `stills/poster_r1_tennis_sim2sim.png` | the ORIGINAL buggy harness: robot with no floor | §6.1, "the first failure was the harness" | video first frame |
| `stills/poster_*.png` (others) | first frames of each result video (mostly the start pose; weak as evidence) | slide thumbnails | video first frames |

Training curves: `results/plots/training_curves.png` (mjlab tennis), `results/plots/isaac_training_curves.png` and
`isaac_training_curves_both.png` (Isaac dance, with and without delay).

## Videos (`results/videos/`, 50 fps, grey = robot, green = reference unless noted)

| File | Shows | Supports |
|---|---|---|
| `r1_tennis_tracking.mp4`, `r1_dancer_tracking.mp4` | policies inside the training engine (mjlab) | §6, Table 3 |
| `r1_tennis_sim2sim_trainDR.mp4` | tennis in plain MuJoCo with training randomization: completes the clip | §6.2 |
| `r1_tennis_sim2sim.mp4` | the original buggy harness (no floor, 125 Hz) | §6.1 |
| `r1_tennis_sim2sim_latency2_FAIL.mp4` | +40 ms delay, policy without delay training: falls at 1.56 s | §6.3 |
| `r1_tennis_sim2sim_latency2_delaytrained.mp4` | +40 ms delay, delay-trained: completes | §6.3 |
| `r1_dance_mjlab_in_mujoco.mp4`, `r1_dance_isaac_in_mujoco.mp4` | dance, mjlab- vs Isaac-trained policy in plain MuJoCo | §6.2, cross-engine |
| `r1_dance_isaac_latency2_nodelay.mp4` / `_delaytrained.mp4` | Isaac dance at +40 ms: falls at 1.18 s vs completes the clip (22.1 s of robot time) | §6.3 |

## Numbers (`results/`)

| File | Contents | Paper |
|---|---|---|
| `sweeps/mjlab_tennis_baseline_full.json` | tennis policy, all 17 scenarios | Table 5 col 1 |
| `sweeps/mjlab_tennis_delay_latency.json`, `_delay_stress.json` | delay-trained tennis | Table 5 col 2 |
| `sweeps/mjlab_dance_baseline.json`, `isaac_dance_baseline.json`, `isaac_dance_delay.json` | dance policies | Table 5 cols 3–5 |
| `sweeps/seed_study/*.json`, `sweep_*.txt` | 6 policies × 7 scenarios | Table 6 |
| `terminal_captures/floor_timestep_ablation.txt` | ablation | Table 4 |
| `terminal_captures/evaluate_results.txt`, `dance_native_eval.txt` | in-engine success (mjlab evaluate) | Table 3 |
| `measurements/` | GVHMR timing, GMR plausibility, ground pass | §4.1–4.2 |
| `diagnostics/dance_left_shoulder.npz` | per-frame shoulder angles, limits, pinned mask, hand error | §7.3 |

## Things not to claim

- No phone video and no physical-robot result exists yet.
- "Joint error" is the L2 norm over 24 joints (mjlab's `error_joint_pos`), not a per-joint error. Per-joint RMS = value / √24.
- The tennis clip is a **serve** (GVHMR's example clip), not a forehand.
- **The dance reference runs 1.2× faster than the dancer** (25 fps video treated as 30 fps, found 2026-10-14). Don't say
  "the R1 dances at the dancer's speed"; the dance sequence and poses are right, the timing is compressed. Times on the dance
  figures are video times.
- The Isaac delay policy's drop (29/30 → 17/30) is a single run and unexplained.
- Most policies are single training seeds. Only the mjlab delay result has 3 seeds.
