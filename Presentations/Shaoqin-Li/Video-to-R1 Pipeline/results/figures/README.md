# Figures

See [../../EVIDENCE.md](../../EVIDENCE.md) for what each file shows and which paper claim it supports.

- Charts: `fig_ablation.png`, `fig_stress.png`, `fig_delay_seeds.png`, `fig_dance_shoulder_limits.png`. Regenerate with `results/make_figures.py`.
- Frame figures: `fig_delay_before_after.png` (`results/make_frame_figures.py`), `fig_dance_video_vs_r1.png`.
- Video vs R1 pairs: tennis `video_frame_116/141` + `r1_frame_116/141`; dance `dance_video_frame_320/560` + `dance_r1_frame_320/560`.
  The R1 images are trained policies running in plain MuJoCo physics, rendered at the same moment of the clip (not kinematic replays).
  Re-render with `4_sim2sim/render_poster_frames.py` from the mjlab root, with `MUJOCO_GL=osmesa`:
  `ONNX=<policy.onnx> MOTION=<motion.npz> python render_poster_frames.py <out_dir> final <azimuth> <video frames>`.
  Tennis: azimuth 170, frames 116,141. Dance: azimuth 300, `RW=900 RH=1600`, frames 320,560. Video frame *i* → harness step round(i·50/30) − 1.
- `pipeline_strip.png`: the 5 stages.
- `stills/`: first frames of the result videos. `poster_r1_tennis_sim2sim.png` shows the original no-floor harness bug.

Videos: GVHMR's example clips (`GVHMR/docs/example_video/tennis.mp4`, `dancer.mp4`). The dance clip is "Woman Dancing" by Antoni Shkraba (Pexels).
