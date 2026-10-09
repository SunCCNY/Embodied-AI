# Figures

Matched frames for posters and slides: the original tennis-serve video next to the R1 doing the same moment.
The R1 images are the trained tennis policy (`policies/mjlab_tennis`) running in plain MuJoCo physics, rendered at the
same point in the clip. They are not kinematic replays.

| File | Moment |
|---|---|
| `video_frame_116.png` / `r1_frame_116.png` | ball toss (video frame 116) |
| `video_frame_141.png` / `r1_frame_141.png` | follow-through (video frame 141) |
| `pipeline_strip.png` | the five pipeline stages |

Video: GVHMR's example clip (`GVHMR/docs/example_video/tennis.mp4`). Frame *i* of the 30 fps video maps to harness step
round(i·50/30) − 1. Re-render with `4_sim2sim/render_poster_frames.py`, run from the mjlab root with `MUJOCO_GL=osmesa`:
`ONNX=<policy.onnx> python render_poster_frames.py <out_dir> final 170 116,141` (170° = camera azimuth matching the video).
