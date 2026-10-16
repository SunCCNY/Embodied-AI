# Paper, LaTeX version

`main.tex` is the LaTeX source of the manuscript "From Video to Humanoid: A Simulation Pipeline for Motion Imitation
on the Unitree R1, and What Breaks It in Transfer" (Shaoqin Li, Yi Sun). `main.pdf` is the compiled result (17 pages).

It is the 2026-10-11 Word manuscript (`../video_to_r1_paper.docx`) converted to LaTeX, with two corrections carried over
from the later evidence (see `../draft.md` and `../../EVIDENCE.md`):

1. **Seed study** (Section 7.4, Table 7, Figure 8): three training seeds per arm show the 40 ms delay gain comes from the
   delay randomization, not from the extra 3,000 iterations. The Word version said no seed study had been run.
2. **Dance timing:** the dance clip is 26.5 s at 25 fps, and GMR's GVHMR loader assumes 30 fps, so the dance reference
   plays 1.2x too fast (22.1 s). All dance results are for that reference. Corrected 2026-10-14.

It also fixes two details found while converting: Figure 3 now points at the training-randomization video
(`r1_tennis_sim2sim_trainDR.mp4`, which completes the clip; `r1_tennis_sim2sim.mp4` is the original harness with no floor), and
the harness termination rule names the dimensionless gravity-direction criterion separately from the 0.25 m height thresholds.
The three moving-camera clips of 2026-10-14 are not part of this paper.

Compile with `tectonic main.tex` (or `pdflatex main.tex` twice). Figures are in `figures/`.
