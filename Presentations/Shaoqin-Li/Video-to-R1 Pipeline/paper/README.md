# Paper

Draft in progress: a write-up of this pipeline, video → GVHMR → GMR → motion-tracking policy → sim2sim, on the Unitree R1.

- `draft.md`: working draft in Markdown (first-draft framing, with the seed study and the 2026-10-14 dance-timing correction).
- `video_to_r1_paper.docx`: the 2026-10-11 Word manuscript. Out of date on two points (no seed study; dance timing), see `latex/README.md`.
- `latex/`: the LaTeX version, the current manuscript: `main.tex`, `main.pdf`, `figures/`.
- `l4dc2027/`: the L4DC 2027 submission version (2026-10-19): refocused on delay randomization and sim-to-sim transfer, five motions, three training seeds per arm. `main.tex`, `main.pdf`, `figures/`. Data: `../results/l4dc_2026-10-19/`; figures and tables: `../results/make_l4dc_figures.py`, `make_l4dc_tables.py`; training and sweep scripts: `../3_motion_to_policy/mjlab_overlay/run_l4dc_*.sh`.
