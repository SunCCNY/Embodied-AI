# Week 32 — From a jump command to motion tracking

Prepared for Shaoqin Li, October 31, 2026. **35 slides, simulation only. Context-first revision for the Week 32 presentation.**

- `embodied ai week 32.pptx` — editable presentation with per-slide speaker notes.
- `week32.pdf` — rendered presentation for quick review.
- `week32_box_by_box.html` — standalone offline artifact: slide previews, exact text from all 337 PowerPoint text boxes, explanations, figure guides, narration, search and glossary. Open in a browser; no external assets or account required.
- `speaker_script_week32.md` — read-aloud script and Q&A preparation (approximately 25–30 minutes).
- `week32_presentation_package.zip` — deck, PDF, HTML, script, builder, figures and evidence notes.
- `preview/contact-sheet.png` — all rendered slides; `preview/slide-*.png` are individual renderings.
- `verified_metrics.json`, `evidence_manifest.json`, `audit_output.txt` — independently aggregated metrics, exact source hashes and actual saved-data audit output.

## Result boundary

The original command policy passed nominal 20/20, but T stayed 16/20, landing softening missed its target, and S4 lost the jump. No new command-policy candidate was promoted. P09 A1–A5 all failed full L1; A6 was skipped. A4 completed 20/20 clips, with no-fall 18/20, recovery 20/20 and joint RMS 18/20, but flight 0/20. Only 16/20 meet all non-flight gates together. This is partial progress, not a full skill PASS.

A4's gain is confounded with extra training and resumed state. Twenty evaluation episodes are not twenty training seeds. The c3 variant is labeled separately. Geometric reference flight and physical rollout flight are not equivalent measurements.

There is **no new rollout video**: the workstation did not render one. The deck uses numeric evidence and an actual local saved-data audit capture, not fabricated training output or an unrelated Week 31 video. LAFAN-derived motion, checkpoints, ONNX and videos are not included. Keep this working presentation private; no public upload was made.

## Evidence freshness and review

Local brain snapshot: `597391c` (2026-10-30 06:04:33 -0400). Bounded sync on October 31 timed out. The connected GitHub tool independently fetched the current default-branch P09 STATE and final report on October 31: the final report blob `0701f329b2f19726085bc312ff1d43924ce2b5ae` exactly matches the local file. The report says the sweep is complete and no further GPU rows are listed. This verifies the published handoff state, not a live workstation process query.

All A1–A5 raw episode counts, medians, gate totals and RMS summaries used in the deck were independently aggregated. P08 chart inputs were read from corrected raw episode files. The earlier S4 DIAGNOSED-not-repaired reward caveat remains. The A4 resume launcher now pins its two-hour cap before sourcing the shared library; observed A4 runtime is within the cap. Private binary hashes/parity are recorded workstation evidence; no new binary rehash or rollout was performed here.

See `EVIDENCE_REVIEW.md` for the final handoff audit. Sources live under `/home/sql/claude_brain/projects/embodied_ai_r1/video_to_r1_pipeline/results/` in `week_queue_2026-10-29` and `p09_s6_2026-10-29`.

## Rebuild

The data path is defined near the top of `build_week32.py`. It reads saved metrics only and never starts simulation or training.

```bash
MPLCONFIGDIR=/tmp/week32_mpl /home/sql/miniconda3/envs/isaac_eco/bin/python build_week32.py
/home/sql/opt/bin/soffice --headless --convert-to pdf --outdir preview 'embodied ai week 32.pptx'
python build_companion.py
```

The LibreOffice converter needed an approved execution outside the restricted sandbox to render successfully. All 35 rendered slides were visually reviewed, including the Week 31 background, diagrams, experiment definitions and retained training curves. PowerPoint text and speaker notes, PDF page count, companion box coverage and artifact links were also checked. No deck copy was made to Windows Downloads and no club-repository or paper edit was performed.

## Training-curve update (2026-10-31)

Slides 24–26 now show actual saved training diagnostics: mean episode duration across A1–A5, joint/anchor-velocity training errors for A2–A5, and A1 tolerance versus duration. They are training distributions, separately labeled from final strict evaluation. A4 is resumed, A5 is a variant and missing A1 RMS is not invented. No new training or video rendering was performed.

## Full-context revision (owner request #8812, 2026-10-31)

The talk now introduces the skill-bag project, explains Week 31 locomotion/jumping-jacks/switching results and their gaps, defines the continuous Week 32 goal, and separates command learning from motion imitation before reporting counts. It adds the missing true-running prerequisite and initial jump-height reward correction, explains reference preparation and the video-versus-mocap source difference, defines all success gates and motivates each attempt. Two Week 31 historical pictures are explicitly labeled; no Week 32 robot rollout image/video is fabricated. The matching companion and speaker script carry the additional definitions and context. Windows delivery for this revision is `Downloads\embodied_ai\week32_full_context`; earlier copies remain separate.
