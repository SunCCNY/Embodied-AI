# Week 32 completed-data review — 2026-10-31

This review supplies the evidence for the presentation; it does not authorize new training.

## P09 result verification

All raw rows have exactly seeds 7000–7019. Original c2 A1/A2/A3/A4 full L1 counts: 0/20 each. c3 A5 variant: 0/20 against its own 0.32-second anchor. A6 is SKIPPED, not a failed training row or reused slot. No full result passed, so no confirmation bank was run.

| Row | Completes | No-fall | Flight | Recovery | RMS gate | Median survival s | Mean joint RMS rad |
|---|---|---|---|---|---|---|---|
| A1 | 0 | 0 | 0 | 0 | 0 | 0.83 | 0.338915 |
| A2 | 20 | 18 | 0 | 10 | 0 | 5.26 | 0.178046 |
| A3 | 0 | 0 | 0 | 0 | 0 | 0.86 | 0.347353 |
| A4 | 20 | 18 | 0 | 20 | 18 | 5.26 | 0.147974 |
| A5 variant | 0 | 0 | 0 | 0 | 0 | 1.22 | 0.321007 |

A4 all-non-flight intersection is 16/20, not 18/20 or 20/20. Thus “flight only” describes the common failed gate, not every episode's sole defect. The deck makes this distinction. A4 is a useful failed parent candidate, never a confirmed complete skill. Its extra 3,000 updates, seed change and resumed lineage confound attribution to velocity weighting alone.

Recorded per-row actor-export discrepancies are below 1e-5. Private checkpoints/ONNX were not available locally for independent binary rehash or re-evaluation; claims are scoped to the saved parity records. The geometric 0.22-second c2 reference flight is not physically equivalent to rollout flight and does not establish impossibility. Gates remain unchanged.

## Budget and implementation

A1 3935 + A2 4595 + A3 4265 + A4 4325 + A5 4025 = 21145 seconds, or 5.87361 GPU-hours, below the 12-hour campaign cap. Every individual duration is below two hours. The supplied launchers specify bounded jobs, reference pins, W&B off and one-GPU admission/lock controls. A4 launcher v2 resolves the previously reviewed five-hour default by pinning no more than 7200 seconds before library initialization; actual A4 elapsed is 4325 seconds. No additional job is requested or launched here.

P08 accounting remains separate: 17714 seconds, about 4.92 GPU-hours. model_20496 remains the original command-policy fallback because S3 missed landing max <=7 BW and S4 lost the jump. The presentation retains timing and latency failures and the apex-reward push caveat.

## Deliverable checks

Charts were regenerated from saved episode JSONs, using the source hashes in evidence_manifest.json. No numeric chart was fabricated from a narrative report. The terminal image is the real output of this local saved-data aggregation, labeled as an audit rather than training. The HTML companion reads exact PowerPoint text boxes and explains 337 boxes over 35 slides. The deck includes no equations, private motion/video/model assets, hardware claims or AI attribution.

The current remote P09 final report was verified against the local Git blob through the connected GitHub tool after the filesystem sync timed out. Its unchanged ledger says the sweep is complete. Live workstation state and unpublished experiments were not queried.

## Training curves added after owner feedback

Three slides use the actual train_diag_every250.json files, now included in evidence_manifest.json. Curves show native training episode duration, joint RMS and anchor-velocity error, and A1 logged tolerance. Curves are not strict evaluations; x is updates since first saved diagnostic per row, including A4 resumed state. No missing A1 training RMS was fabricated. No rollout videos were produced or embedded.

## Context revision evidence

Week 31 background follows its accepted speaker script. Two unchanged Week 31 poster frames are included only as explicitly labeled historical illustrations, never Week 32 evidence. Added P04 files: raw/runftv2_model_14498_vx2.json (20/20 true-running), raw/jump_model_17497_jump_vx2.json (0/20, height gate failed), and raw/jumpapex_model_20496_jump_vx2.json (20/20). They explain the prerequisite and engineering correction before the P08 robustness study. Reward-definition/weight changes and extra training are disclosed, not represented as an isolated ablation. P04 README/JUMP_STAGE and the Week 31 script/poster hashes are now recorded in the evidence manifest.
