#!/bin/bash
# Run from the mjlab root with PIPE set (see SETUP.md). Resumes from policies/mjlab_tennis/model_12998.pt,
# which must sit in logs/rsl_rl/r1_tracking/2026-09-24_17-01-04_r1_tennis_run1_continued/.
cd ~/mjlab
exec uv run train Mjlab-Tracking-Flat-Unitree-R1-Delay \
  --env.commands.motion.motion-file "$PIPE/data/r1_motions/tennis_motion.npz" \
  --agent.max-iterations 3000 --agent.logger tensorboard \
  --agent.run-name r1_tennis_delay_ft \
  --agent.resume True --agent.load-run 2026-09-24_17-01-04_r1_tennis_run1_continued \
  --agent.load-checkpoint model_12998.pt --env.scene.num-envs 4096 --video False
