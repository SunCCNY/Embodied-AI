#!/bin/bash
# Week 31 jump clips (Pexels 7299359 jumping jacks, 4812842 jump rope) through training, then sim2sim.
# Recipe validated by the seed study: 3,000 iterations of the plain R1 tracking task from scratch,
# then 3,000 more with 0-50 ms random command delay. The clips train in parallel.
cd ~/mjlab
export WANDB_MODE=disabled
CLIPS="jumping_jacks_fixed jump_rope_fixed"
COMMON="--env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard --video False"
log() { echo "[$(date +%F' '%T)] $*"; }

log "stage A: plain task from scratch"
for c in $CLIPS; do
  uv run train Mjlab-Tracking-Flat-Unitree-R1 --env.commands.motion.motion-file motions/new_clips/$c.npz \
    $COMMON --agent.run-name ${c}_base > logs/jumpclip_${c}_base.log 2>&1 &
  sleep 20
done
wait; log "stage A done"

log "stage B: delay fine-tune"
for c in $CLIPS; do
  run=$(basename $(ls -d logs/rsl_rl/r1_tracking/*_${c}_base | tail -1))
  ckpt=$(ls logs/rsl_rl/r1_tracking/$run/model_*.pt | sort -V | tail -1 | xargs basename)
  log "$c resumes $run/$ckpt"
  uv run train Mjlab-Tracking-Flat-Unitree-R1-Delay --env.commands.motion.motion-file motions/new_clips/$c.npz \
    $COMMON --agent.run-name ${c}_delay --agent.resume True --agent.load-run $run --agent.load-checkpoint $ckpt \
    > logs/jumpclip_${c}_delay.log 2>&1 &
  sleep 20
done
wait; log "stage B done"

mkdir -p jump_clips_eval
SCN=nominal,train_DR,train_DR+latency1,train_DR+latency2,train_DR+latency3,train_DR+push3x,train_DR+fric0.15
for c in $CLIPS; do
  for stage in base delay; do
    d=$(ls -d logs/rsl_rl/r1_tracking/*_${c}_${stage} | tail -1); cp $d/*.onnx jump_clips_eval/${c}_${stage}.onnx
    log "sweep $c $stage"
    ONNX=jump_clips_eval/${c}_${stage}.onnx MOTION=motions/new_clips/$c.npz TAG=_jumpclip_${c}_${stage} \
      .venv/bin/python sim2sim_v5_dr.py sweep $SCN > jump_clips_eval/sweep_${c}_${stage}.txt 2>&1
    mv results_v5_dr_subset_jumpclip_${c}_${stage}.json jump_clips_eval/
  done
done
log "ALL DONE"
