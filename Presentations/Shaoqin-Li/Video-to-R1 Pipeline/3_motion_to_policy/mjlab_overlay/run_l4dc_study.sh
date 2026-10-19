#!/bin/bash
# L4DC revision: seed study (control = +3,000 plain iterations, delay = +3,000 with 0-50 ms random command delay)
# on four more motions, each arm 3 training seeds from the same checkpoint, plus the dance retrained at true speed.
# At most 3 runs at once on the RTX 5080. Sweeps run separately (fine delay grid) once the harness extension is verified.
cd ~/mjlab
export WANDB_MODE=disabled
LOGD=logs/l4dc; mkdir -p $LOGD
RUNS=logs/rsl_rl/r1_tracking
COMMON="--env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard --video False"
log() { echo "[$(date +%F' '%T)] $*"; }

job() {  # task motion runname seed [resume_run]
  local task=$1 motion=$2 name=$3 seed=$4 resume=$5 extra=""
  if [ -n "$resume" ]; then
    local ck=$(ls $RUNS/$resume/model_*.pt | sort -V | tail -1 | xargs basename)
    extra="--agent.resume True --agent.load-run $resume --agent.load-checkpoint $ck"
  fi
  echo "uv run train $task --env.commands.motion.motion-file $motion $COMMON --agent.seed $seed --env.seed $seed --agent.run-name $name $extra > $LOGD/$name.log 2>&1; echo \"[\$(date +%F' '%T)] done $name exit \$?\""
}
base_of() { basename $(ls -d $RUNS/*_$1_base | tail -1); }

log "phase 1: dance base (true speed) + seed study for woman / boxing / street"
{
  job Mjlab-Tracking-Flat-Unitree-R1 motions/new_clips/dancer_truespeed.npz dancer_truespeed_base 42
  for c in woman_home_fixed rdne_boxing_moving street_hiphop_moving; do
    b=$(base_of $c)
    for s in 42 43 44; do job Mjlab-Tracking-Flat-Unitree-R1 motions/new_clips/$c.npz ${c}_ctrl_s$s $s $b; done
    for s in 43 44; do job Mjlab-Tracking-Flat-Unitree-R1-Delay motions/new_clips/$c.npz ${c}_dly_s$s $s $b; done
  done
} > $LOGD/phase1.jobs
xargs -P 3 -d '\n' -I{} bash -c '{}' < $LOGD/phase1.jobs
log "phase 1 done"

log "phase 2: dance seed study"
b=$(base_of dancer_truespeed)
{
  for s in 42 43 44; do job Mjlab-Tracking-Flat-Unitree-R1 motions/new_clips/dancer_truespeed.npz dancer_truespeed_ctrl_s$s $s $b; done
  for s in 42 43 44; do job Mjlab-Tracking-Flat-Unitree-R1-Delay motions/new_clips/dancer_truespeed.npz dancer_truespeed_dly_s$s $s $b; done
} > $LOGD/phase2.jobs
xargs -P 3 -d '\n' -I{} bash -c '{}' < $LOGD/phase2.jobs
log "ALL TRAINING DONE"
