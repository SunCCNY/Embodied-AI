#!/bin/bash
# Walk/run delay study (2026-10-23): after the delay fine-tune (Unitree-R1-Flat-Delay-FT, +3000 from
# model_10000) finishes, train the matched no-delay control (Unitree-R1-Flat-FT, +3000 from the same
# checkpoint), then evaluate base / control / delay: in-training-sim speed tracking, ONNX export,
# CPU sim2sim sweep over speeds x constant command delays, and a video of the delay policy.
cd ~/unitree_rl_mjlab
log() { echo "[$(date +%F' '%T)] $*"; }
BASE_RUN=2026-10-21_21-54-28_walk_run_flat
OUT=logs/walk_delay_study_2026-10-23
mkdir -p $OUT
ORIG_LD="$LD_LIBRARY_PATH"
export MUJOCO_GL=osmesa LD_LIBRARY_PATH=$HOME/local_osmesa/extracted/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

log "waiting for the delay fine-tune to exit"
while pgrep -f "^.venv/bin/python scripts/train.py Unitree-R1-Flat-Delay-FT" > /dev/null; do sleep 60; done
log "delay fine-tune exited: $(grep -a 'Learning iteration' logs/walk_run_delay_ft_train.log | tail -1)"

log "control fine-tune (no delay) starts"
env -u MUJOCO_GL LD_LIBRARY_PATH="$ORIG_LD" .venv/bin/python scripts/train.py Unitree-R1-Flat-FT --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run $BASE_RUN --agent.load-checkpoint model_10000.pt --agent.max-iterations 3000 \
  --agent.run-name walk_run_control_ft --agent.logger tensorboard > logs/walk_run_control_ft_train.log 2>&1 < /dev/null
log "control fine-tune exited: $(grep -a 'Learning iteration' logs/walk_run_control_ft_train.log | tail -1)"

last_ckpt() { ls logs/rsl_rl/r1_velocity/$1/model_*.pt | sort -V | tail -1; }
DELAY_RUN=$(basename $(ls -d logs/rsl_rl/r1_velocity/*_walk_run_delay_ft | tail -1))
CTRL_RUN=$(basename $(ls -d logs/rsl_rl/r1_velocity/*_walk_run_control_ft | tail -1))
declare -A CK=( [base]=logs/rsl_rl/r1_velocity/$BASE_RUN/model_10000.pt
                [control]=$(last_ckpt $CTRL_RUN) [delay]=$(last_ckpt $DELAY_RUN) )
for k in base control delay; do log "$k = ${CK[$k]}"; done

for k in base control delay; do
  log "speed tracking + ONNX export: $k"
  .venv/bin/python scripts/eval_walk_run.py --ckpt ${CK[$k]} --tag walk_run_$k --out-dir $OUT \
    --speeds=-0.5,0,0.5,1.0,1.5,2.0 --export-onnx > $OUT/eval_$k.log 2>&1
done

for k in base control delay; do
  log "CPU sim2sim sweep: $k"
  .venv/bin/python scripts/sim2sim_cpu.py sweep --onnx $OUT/walk_run_$k.onnx --speeds 0,0.5,1.0,1.5,2.0 \
    --delays 0,20,30,40,50,60,80 --seeds 10 --seconds 20 --out $OUT/sim2sim_$k.json > $OUT/sim2sim_$k.txt 2>&1
done

log "video: delay policy"
.venv/bin/python scripts/eval_walk_run.py --ckpt ${CK[delay]} --tag walk_run_delay_video --out-dir $OUT \
  --speeds 0 --envs-per-speed 4 --video-schedule 3:0,7:0.5,11:1.0,15:1.5,19:2.0,23:0 > $OUT/video_delay.log 2>&1
log "ALL DONE"
