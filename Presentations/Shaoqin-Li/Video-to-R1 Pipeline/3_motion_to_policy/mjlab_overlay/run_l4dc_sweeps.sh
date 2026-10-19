#!/bin/bash
# L4DC: wait for the training queue, then sweep every policy (5 motions x 2 arms x 3 seeds) with the fine delay grid.
cd ~/mjlab
log() { echo "[$(date +%F' '%T)] $*"; }
until grep -q "ALL TRAINING DONE" l4dc_driver.log; do sleep 120; done
log "training done; sweeping"
OUT=l4dc_eval; mkdir -p $OUT
RUNS=logs/rsl_rl/r1_tracking
SCN=train_DR,train_DR+delay0ms,train_DR+delay10ms,train_DR+delay20ms,train_DR+delay30ms,train_DR+delay40ms,train_DR+delay50ms,train_DR+delay60ms,train_DR+delay70ms,train_DR+delay80ms,train_DR+push3x,train_DR+fric0.15,train_DR+kp-20%,train_DR+payload2kg,train_DR+no_lin_vel

sweep() {  # motion_name onnx motion_npz arm seed
  local tag=${1}_${4}_s${5}
  [ -f $OUT/$tag.json ] && { log "skip $tag"; return; }
  cp $2 $OUT/$tag.onnx
  ONNX=$OUT/$tag.onnx MOTION=$3 TAG=_l4dc_$tag .venv/bin/python sim2sim_v5_dr.py sweep $SCN > $OUT/$tag.txt 2>&1
  mv results_v5_dr_subset_l4dc_$tag.json $OUT/$tag.json; log "done $tag"
}
newest_onnx() { ls $RUNS/*_$1/*.onnx | tail -1; }

# tennis: the 2026-10-09 seed study (control = seed_study/control_s*, delay = seed_study/delay_s*)
for s in 42 43 44; do
  sweep tennis seed_study/control_s$s.onnx artifacts/r1_tennis:v0/motion.npz ctrl $s
  sweep tennis seed_study/delay_s$s.onnx   artifacts/r1_tennis:v0/motion.npz dly  $s
done
# woman / boxing / street: delay seed 42 = the 2026-10-14 stage-B run
for c in woman_home_fixed rdne_boxing_moving street_hiphop_moving; do
  for s in 42 43 44; do sweep $c $(newest_onnx ${c}_ctrl_s$s) motions/new_clips/$c.npz ctrl $s; done
  sweep $c $(newest_onnx ${c}_delay) motions/new_clips/$c.npz dly 42
  for s in 43 44; do sweep $c $(newest_onnx ${c}_dly_s$s) motions/new_clips/$c.npz dly $s; done
done
# dance at true speed
for s in 42 43 44; do
  sweep dancer_truespeed $(newest_onnx dancer_truespeed_ctrl_s$s) motions/new_clips/dancer_truespeed.npz ctrl $s
  sweep dancer_truespeed $(newest_onnx dancer_truespeed_dly_s$s)  motions/new_clips/dancer_truespeed.npz dly  $s
done
# base policies (before the +3,000 iterations), for reference
sweep dancer_truespeed $(newest_onnx dancer_truespeed_base) motions/new_clips/dancer_truespeed.npz base 42
for c in woman_home_fixed rdne_boxing_moving street_hiphop_moving; do sweep $c $(newest_onnx ${c}_base) motions/new_clips/$c.npz base 42; done
sweep tennis logs/rsl_rl/r1_tracking/2026-09-24_17-01-04_r1_tennis_run1_continued/2026-09-24_17-01-04_r1_tennis_run1_continued.onnx artifacts/r1_tennis:v0/motion.npz base 42
log "ALL SWEEPS DONE"
