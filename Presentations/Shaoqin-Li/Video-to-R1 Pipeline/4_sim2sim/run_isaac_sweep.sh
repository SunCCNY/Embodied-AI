#!/bin/bash
# usage: run_isaac_sweep.sh ONNX SCENARIOS OFFS...
cd ~/mjlab
onnx=$1; scn=$2; shift 2
for off in "$@"; do
  echo "OFF=$off"
  OFF=$off ONNX=$onnx TAG=_isaac_off$off timeout -s KILL 1500 .venv/bin/python sim2sim_isaac.py sweep $scn 2>&1 | tail -4
done
