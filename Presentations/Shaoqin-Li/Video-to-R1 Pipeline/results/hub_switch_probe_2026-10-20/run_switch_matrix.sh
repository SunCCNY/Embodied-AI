#!/bin/bash
# usage: run_switch_matrix.sh <scenario> <cut> ; writes switch_probe_out/matrix_<scn>_cut<cut>.log
cd ~/mjlab
SCN=${1:-train_DR}; export CUT=${2:-1.0}
jobs=()
for pair in "tennis dance" "dance tennis"; do
  for v in direct bridge stand; do jobs+=("$pair $v $SCN 20"); done
done
if [ "$CUT" = "1.0" ]; then jobs+=("tennis tennis alone $SCN 20" "dance dance alone $SCN 20"); fi
printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c 'uv run python switch_probe.py {} 2>&1 | grep -v "^\[stand\]"' | sort | tee switch_probe_out/matrix_${SCN}_cut${CUT}.log
