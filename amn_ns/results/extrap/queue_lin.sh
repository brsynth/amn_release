#!/bin/bash
# linear-encoder AMN-NS (CPU): complex split seeds 1-5, low-data and random CV seeds 1-3
cd "$(dirname "$0")/../.."
P="python -u amn_nullspace.py --model ns --encoder linear --param null --plateau --epochs 100 --out results/extrap"
for s in 1 2 3; do
  TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 bash -c "
    $P --split complex --seed $s --tag complex_nslin_s$s
    [ $s -le 2 ] && $P --split complex --seed \$(($s + 3)) --tag complex_nslin_s\$(($s + 3))
    $P --split lowdata --seed $s --tag lowdata_nslin_s$s
    $P --split random --seed $s --tag random_nslin_s$s" > results/extrap/lin_s$s.log 2>&1 &
done
wait
date > results/extrap/finished_lin
