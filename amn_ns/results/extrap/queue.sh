#!/bin/bash
# complex split then low-data curve, AMN-NS (ReLU encoder) and MLP, 3 seeds
cd "$(dirname "$0")/../.."
P=python
for s in 1 2 3; do
  TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 bash -c "
    for sp in complex lowdata; do
      $P -u amn_nullspace.py --model ns  --split \$sp --encoder relu --param null --plateau --epochs 100 --seed $s --tag \${sp}_ns_s$s  --out results/extrap
      $P -u amn_nullspace.py --model mlp --split \$sp --seed $s --tag \${sp}_mlp_s$s --out results/extrap
    done" > results/extrap/s$s.log 2>&1 &
done
wait
date > results/extrap/finished
