#!/bin/bash
# nested tuning of AMN-NS (linear encoder): beta x uptake x epochs, one job per outer fold
# (seed 1 only: stopped after it showed no gain over the defaults, see README 5.2)
cd "$(dirname "$0")/../.."
for s in 1; do
  for f in fold0 fold1 fold2 fold3 fold4; do echo "random $f $s"; done
  for f in le10 le12 le14; do echo "complex $f $s"; done
done | xargs -P 8 -L 1 bash -c 'TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 python -u amn_nullspace.py --model ns --split $0 --only $1 --seed $2 --tune --betas 0.03 0.1 0.3 --uptakes 5 6.5 10 --epochs-grid 100 200 --plateau --tag tune_$0_s$2_$1 --out results/tune >> results/tune/log_$0_s$2.txt 2>&1'
date > results/tune/finished
