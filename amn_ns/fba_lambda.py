#!/usr/bin/env python
"""
FBA baseline as in Faure et al. (2023): every medium compound that is ON
(minimal medium, including O2, and present variable nutrients) gets the
uptake upper bound lambda, compounds that are OFF get 0, and the biomass flux
is maximised. lambda is scanned over a grid.

Reports
  - the best R2 over lambda on all media (published protocol), and
  - a cross-validated R2: lambda chosen on the training folds only
    (5-fold KFold, same folds as amn_nullspace.py --seed).

Usage:
    python fba_lambda.py --seed 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from Library.Build_Model import Neural_Model, model_input

ap = argparse.ArgumentParser()
ap.add_argument('--lambdas', type=float, nargs='+',
                default=list(np.round(np.arange(0.5, 20.01, 0.5), 2)))
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='./results/fba_lambda')
ap.add_argument('--trainingfile', default='./Dataset_input/Ecoli/iML1515EXP',
                help='dataset path without extension: <path>.npz (training set) '
                     'and <path>.xml (GEM)')
ap.add_argument('--uptake-prefix', default='EX_', help='id prefix of uptake reactions')
ap.add_argument('--uptake-suffix', default='_i', help='id suffix of uptake reactions')
args = ap.parse_args()

param = Neural_Model(model_type='AMN_QP', scaler=True, objective=True,
                     n_hidden=1, hidden_dim=1000, output_dim=1,
                     activation='relu', scoring_function=r2_score,
                     xfold=1, niter=0, batch_size=10,
                     trainingfile=args.trainingfile,
                     verbose=False)
X, Y = model_input(param, verbose=False)
y = Y[:, 0]
m = param.cobramodel
m.solver.configuration.timeout = 30  # seconds; a stalled LP counts as no growth
ids = [r.id for r in m.reactions]
var_ids = [ids[j] for j in np.argmax(param.Pin, axis=1)]  # one per X column
min_ids = [r for r in param.medium if r not in var_ids]   # always ON
uptakes = [r.id for r in m.reactions if r.id.startswith(args.uptake_prefix) and r.id.endswith(args.uptake_suffix)]

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
G = np.zeros((len(args.lambdas), len(X)))  # FBA growth per lambda and medium
for a, lam in enumerate(args.lambdas):
    with m:
        for rid in uptakes:
            m.reactions.get_by_id(rid).upper_bound = lam if rid in min_ids else 0.0
        for i in range(len(X)):
            for k, rid in enumerate(var_ids):
                m.reactions.get_by_id(rid).upper_bound = lam * X[i, k]
            G[a, i] = m.slim_optimize(error_value=0.0)
    print(f'lambda={lam:5.2f}  R2={r2_score(y, G[a]):7.3f}  '
          f'pearson={np.corrcoef(y, G[a])[0, 1]:.3f}', flush=True)
pd.DataFrame(G.T, columns=[f'{l:g}' for l in args.lambdas]).assign(y=y).to_csv(
    out / 'fba_growth_by_lambda.csv', index=False)

r2 = np.array([r2_score(y, g) for g in G])
best = int(np.argmax(r2))
print(f'\npublished protocol: best lambda={args.lambdas[best]:g}  R2={r2[best]:.3f}')

pred = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(X):
    a = int(np.argmax([r2_score(y[tr], g[tr]) for g in G]))
    pred[te] = G[a, te]
    print(f'  fold: lambda chosen on training = {args.lambdas[a]:g}')
print(f'cross-validated (lambda chosen on training folds): R2={r2_score(y, pred):.3f}')
