#!/usr/bin/env python
"""
Ablation: does the mechanistic constraint (SV = 0, Pin.V <= Vin) help AMN?

Same AMN network, same data / folds / subsamples as learning_curve_unified.py;
only the weight `lam` of the constraint terms in the loss changes:
    loss = mean( w * (y_true - y_pred)^2 ),  w = [1, lam, lam, lam]
    lam = 1  -> original AMN loss (my_mse)
    lam = 0  -> unconstrained: growth error only, same architecture
    lam > 1  -> constraints enforced more strongly

Besides the collated growth R², records how well the predicted flux vectors
V of the test set satisfy the constraints (SV residual, Pin violation).

Usage:
    python ablation_sv.py --lam 0 --tag unconstrained
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import Library.Build_Model as BM
from Library.Build_Model import Neural_Model, model_input, train_model

ap = argparse.ArgumentParser()
ap.add_argument('--lam', type=float, required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--sizes', type=int, nargs='+', default=[10, 20, 50, 100])
ap.add_argument('--draws', type=int, default=3)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--epochs', type=int, default=1000)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='./results/ablation')
ap.add_argument('--trainingfile', default='./Dataset_input/Ecoli/iML1515EXP',
                help='dataset path without extension: <path>.npz (training set) '
                     'and <path>.xml (GEM)')
args = ap.parse_args()

# Weighted loss replaces my_mse; AMN_QP looks it up at compile time
W = BM.tf.constant([1.0, args.lam, args.lam, args.lam])
def weighted_mse(y_true, y_pred):
    return BM.tf.reduce_mean(W * BM.tf.square(y_true[:, :4] - y_pred[:, :4]), axis=-1)
BM.my_mse = weighted_mse

# Data, exactly as learning_curve_unified.py
param = Neural_Model(model_type='AMN_QP', scaler=True, objective=True,
                     n_hidden=1, hidden_dim=1000, output_dim=1,
                     activation='relu', scoring_function=r2_score,
                     epochs=args.epochs, xfold=1, niter=0, batch_size=10,
                     trainingfile=args.trainingfile,
                     verbose=False)
X, Y = model_input(param, verbose=False)
param.X, param.Y = X, Y
y = Y[:, 0]
S, Pin = param.S, param.Pin
n_rxn = S.shape[1]

# Folds and nested subsamples, same seeding as learning_curve_unified.py
folds = []
for k, (tr, te) in enumerate(KFold(args.folds, shuffle=True,
                                   random_state=args.seed).split(X)):
    sub = {}
    for s in args.sizes:
        n = max(1, int(len(tr) * s / 100.0))
        sub[s] = []
        for d in range(args.draws):
            np.random.seed(args.seed + k + d * 100)
            sub[s].append(np.sort(np.random.permutation(len(tr))[:n]))
    folds.append((tr, te, sub))

def constraint_stats(V, Xin):
    """Constraint satisfaction of predicted flux vectors V (n_samples, n_rxn)."""
    SV = V @ S.T
    viol = np.maximum(V @ Pin.T - Xin, 0)
    return {
        'V_norm': np.linalg.norm(V, axis=1).mean(),
        'SV_norm': np.linalg.norm(SV, axis=1).mean(),
        'SV_rel': (np.linalg.norm(SV, axis=1)
                   / np.maximum(np.linalg.norm(V, axis=1), 1e-12)).mean(),
        'SV_maxabs': np.abs(SV).max(),
        'Pin_viol': np.linalg.norm(viol, axis=1).mean(),
    }

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
csv = out / f'ablation_{args.tag}.csv'
print(f'lam={args.lam}  tag={args.tag}  -> {csv}', flush=True)

for s in args.sizes:
    for d in range(args.draws):
        t0 = time.time()
        y_pred = np.zeros(len(y))
        V_test = np.zeros((len(y), n_rxn))
        for tr, te, sub in folds:
            idx = tr[sub[s][d]]
            _, _, ytest, *_ = train_model(param, X[idx], Y[idx], X[te], Y[te],
                                          failure=0, verbose=False)
            y_pred[te] = ytest[:, 0]
            V_test[te] = ytest[:, -n_rxn:]
        row = {'tag': args.tag, 'lam': args.lam, 'size_percent': s, 'draw': d,
               'collated_r2': r2_score(y, y_pred),
               **constraint_stats(V_test, X),
               'time_s': time.time() - t0}
        pd.DataFrame([row]).to_csv(csv, mode='a', header=not csv.exists(),
                                   index=False)
        print(f"size {s:3d}% draw {d}: R2={row['collated_r2']:.4f} "
              f"SV_rel={row['SV_rel']:.3g} Pin_viol={row['Pin_viol']:.3g} "
              f"({row['time_s']/60:.1f} min)", flush=True)
