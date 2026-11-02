#!/usr/bin/env python
"""
Summary bar chart: R² on test media of FBA, MLR, MLP and
AMN-NS for the low-data
learning curve (20, 40, 80 training media), the simple-to-complex split
(130 training media with <= 12 variable nutrients) and random 5-fold CV
(224 training media).

Reads the prediction files written by amn_nullspace.py and fba_lambda.py:
    results/extrap/pred_lowdata_{mlp,nslin}_s*.csv
    results/extrap/pred_complex_{mlp,nslin}_s*.csv   (training set: t = 12)
    results/extrap/pred_random_nslin_s*.csv
    results/optim/pred_A_mlp_s*.csv                  (MLP, random CV)
    results/fba_lambda/fba_growth_by_lambda.csv
MLR (multiple linear regression, least squares on the presence vector) is fitted
here, on the same splits as amn_nullspace.py (seeds 1-3).
R² is collated per seed (one R² over all test predictions of a run); bars
are the mean over seeds and error bars the sd. FBA has no training and no
seed: on random CV and the learning curve (whose test folds together cover
all 280 media) it is the published protocol (best lambda over all media),
on the complex split lambda is chosen on the 130 training media. The
complex split does not depend on the seed, so MLR has no sd
there.

Writes results/summary/r2_summary.csv and r2_summary.png.

Usage:
    python plot_r2_summary.py
"""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

R = Path('results')
OUT = R / 'summary'


def r2(y, p):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def seeds(pattern, fold=None):
    """Collated R² of every seed file matching pattern (optionally one fold prefix)."""
    out = []
    for f in sorted(glob.glob(str(R / pattern))):
        d = pd.read_csv(f)
        if fold:
            d = d[d.fold.str.startswith(fold)]
        out.append(r2(d.y, d.y_pred))
    return np.array(out)


# FBA growth per lambda (columns) and medium (rows)
G = pd.read_csv(R / 'fba_lambda' / 'fba_growth_by_lambda.csv')
y = G.pop('y').values
fba_all = max(r2(y, G[c].values) for c in G)
test = pd.read_csv(R / 'extrap' / 'pred_complex_mlp_s1.csv').query("fold == 'le12'").idx.values
train = np.setdiff1d(np.arange(len(y)), test)
lam = max(G, key=lambda c: r2(y[train], G[c].values[train]))
fba_le12 = r2(y[test], G[lam].values[test])

# MLR on the splits of amn_nullspace.py
D = np.load('Dataset_input/Ecoli/iML1515EXP.npz')
X, Yd = D['X'], D['Y'][:, 0]


def ols(n_train, seed):
    """Collated R² of OLS for one test (n_train as in `tests`) and one seed."""
    if n_train == 130:
        tr, te = X.sum(1) <= 12, X.sum(1) > 12
        return r2(Yd[te], LinearRegression().fit(X[tr], Yd[tr]).predict(X[te]))
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    if n_train < 224:  # same draws, in the same order, as amn_nullspace.py --split lowdata
        rng = np.random.default_rng(seed)
        sub = {(m, k): np.sort(rng.choice(tr, m, replace=False))
               for m in (20, 40, 80) for k, (tr, te) in enumerate(folds)}
        folds = [(sub[n_train, k], te) for k, (tr, te) in enumerate(folds)]
    p = np.zeros(len(Yd))
    for tr, te in folds:
        p[te] = LinearRegression().fit(X[tr], Yd[tr]).predict(X[te])
    return r2(Yd, p)


tests = [  # label, n training media, MLP pattern, AMN-NS pattern, fold prefix, FBA
    ('20 media\n(10 %)', 20, 'extrap/pred_lowdata_mlp_s*.csv', 'extrap/pred_lowdata_nslin_s*.csv', 'n20_', fba_all),
    ('40 media\n(20 %)', 40, 'extrap/pred_lowdata_mlp_s*.csv', 'extrap/pred_lowdata_nslin_s*.csv', 'n40_', fba_all),
    ('80 media\n(35 %)', 80, 'extrap/pred_lowdata_mlp_s*.csv', 'extrap/pred_lowdata_nslin_s*.csv', 'n80_', fba_all),
    ('130 media\n(≤ 12 nutr.)', 130, 'extrap/pred_complex_mlp_s*.csv', 'extrap/pred_complex_nslin_s*.csv', 'le12', fba_le12),
    ('224 media\n(100 %)', 224, 'optim/pred_A_mlp_s*.csv', 'extrap/pred_random_nslin_s*.csv', None, fba_all),
]
rows = []
for label, n, mlp, ns, fold, fba in tests:
    test_name = label.split('\n')[0]
    rows.append(dict(test=test_name, n_train=n, model='FBA', r2_mean=fba, r2_sd=np.nan, n_seeds=0))
    v = np.array([ols(n, s) for s in (1, 2, 3)]) if n != 130 else np.array([ols(n, 1)])
    rows.append(dict(test=test_name, n_train=n, model='MLR', r2_mean=v.mean(),
                     r2_sd=v.std(ddof=1) if len(v) > 1 else np.nan, n_seeds=len(v) if len(v) > 1 else 0))
    for model, pattern in (('MLP', mlp), ('AMN-NS', ns)):
        v = seeds(pattern, fold)
        rows.append(dict(test=test_name, n_train=n, model=model, r2_mean=v.mean(),
                         r2_sd=v.std(ddof=1), n_seeds=len(v)))
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.round(4).to_csv(OUT / 'r2_summary.csv', index=False)
print(df.round(3).to_string(index=False))

colors = {'FBA': '#F5A98A', 'MLR': '#C9B3EC', 'MLP': '#86B6EF', 'AMN-NS': '#7FD3B2'}
fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
x = np.arange(len(tests))
w = 0.2
for k, model in enumerate(colors):
    d = df[df.model == model]
    pos = x + (k - 1.5) * w
    ax.bar(pos, d.r2_mean.clip(lower=0), w - 0.03, color=colors[model], label=model, zorder=2)
    pos_ok = d.r2_mean.values > 0  # a negative R² is written at the baseline instead of drawn
    if d.r2_sd.notna().any():
        ax.errorbar(pos[pos_ok], d.r2_mean[pos_ok], yerr=d.r2_sd[pos_ok], fmt='none',
                    ecolor='#333333', elinewidth=0.8, capsize=2, zorder=3)
    for p, m in zip(pos[~pos_ok], d.r2_mean[~pos_ok]):
        ax.text(p, 0.01, f'{m:.2f}', ha='center', va='bottom', rotation=90, fontsize=6.5)
    if model == 'AMN-NS':
        for p, m, s in zip(pos, d.r2_mean, d.r2_sd):
            ax.text(p, m + s + 0.015, f'{m:.2f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x, [t[0] for t in tests], fontsize=7.8)
ax.set_xlabel('Training set')
ax.set_ylabel('R² on test media')
ax.set_ylim(0, 0.9)
ax.yaxis.grid(True, color='#e3e2de', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / 'r2_summary.png')
print(f'wrote {OUT}/r2_summary.csv and r2_summary.png')
