#!/usr/bin/env python
"""
AMN with exact mass balance (null-space projection) vs MLP, AMN_QP and FBA,
on random CV, leave-nutrient-out, simple-to-complex and low-data splits.

AMN-NS: X -> Dense(1000) -> Dropout -> V0                          (learned)
             --encoder linear : no activation, V0 is affine in X (default)
             --encoder relu   : ReLU hidden layer
             --param null : V0 = N z, z = Dense(null dim)  (V0 already balanced, default)
             --param flux : V0 = Dense(n_rxn)
        -> K Dykstra steps alternating two fixed projections (no weights)
              NN^T   : x -> N N^T x          (SV = 0 exactly)
              clip   : clip(x, lb, ub(X))   (V >= 0, ATPM >= 6.86,
                                             variable uptake <= 10 * presence,
                                             O2 <= 20, other minimal <= CSV row 3,
                                             uptake = 0 if not in medium)
  (with O2 <= 6.5, as in the dataset header, 165/280 measured growth rates
   exceed the FBA maximum; with the defaults above none do)
        -> V = NN^T x_K, growth = Pout V (biomass flux)
        loss = MSE(growth) + beta * ||clip(V) - V||^2  (residual box violation)
The flux space is the GEM (reversible reactions split into two irreversible
ones) restricted to reactions that are not blocked when all medium compounds
are available (example dataset: iML1515, 51 medium compounds).

Splits:
  random   : 5-fold KFold (shuffled with --seed; seed 1 = learning_curve_unified.py)
  nutrient : for each held-out nutrient family (nutrients with identical
             presence form one family; example dataset: 28 nutrients, 14 families), test = media containing it,
             train = media without it (never seen in training)
  complex  : for each t in --max-train-nutr, train = media with <= t variable
             nutrients, test = richer media (extrapolation to complex media)
  lowdata  : 5-fold KFold as random, but only --n-train media drawn from each
             training fold (learning curve; folds named n<size>_fold<k>);
             AMN-NS epochs are scaled so every size gets the same number of
             gradient steps as a full training fold

--tune selects hyperparameters inside each training fold (nested): 20 % of the
training media are held out for validation, every grid configuration is fitted
on the other 80 % and scored by validation MSE, and the best one is refitted
on the whole training fold. Grids: --betas x --uptakes x --epochs-grid (ns),
--alphas x --hiddens (mlp).

--k-eval re-runs the projection of the trained network with other numbers of
Dykstra steps at prediction time (columns y_pred_K<k>, box_viol_K<k>).

Usage:
    python amn_nullspace.py --model ns --split random --tag ns_random --plateau --epochs 100
    python amn_nullspace.py --model ns --split complex --tag ns_complex --plateau --epochs 100
    python amn_nullspace.py --model ns --split random --tag ns_tuned --tune --seed 2
    python amn_nullspace.py --model fba --split nutrient --tag fba_nutrient
    python amn_nullspace.py --model ns --split random --tag other --trainingfile <path/dataset>
"""

import argparse
import itertools
import time
import numpy as np
import pandas as pd
import scipy.linalg as la
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from Library.Build_Model import Neural_Model, model_input, train_model

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['ns', 'amn', 'mlp', 'fba'], required=True)
ap.add_argument('--split', choices=['random', 'nutrient', 'complex', 'lowdata'], required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--holdout', nargs='+', default=None,
                help='variable-medium reaction ids to hold out (default: all 28)')
ap.add_argument('--max-train-nutr', type=int, nargs='+', default=[10, 12, 14],
                help='complex split: largest number of nutrients in a training medium')
ap.add_argument('--n-train', type=int, nargs='+', default=[20, 40, 80],
                help='lowdata split: training-set sizes')
ap.add_argument('--epochs', type=int, default=1000)
ap.add_argument('--param', choices=['flux', 'null'], default='null',
                help='network output: fluxes, or null-space coordinates z (V0 = N z)')
ap.add_argument('--encoder', choices=['relu', 'linear'], default='linear',
                help='activation of the hidden layer (linear: V0 is linear in the medium)')
ap.add_argument('--K', type=int, default=20, help='Dykstra iterations')
ap.add_argument('--k-eval', type=int, nargs='+', default=[],
                help='also predict with these numbers of Dykstra steps')
ap.add_argument('--beta', type=float, default=0.1,
                help='loss weight of the residual box violation')
ap.add_argument('--uptake', type=float, default=10.0,
                help='uptake bound of the variable nutrients when present')
ap.add_argument('--o2', type=float, default=20.0, help='O2 uptake bound')
ap.add_argument('--l2', type=float, default=0.0,
                help='L2 penalty on the flux output layer (AMN-light: 0.01)')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--plateau', action='store_true',
                help='halve the learning rate when the training loss stalls')
ap.add_argument('--alpha', type=float, default=1e-4, help='MLP L2 penalty')
ap.add_argument('--hidden', type=int, default=1000, help='MLP hidden units')
ap.add_argument('--tune', action='store_true', help='nested hyperparameter selection')
ap.add_argument('--betas', type=float, nargs='+', default=[0.03, 0.1, 0.3])
ap.add_argument('--uptakes', type=float, nargs='+', default=[10.0, 20.0])
ap.add_argument('--alphas', type=float, nargs='+', default=[1e-4, 1e-2, 1.0])
ap.add_argument('--hiddens', type=int, nargs='+', default=[256, 1000])
ap.add_argument('--epochs-grid', type=int, nargs='+', default=None,
                help='with --tune: AMN-NS epochs to select from (default: --epochs)')
ap.add_argument('--only', nargs='+', default=None,
                help='run only these fold names (e.g. fold0)')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='./results/nullspace')
ap.add_argument('--trainingfile', default='./Dataset_input/Ecoli/iML1515EXP',
                help='dataset path without extension: <path>.npz (training set) '
                     'and <path>.xml (GEM)')
ap.add_argument('--uptake-prefix', default='EX_', help='id prefix of uptake reactions')
ap.add_argument('--uptake-suffix', default='_i', help='id suffix of uptake reactions')
ap.add_argument('--o2-id', default='EX_o2_e_i', help='O2 uptake reaction id')
args = ap.parse_args()

FLUX_MAX = 1000.0  # cap on unbounded fluxes
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
data_tag = Path(args.trainingfile).name  # caches are per dataset

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
ids = [r.id for r in param.cobramodel.reactions]
var_ids = [ids[j] for j in np.argmax(param.Pin, axis=1)]  # one per X column


def bounded_cobra_model(uptake, o2):
    """GEM with FBA bounds; uptakes of compounds not in the medium are 0."""
    m = param.cobramodel.copy()
    for r in m.reactions:
        if not (r.id.startswith(args.uptake_prefix) and r.id.endswith(args.uptake_suffix)):
            continue
        if r.id == args.o2_id:
            r.upper_bound = o2
        elif r.id in var_ids:
            r.upper_bound = uptake
        else:
            # minimal medium: uptake bound from the dataset (CSV row 3)
            r.upper_bound = float(param.medium[r.id][1]) if r.id in param.medium else 0.0
    return m


def reduced_model(uptake, o2):
    """Unblocked reactions, S, null-space basis N, bounds; cached on disk."""
    cache = out / f'reduced_model_{data_tag}_u{uptake:g}_o{o2:g}.npz'
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    import cobra
    m = bounded_cobra_model(uptake, o2)
    blocked = set(cobra.flux_analysis.find_blocked_reactions(
        m, open_exchanges=False, processes=1))
    keep = np.array([j for j, r in enumerate(ids) if r not in blocked])
    S = param.S[:, keep]
    S = S[np.abs(S).sum(axis=1) > 0]
    rxns = [m.reactions[j] for j in keep]
    lb = np.array([r.lower_bound for r in rxns])
    ub = np.minimum([r.upper_bound for r in rxns], FLUX_MAX)
    kept = [ids[j] for j in keep]
    var_pos = np.array([kept.index(r) for r in var_ids])
    ub[var_pos] = 0.0  # set per sample from X
    d = dict(keep=keep, N=la.null_space(S), lb=lb, ub=ub, var_pos=var_pos,
             growth_pos=kept.index(param.objective[0]))
    np.savez(cache, **d)
    return d


def fba_growth(uptake, o2):
    """Max growth per medium under the same bounds (no learning)."""
    cache = out / f'fba_growth_{data_tag}_u{uptake:g}_o{o2:g}.npy'
    if cache.exists():
        return np.load(cache)
    m = bounded_cobra_model(uptake, o2)
    gr = np.zeros(len(X))
    for i in range(len(X)):
        with m:
            for k, rid in enumerate(var_ids):
                m.reactions.get_by_id(rid).upper_bound = uptake * X[i, k]
            gr[i] = m.slim_optimize(error_value=0.0)
    np.save(cache, gr)
    return gr


def splits():
    if args.split == 'random':
        kf = KFold(5, shuffle=True, random_state=args.seed)
        return [(f'fold{k}', tr, te) for k, (tr, te) in enumerate(kf.split(X))]
    if args.split == 'complex':
        n = X.sum(1)
        return [(f'le{t}', np.where(n <= t)[0], np.where(n > t)[0])
                for t in args.max_train_nutr]
    if args.split == 'lowdata':
        kf = KFold(5, shuffle=True, random_state=args.seed)
        rng = np.random.default_rng(args.seed)
        folds = list(kf.split(X))
        return [(f'n{m}_fold{k}', np.sort(rng.choice(tr, m, replace=False)), te)
                for m in args.n_train for k, (tr, te) in enumerate(folds)]
    # nutrients are added in families (identical presence columns): 14 groups
    groups = {}
    for j, r in enumerate(var_ids):
        groups.setdefault(X[:, j].tobytes(), []).append(j)
    held = [g for g in groups.values()
            if args.holdout is None or {var_ids[j] for j in g} & set(args.holdout)]
    return [('+'.join(var_ids[j][3:-4] for j in g),
             np.where(X[:, g[0]] == 0)[0], np.where(X[:, g[0]] == 1)[0])
            for g in held]


def upper_bounds(red, uptake, Xs):
    """Per-sample upper bounds: fixed bounds + uptake * presence."""
    E = np.zeros((X.shape[1], len(red['lb'])), np.float32)
    E[np.arange(X.shape[1]), red['var_pos']] = uptake
    return red['ub'][None, :] + Xs @ E, E


def dykstra(V0, red, ub, K):
    """NumPy twin of the projection layer, for evaluation with any K."""
    N, lb = red['N'], red['lb']
    x, q = V0, np.zeros_like(V0)
    for _ in range(K):
        p = (x @ N) @ N.T
        x = np.clip(p + q, lb, ub)
        q = p + q - x
    V = (x @ N) @ N.T
    return V[:, int(red['growth_pos'])], np.linalg.norm(np.clip(V, lb, ub) - V, axis=1)


def build_ns(red, cfg):
    """Return (trainable model, sub-model giving V0)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    N = tf.constant(red['N'], tf.float32)
    lb = tf.constant(red['lb'], tf.float32)
    ub0 = tf.constant(red['ub'], tf.float32)
    E = tf.constant(upper_bounds(red, cfg['uptake'], X[:1])[1])
    g = int(red['growth_pos'])

    def project(x):
        return tf.matmul(tf.matmul(x, N), N, transpose_b=True)

    inputs = Input(shape=(X.shape[1],))
    h = Dense(1000, activation=None if cfg['encoder'] == 'linear' else 'relu',
              kernel_initializer='random_normal')(inputs)
    h = Dropout(0.25)(h)
    if cfg['param'] == 'null':
        z = Dense(red['N'].shape[1], kernel_initializer='random_normal',
                  kernel_regularizer=l2(args.l2))(h)
        V0 = tf.matmul(z, N, transpose_b=True)
    else:
        V0 = Dense(len(red['lb']), kernel_initializer='random_normal',
                   kernel_regularizer=l2(args.l2))(h)
    ub = ub0 + tf.matmul(inputs, E)
    x, q = V0, tf.zeros_like(V0)
    for _ in range(args.K):  # Dykstra: subspace step needs no correction
        p = project(x)
        x = tf.clip_by_value(p + q, lb, ub)
        q = p + q - x
    V = project(x)
    viol = tf.norm(tf.clip_by_value(V, lb, ub) - V, axis=1, keepdims=True)
    model = keras.Model(inputs, [V[:, g:g + 1], viol])
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss=['mse', 'mse'], loss_weights=[1.0, cfg['beta']])
    return model, keras.Model(inputs, V0)


def fit_ns(tr, te, cfg, seed):
    """Train AMN-NS on tr; return dict of test predictions."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    red = reduced_model(cfg['uptake'], args.o2)
    model, v0_model = build_ns(red, cfg)
    cb = [tf.keras.callbacks.ReduceLROnPlateau('loss', factor=0.5, patience=10,
                                               min_lr=1e-5)] if args.plateau else []
    epochs = cfg['epochs']
    if args.split == 'lowdata':  # same number of gradient steps as a full 5-fold training set
        epochs = int(round(epochs * 0.8 * len(X) / len(tr)))
    model.fit(X[tr], [y[tr], np.zeros(len(tr))], epochs=epochs,
              batch_size=10, callbacks=cb, verbose=0)
    g, viol = model.predict(X[te], verbose=0)
    res = {'y_pred': g[:, 0], 'box_viol': viol[:, 0]}
    if args.k_eval:
        V0 = v0_model.predict(X[te], verbose=0)
        ub = upper_bounds(red, cfg['uptake'], X[te])[0]
        for K in args.k_eval:
            res[f'y_pred_K{K}'], res[f'box_viol_K{K}'] = dykstra(V0, red, ub, K)
    return res


def fit_mlp(tr, te, cfg, seed):
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor((cfg['hidden'],), activation='relu', solver='adam',
                       alpha=cfg['alpha'], learning_rate_init=0.001,
                       batch_size=10, max_iter=args.epochs, random_state=seed)
    return {'y_pred': mlp.fit(X[tr], y[tr]).predict(X[te])}


def configs():
    """Hyperparameter grid (one configuration when not tuning)."""
    if args.model == 'ns':
        betas, ups = (args.betas, args.uptakes) if args.tune else ([args.beta], [args.uptake])
        eps = args.epochs_grid if args.tune and args.epochs_grid else [args.epochs]
        return [dict(param=args.param, encoder=args.encoder, beta=b, uptake=u, epochs=e)
                for b, u, e in itertools.product(betas, ups, eps)]
    if args.model == 'mlp':
        alphas, hids = (args.alphas, args.hiddens) if args.tune else ([args.alpha], [args.hidden])
        return [dict(alpha=a, hidden=h) for a, h in itertools.product(alphas, hids)]
    return [{}]


def fit_predict(tr, te, k):
    """Return (test predictions dict, chosen configuration)."""
    seed = args.seed * 100 + k
    if args.model == 'fba':
        return {'y_pred': fba_growth(args.uptake, args.o2)[te]}, {}
    if args.model == 'amn':
        _, _, ytest, *_ = train_model(param, X[tr], Y[tr], X[te], Y[te],
                                      failure=0, verbose=False)
        return {'y_pred': ytest[:, 0]}, {}
    fit = fit_ns if args.model == 'ns' else fit_mlp
    grid = configs()
    best = grid[0]
    if len(grid) > 1:  # nested selection on an inner validation split
        itr, iva = train_test_split(tr, test_size=0.2, random_state=seed)
        mse = [mean_squared_error(y[iva], fit(itr, iva, c, seed)['y_pred']) for c in grid]
        best = grid[int(np.argmin(mse))]
    return fit(tr, te, best, seed), best


if args.model == 'ns':
    for u in (args.uptakes if args.tune else [args.uptake]):
        red = reduced_model(u, args.o2)
        print(f"reduced model u={u:g}: {len(red['lb'])} reactions, "
              f"null dim {red['N'].shape[1]}, K={args.K}", flush=True)

csv = out / f'pred_{args.tag}.csv'
done = set(pd.read_csv(csv)['fold']) if csv.exists() else set()
for k, (name, tr, te) in enumerate(splits()):
    if name in done or (args.only and name not in args.only):
        continue
    t0 = time.time()
    res, cfg = fit_predict(tr, te, k)
    pd.DataFrame({'tag': args.tag, 'model': args.model, 'split': args.split,
                  'seed': args.seed, 'fold': name, 'idx': te, 'y': y[te],
                  'box_viol': np.nan, **res, **cfg}).to_csv(
        csv, mode='a', index=False, header=not csv.exists())
    print(f'{name:16s} n_train={len(tr):3d} n_test={len(te):3d} '
          f"R2={r2_score(y[te], res['y_pred']):.4f} "
          f"viol={np.nanmean(res.get('box_viol', np.nan)):.3g} {cfg} "
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)

df = pd.read_csv(csv)
print(f'pooled R2 over all test predictions: {r2_score(df.y, df.y_pred):.4f}')
