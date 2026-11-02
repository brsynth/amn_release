#!/usr/bin/env python
"""
UNIFIED Learning Curve: AMN vs Baselines (MLR, MLP)
Ensures 100% identical data, folds, and subsampling across all models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import time
import json

# Import AMN dependencies
from Library.Import import *
from Library.Build_Model import Neural_Model, model_input, train_model

# Configuration
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--trainingfile', default='./Dataset_input/Ecoli/iML1515EXP',
                help='dataset path without extension: <path>.npz (training set) '
                     'and <path>.xml (GEM)')
trainingfile = ap.parse_args().trainingfile

sizes_percent = [10, 20, 50, 100]
n_folds = 5
n_draws = 5
seed = 1

# ============================================================================
# STEP 1: Load data EXACTLY as AMN does
# ============================================================================
print("="*80)
print("LOADING DATA (AMN method)")
print("="*80)

model_params_amn = {
    'model_type': 'AMN_QP',
    'scaler': True,
    'objective': True,
    'n_hidden': 1,
    'hidden_dim': 1000,
    'output_dim': 1,
    'activation': 'relu',
    'scoring_function': r2_score,
    'epochs': 1000,
    'xfold': 1,
    'niter': 0,
    'batch_size': 10
}

param_amn = Neural_Model(**model_params_amn, trainingfile=trainingfile, verbose=False)
X_amn, Y_amn = model_input(param_amn, verbose=False)
param_amn.X, param_amn.Y = X_amn, Y_amn

# Extract features and growth rates
X = X_amn  # (280, 28) - medium features (SCALED by AMN)
y = Y_amn[:, 0]  # (280,) - growth rates only

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Growth rate range: [{y.min():.3f}, {y.max():.3f}]")
print("-" * 80)

# ============================================================================
# STEP 2: Define models with EXACT same architecture where possible
# ============================================================================
print("\nMODEL CONFIGURATIONS:")
print("-" * 80)

models = {
    'MLR': {
        'name': 'Multiple Linear Regression',
        'type': 'baseline',
        'config': {
            'class': LinearRegression,
            'params': {}
        }
    },
    'MLP': {
        'name': 'Multi-Layer Perceptron (sklearn)',
        'type': 'baseline',
        'config': {
            'class': MLPRegressor,
            'params': {
                'hidden_layer_sizes': (1000,),  # 1 layer × 1000 neurons
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate_init': 0.001,  # Match AMN
                'batch_size': 10,  # Match AMN (not 'auto')
                'max_iter': 1000,  # Match AMN epochs
                'random_state': seed,
                'early_stopping': False,
                'verbose': False,
                # NOTE: sklearn MLP has NO dropout (AMN has 0.25)
                # This makes MLP weaker, but can't be helped with sklearn
            }
        }
    },
    'AMN': {
        'name': 'Augmented Metabolic Network',
        'type': 'amn',
        'config': param_amn
    }
}

for name, info in models.items():
    print(f"{name}: {info['name']}")
    if name == 'MLP':
        print(f"  WARNING: sklearn MLP has NO dropout (AMN uses 0.25)")
        print(f"  This will make MLP perform worse than it could")

print("-" * 80)

# ============================================================================
# STEP 3: Define IDENTICAL fold/subsample generation
# ============================================================================

def generate_folds_and_subsamples(X, y, n_folds, sizes_percent, n_draws, seed):
    """
    Generate ALL folds and subsamples upfront to ensure 100% consistency.

    Returns:
        fold_configs: List of (train_idx, test_idx, subsamples_dict)
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_configs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        # Generate subsamples for this fold
        subsamples = {}
        for size_pct in sizes_percent:
            n_samples = max(1, int(len(train_idx) * size_pct / 100.0))
            subsamples[size_pct] = []

            for draw_idx in range(n_draws):
                # Use consistent seeding
                np.random.seed(seed + fold_idx + draw_idx * 100)
                indices = np.arange(len(train_idx))
                shuffled = np.random.permutation(indices)
                subsample_idx = np.sort(shuffled[:n_samples])
                subsamples[size_pct].append(subsample_idx)

        fold_configs.append((train_idx, test_idx, subsamples))

    return fold_configs

print("\nGENERATING FOLDS AND SUBSAMPLES...")
fold_configs = generate_folds_and_subsamples(X, y, n_folds, sizes_percent, n_draws, seed)
print(f"✓ Generated {len(fold_configs)} folds with {len(sizes_percent)} sizes × {n_draws} draws each")
print("-" * 80)

# ============================================================================
# STEP 4: Run experiments with COLLATED R²
# ============================================================================

def train_baseline_model(model_class, model_params, X_train, y_train, X_test, y_test):
    """Train sklearn baseline model (MLR or MLP)."""
    # Baselines use X as-is (already scaled by AMN)
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_amn_model(param_amn, X_train, Y_train_full, X_test, Y_test_full):
    """Train AMN model."""
    Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param_amn, X_train, Y_train_full, X_test, Y_test_full,
                   failure=0, verbose=False)
    # Extract growth rate predictions (first column)
    return ytest[:, 0]

# Run experiment
print("\nRUNNING EXPERIMENT WITH COLLATED R²...")
print("="*80)

all_results = []
output_path = Path('./results/learning_curve')
output_path.mkdir(parents=True, exist_ok=True)

for model_name, model_info in models.items():
    print(f"\nMODEL: {model_name}")
    print("-" * 80)

    for size_pct in sizes_percent:
        print(f"  Size: {size_pct}%", end='', flush=True)

        for draw_idx in range(n_draws):
            # Collect predictions from all folds
            y_true_all = np.zeros(len(y))
            y_pred_all = np.zeros(len(y))

            total_time = 0

            for fold_idx, (train_idx, test_idx, subsamples) in enumerate(fold_configs):
                # Get subsample indices for this size/draw
                subsample_idx = subsamples[size_pct][draw_idx]

                # Get data for this fold
                X_train_full = X[train_idx]
                X_train = X_train_full[subsample_idx]
                X_test = X[test_idx]

                y_train_full = y[train_idx]
                y_train = y_train_full[subsample_idx]
                y_test = y[test_idx]

                # Train model
                start_time = time.time()

                if model_info['type'] == 'baseline':
                    # Train baseline (MLR or MLP)
                    y_pred = train_baseline_model(
                        model_info['config']['class'],
                        model_info['config']['params'],
                        X_train, y_train, X_test, y_test
                    )
                else:  # AMN
                    # Prepare Y for AMN (with constraint columns)
                    Y_train_full = Y_amn[train_idx]
                    Y_train = Y_train_full[subsample_idx]
                    Y_test = Y_amn[test_idx]

                    y_pred = train_amn_model(
                        param_amn, X_train, Y_train, X_test, Y_test
                    )

                total_time += time.time() - start_time

                # Collect predictions
                y_true_all[test_idx] = y_test
                y_pred_all[test_idx] = y_pred

            # Compute collated R²
            r2_collated = r2_score(y_true_all, y_pred_all)

            # Store results
            all_results.append({
                'model': model_name,
                'size_percent': size_pct,
                'draw': draw_idx,
                'collated_r2': r2_collated,
                'n_train_samples': len(subsample_idx),
                'total_time_seconds': total_time
            })

            if draw_idx == 0:
                print(f" → {r2_collated:.4f}", end='', flush=True)

        # Print mean for this size
        size_results = [r for r in all_results if r['model']==model_name and r['size_percent']==size_pct]
        r2_values = [r['collated_r2'] for r in size_results]
        print(f" (mean={np.mean(r2_values):.4f}±{np.std(r2_values):.4f})")

# Save results
df = pd.DataFrame(all_results)
final_csv = output_path / 'learning_curve_unified.csv'
df.to_csv(final_csv, index=False)

print(f"\n{'='*80}")
print("EXPERIMENT COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {final_csv}")

# Print summary
print(f"\nFINAL SUMMARY (COLLATED R²):")
print("="*80)
for model_name in models.keys():
    print(f"\n{model_name}:")
    df_model = df[df['model'] == model_name]
    summary = df_model.groupby('size_percent')['collated_r2'].agg(['mean', 'std'])
    print(summary)

    r2_100 = summary.loc[100, 'mean']
    print(f"  → At 100%: {r2_100:.4f}")

print("\n" + "="*80)
print("NOTES:")
print("- All models use IDENTICAL data, folds, and subsamples")
print("- All models use data scaled the same way (via AMN's MaxScaler)")
print("- MLP lacks dropout (sklearn limitation) → may underperform")
print("- Collated R² = one R² computed on all 280 predictions from 5 folds")
print("="*80)
