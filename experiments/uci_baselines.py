"""Chronological UCI classification with matched and source-selected SVM settings."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/uci_baselines'
import common as old

SEEDS = list(range(1, 9))
GRID = [(3., .003), (3., 'scale'), (10., .003), (10., 'scale')]
METHODS = ['RAW', 'RING', 'RAW-MIX', 'RING-MIX', 'RING-CONC']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metric(y, p):
    return dict(macro_f1=float(f1_score(y, p, labels=np.arange(1, 7), average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, p)),
                accuracy=float(accuracy_score(y, p)), error_rate=float(1 - accuracy_score(y, p)))

def main():
    started = time.time()
    (OUT / 'analysis').mkdir(parents=True, exist_ok=True)

    X, y, conc, batch = old.load_dataset_from_zip('uci270', ROOT / 'data/raw/uci270.zip')
    X0, y0, _, b0 = old.load_dataset_from_zip('uci224', ROOT / 'data/raw/uci224.zip')
    same = bool(np.array_equal(X, X0) and np.array_equal(y, y0) and np.array_equal(batch, b0))
    assert same and X.shape == (13910, 128)
    count = pd.crosstab(batch, y).rename_axis('batch').reset_index()
    count.to_csv(OUT / 'analysis/batch_class_counts.csv', index=False)
    pd.DataFrame({'class_id': y, 'concentration_ppmv': conc}).groupby('class_id').agg(
        minimum_ppmv=('concentration_ppmv','min'),maximum_ppmv=('concentration_ppmv','max'),
        distinct_values=('concentration_ppmv','nunique')).reset_index().to_csv(OUT / 'analysis/concentration_ranges.csv', index=False)
    train = batch <= 6
    inner = batch <= 5
    val = batch == 6
    target = batch >= 7

    def features(kind, mask):
        if kind == 'RAW':
            return X
        values = X
        if kind == 'SCALED-RING':
            values = StandardScaler().fit(X[mask]).transform(X)
        means = values.reshape(-1, 16, 8).mean(axis=2)
        if kind == 'IDENTITY':
            smooth = means
        else:
            L = old.build_ring_laplacian(16)
            if kind == 'PERMUTED-RING':
                perm = np.random.default_rng(20261014).permutation(16)
                L = L[np.ix_(perm, perm)]
            smooth = means @ np.linalg.inv(np.eye(16) + .4 * L)
        return np.column_stack((X, smooth, means-smooth))

    feature_cache = {}
    def xy(method, fit_mask, seed):
        kind = method.replace('-CONC','').replace('-MIX','')
        key = (kind, int(fit_mask.sum()))
        if key not in feature_cache:
            feature_cache[key] = features(kind, fit_mask)
        F = feature_cache[key]
        a, z = old.build_training_matrix_with_mixup(
            F[fit_mask], y[fit_mask], batch[fit_mask], conc[fit_mask],
            use_mixup=method.endswith(('-MIX','-CONC')), use_concentration_mixup=method.endswith('-CONC'),
            seed=seed, per_pair=40, beta_alpha=.8)
        return a, z, F

    validation = []
    vp = {}
    for method in METHODS:
        for g, (C, gamma) in enumerate(GRID):
            for seed in (SEEDS if method.endswith(('-MIX','-CONC')) else [1]):
                a,z,F = xy(method, inner, seed)
                p = old.fit_predict_svm(a,z,F[val], C, gamma)
                vp[(method,g,seed)] = p
                validation.append(dict(method=method, grid=g, seed=seed, C=C, gamma=gamma,
                                       n_real=int(inner.sum()), n_synthetic=int(len(z)-inner.sum()), **metric(y[val],p)))
            print(f'VALIDATION {method} grid={g} complete', flush=True)
    vf = pd.DataFrame(validation)
    vf.to_csv(OUT / 'analysis/source_validation.csv', index=False)
    choice = vf.groupby(['method','grid']).macro_f1.mean().reset_index().sort_values(['method','macro_f1','grid'], ascending=[True,False,True]).drop_duplicates('method')
    chosen = dict(zip(choice.method, choice.grid.astype(int)))
    choice.to_csv(OUT / 'analysis/selected_parameters.csv', index=False)

    scores, predictions = [], {}
    fits = {}
    def evaluate(regime, method, seed, grid):
        # Reuse deterministic fits and repeated parameter combinations exactly.
        key = (method, seed if method.endswith(('-MIX','-CONC')) else 1, grid)
        if key not in fits:
            a,z,F = xy(method, train, seed)
            C,gamma = GRID[grid]
            fits[key] = (old.fit_predict_svm(a,z,F[target],C,gamma), len(z)-int(train.sum()))
        p, nsynthetic = fits[key]
        predictions[f'{regime}__{method}__{seed}'] = p
        for scope, mask in [('pooled',np.ones(int(target.sum()),dtype=bool))] + [(f'batch{b}', batch[target]==b) for b in range(7,11)]:
            scores.append(dict(regime=regime, method=method, seed=seed, grid=grid, scope=scope,
                               n_synthetic=nsynthetic, n_test=int(mask.sum()), **metric(y[target][mask],p[mask])))
    for regime in ['matched','source_selected']:
        for method in METHODS:
            grid = 0 if regime=='matched' else chosen[method]
            for seed in SEEDS:
                evaluate(regime,method,seed,grid)
            print(f'TARGET {regime} {method} complete', flush=True)
    for method in ['IDENTITY-MIX','PERMUTED-RING-MIX','SCALED-RING-MIX']:
        for seed in SEEDS:
            evaluate('control',method,seed,0)
        print(f'CONTROL {method} complete',flush=True)

    selection = []
    for seed in SEEDS:
        f_mix = metric(y[val],vp[('RING-MIX',0,seed)])['macro_f1']
        f_conc = metric(y[val],vp[('RING-CONC',0,seed)])['macro_f1']
        take_conc = f_conc > f_mix
        selection.append(dict(seed=seed, validation_mix=f_mix, validation_conc=f_conc, delta=f_conc-f_mix, concentration_selected=take_conc))
        selected_method = 'RING-CONC' if take_conc else 'RING-MIX'
        p = predictions[f'matched__{selected_method}__{seed}']
        predictions[f'matched__SOURCE-CHOICE__{seed}'] = p
        for scope, mask in [('pooled',np.ones(int(target.sum()),dtype=bool))] + [(f'batch{b}',batch[target]==b) for b in range(7,11)]:
            scores.append(dict(regime='matched',method='SOURCE-CHOICE',seed=seed,grid=0,scope=scope,
                               n_synthetic=fits[(selected_method,seed,0)][1], n_test=int(mask.sum()), **metric(y[target][mask],p[mask])))
    pd.DataFrame(selection).to_csv(OUT / 'analysis/concentration_selection.csv',index=False)
    sf = pd.DataFrame(scores)
    sf.to_csv(OUT / 'analysis/scores.csv',index=False)
    summary = sf.groupby(['regime','method','scope'])[['macro_f1','balanced_accuracy','accuracy','error_rate']].agg(['mean','std','min','max'])
    summary.to_csv(OUT / 'analysis/summary.csv')
    np.savez_compressed(OUT / 'analysis/predictions.npz', row_index=np.where(target)[0], y=y[target], batch=batch[target], **predictions)
    meta = dict(dataset_views_identical=same, n_unique=len(y), n_source=int(train.sum()), n_target=int(target.sum()),
                seeds=SEEDS, grid=[dict(C=c,gamma=g) for c,g in GRID], chosen_grid={k:int(v) for k,v in chosen.items()},
                source_validation='batches 1-5 fit; batch 6 validation; select mean Macro-F1 across seeds; ties choose grid order',
                matched_grid=0, ring_alpha=.4, beta_alpha=.8, per_pair=40,
                concentration_bins='pooled source 25/50/75 percentiles, not class specific', permutation_seed=20261014,
                retrospective=True, target_used_for_fit_or_selection=False,
                versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                input_hashes={n:digest(ROOT/'data/raw'/n) for n in ['uci224.zip','uci270.zip']},
                implementation_sha256=digest(Path(__file__)), seconds=time.time()-started)
    (OUT/'analysis/run_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(sf[sf.scope=='pooled'].groupby(['regime','method']).macro_f1.agg(['mean','std']).to_string(),flush=True)
    print('UCI BASELINES PASS: identical views counted once; source-only selection; predictions saved.',flush=True)

if __name__ == '__main__':
    main()
