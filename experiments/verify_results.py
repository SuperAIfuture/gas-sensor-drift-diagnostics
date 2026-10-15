"""Check file hashes, saved prediction metrics, row identities, and selections."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import common

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT/'SHA256.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    X, y, concentration, batch = common.load_dataset_from_zip('uci270', ROOT/'data/raw/uci270.zip')
    X0, y0, _, batch0 = common.load_dataset_from_zip('uci224', ROOT/'data/raw/uci224.zip')
    assert X.shape == (13910, 128)
    assert np.array_equal(X, X0) and np.array_equal(y, y0) and np.array_equal(batch, batch0)
    data = pd.read_csv(ROOT/'data/independent/extracted_features.csv', sep=' ')
    date = pd.to_datetime(data['Date[yyyy-mm-dd]'], format='%y-%m-%d')
    days = sorted(date.unique())
    ib = np.array([days.index(day)+1 for day in date])
    iy = pd.Categorical(data.Class, categories=sorted(data.Class.unique())).codes+1
    assert len(data) == 624 and len(days) == 36 and (ib <= 24).sum() == 431
    target_rows = validation_rows = 0
    differences = []

    def compare(truth, prediction, row, labels):
        values = {'macro_f1': f1_score(truth, prediction, labels=labels, average='macro', zero_division=0),
                  'accuracy': accuracy_score(truth, prediction)}
        if 'balanced_accuracy' in row:
            values['balanced_accuracy'] = balanced_accuracy_score(truth, prediction)
        if 'error_rate' in row:
            values['error_rate'] = 1-values['accuracy']
        if 'fixed6_f1' in row:
            values['fixed6_f1'] = f1_score(truth, prediction, labels=range(1, 7), average='macro', zero_division=0)
        for name, value in values.items():
            differences.append(abs(value-row[name]))

    for folder in ['uci_baselines', 'representation_controls', 'dg', 'independent']:
        base = ROOT/'results'/folder
        files = [('scores.csv', 'predictions.npz')]
        if folder == 'uci_baselines':
            files.append(('conventional_scores.csv', 'predictions.npz'))
        if folder == 'representation_controls':
            files = [('representation_scores.csv', 'representation_predictions.npz'),
                     ('augmentation_scores.csv', 'augmentation_predictions.npz')]
        for table, arrays in files:
            p = np.load(base/arrays)
            truth, domains = (iy, ib) if folder == 'independent' else (y, batch)
            target = domains > (24 if folder == 'independent' else 6)
            assert np.array_equal(p['y'], truth[target])
            assert np.array_equal(p['batch'], domains[target])
            if 'row_index' in p:
                assert np.array_equal(p['row_index'], np.where(target)[0])
            for _, row in pd.read_csv(base/table).iterrows():
                if table == 'representation_scores.csv':
                    key = f'{row.regime}__{row.method}'
                elif table == 'augmentation_scores.csv':
                    key = f'{row.method}__{int(row.seed)}'
                else:
                    key = f'{row.regime}__{row.method}__{int(row.seed)}'
                if folder == 'independent':
                    mask = np.ones(len(p['y']), bool) if row.scope == 'pooled' else (p['batch'] <= 30 if row.scope == 'early' else p['batch'] > 30)
                elif 'batch' in row:
                    mask = np.ones(len(p['y']), bool) if row.batch == 0 else p['batch'] == row.batch
                else:
                    mask = np.ones(len(p['y']), bool) if row.scope == 'pooled' else p['batch'] == int(row.scope.removeprefix('batch'))
                compare(p['y'][mask], p[key][mask], row, range(1, 6 if folder == 'independent' else 7))
                target_rows += 1

    selections = 0
    for folder in ['representation_controls', 'dg', 'independent']:
        base = ROOT/'results'/folder
        prefix = 'representation_' if folder == 'representation_controls' else ''
        validation = pd.read_csv(base/(prefix+'validation.csv'))
        p = np.load(base/(prefix+'validation_predictions.npz'))
        for _, row in validation.iterrows():
            if folder == 'independent':
                mask = (ib > row.fold_end-4) & (ib <= row.fold_end)
                key = f'{row.method}__{int(row.fold_end-4)}__{int(row.grid)}__{int(row.seed)}'
                truth, labels = iy[mask], range(1, 6)
            else:
                truth = y[batch == row.fold]
                labels = np.unique(truth)
                key = f'{row.method}__fold{int(row.fold)}__grid{int(row.grid)}'
                if folder == 'dg':
                    key += f'__seed{int(row.seed)}'
            compare(truth, p[key], row, labels)
            validation_rows += 1
        for _, row in pd.read_csv(base/(prefix+'choices.csv')).iterrows():
            if folder != 'independent' and row.regime == 'fixed':
                assert row.grid == 0
                continue
            candidates = validation[validation.method == row.method]
            if folder != 'independent' and row.regime == 'batch6':
                candidates = candidates[candidates.fold == 6]
            metric = 'fixed6_f1' if folder != 'independent' and row.regime == 'rolling6' else 'macro_f1'
            expected = candidates.groupby('grid')[metric].mean().sort_values(ascending=False, kind='stable').index[0]
            assert row.grid == expected, (folder, row.method)
            selections += 1
    base = ROOT/'results/uci_baselines'
    for table, choice in [('source_validation.csv', 'selected_parameters.csv'),
                          ('conventional_validation.csv', 'conventional_parameters.csv')]:
        val = pd.read_csv(base/table)
        for _, row in pd.read_csv(base/choice).iterrows():
            expected = val[val.method == row.method].groupby('grid').macro_f1.mean().sort_values(ascending=False, kind='stable').index[0]
            assert row.grid == expected
            selections += 1
    a = np.load(ROOT/'results/representation_controls/augmentation_predictions.npz')
    b = np.load(base/'predictions.npz')
    for seed in range(1, 9):
        assert np.array_equal(a[f'default__{seed}'], b[f'matched__RING-MIX__{seed}'])
    penalty = pd.read_csv(ROOT/'results/representation_controls/augmentation_penalty_counts.csv')
    assert np.allclose(penalty[penalty.method.isin(['mass','equal_mix','equal_conc'])].penalty_mass, 2966.5)
    counts = penalty[penalty.method.isin(['equal_mix','equal_conc'])].pivot(index=['seed','class_id'], columns='method', values='n_synthetic')
    assert (counts.equal_mix == counts.equal_conc).all()
    assert target_rows == 1284 and validation_rows == 512
    assert max(differences) < 1e-12, max(differences)
    print(f'VERIFIED: {len(manifest)} file hashes; {target_rows} target metric rows; {validation_rows} validation rows; {selections} selections; row identities and interpolation controls agree.')

if __name__ == '__main__':
    main()
