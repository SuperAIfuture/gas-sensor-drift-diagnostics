from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t, binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/statistics'
DATA = ROOT/'results/uci_baselines'
OUT.mkdir(parents=True,exist_ok=True)
d = pd.read_csv(DATA/'scores.csv')
pairs = [('matched', 'RAW-MIX', 'RAW'), ('matched', 'RING-MIX', 'RING'),
         ('source_selected', 'RAW-MIX', 'RAW'), ('source_selected', 'RING-MIX', 'RING')]
rows = []
for regime, augmented, reference in pairs:
    q = d[(d.regime == regime)&(d.scope == 'pooled')].pivot(index='seed', columns='method', values='macro_f1')
    delta = (q[augmented]-q[reference]).to_numpy()
    assert len(delta) == 8 and np.all(np.isfinite(delta))
    lo, hi = t.interval(.95, 7, loc=delta.mean(), scale=delta.std(ddof=1)/np.sqrt(8))
    ntie = int(np.count_nonzero(delta == 0))
    npos = int(np.count_nonzero(delta > 0))
    p = float(binomtest(npos, len(delta)-ntie, .5, alternative='two-sided').pvalue)
    inferential = regime == 'matched'
    rows.append(dict(comparison_id=f'{regime}:{augmented}-{reference}', regime=regime,
                     augmented=augmented, reference=reference, n_seeds=8,
                     mean_difference=float(delta.mean()), sample_sd=float(delta.std(ddof=1)),
                     ci_low=float(lo) if inferential else None, ci_high=float(hi) if inferential else None,
                     observed_min=float(delta.min()), observed_max=float(delta.max()),
                     positive_seeds=npos, ties=ntie,
                     p_raw=p if inferential else None, seed_differences=delta.tolist(),
                     inference_performed=inferential))
tested = [i for i,r in enumerate(rows) if r['inference_performed']]
order = sorted(tested, key=lambda i: rows[i]['p_raw'])
running = 0.
for position, index in enumerate(order):
    running = max(running, min(1., (len(tested)-position)*rows[index]['p_raw']))
    rows[index]['p_holm'] = running
for row in rows:
    row.setdefault('p_holm', None)
    row.update(metric='pooled target Macro-F1', test_name='two-sided exact sign test across augmentation randomizations' if row['inference_performed'] else 'not tested: parameter selection reused these seeds',
               ci_method='pointwise Student t interval for the conditional seed-mean difference, df=7' if row['inference_performed'] else 'not estimated: selection-induced dependence',
               estimand='mean pooled target Macro-F1 difference conditional on fixed records, split, representation, and fixed C=3 and gamma=0.003' if row['inference_performed'] else 'descriptive difference at settings selected using these same seeds',
               independence_unit='augmentation randomization, not observations, batches, archives, or instruments',
               assumptions='independent pseudorandom augmentation realizations; t-interval approximation with eight seeds; sign null probability=0.5 after excluding exact ties' if row['inference_performed'] else 'descriptive summaries of eight augmentation realizations after same-seed parameter selection; no t interval or sign test applied',
               retrospective=True, family_size=2 if row['inference_performed'] else None,
               population_or_external_generalization=False,
               significance_statement='algorithm-conditional only; no test of generalization to other sensor trajectories')
(OUT/'conditional_seed_statistics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
pd.DataFrame([{k:v for k,v in r.items() if k!='seed_differences'} for r in rows]).to_csv(OUT/'conditional_seed_statistics.csv', index=False)
for row in rows:
    detail = f"CI=[{row['ci_low']:.6f}, {row['ci_high']:.6f}]; p_Holm={row['p_holm']:.6f}" if row['inference_performed'] else 'CI and p-value not estimated after seed-reusing parameter selection'
    print(row['comparison_id'], f"mean={row['mean_difference']:.6f}; positive={row['positive_seeds']}/8; {detail}")
print('STATISTICS PASS: two fixed-parameter conditional tests; two selected-parameter descriptive comparisons; no independent-instrument inference.')
