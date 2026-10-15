from pathlib import Path
import sys,json,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score,accuracy_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/diagnostics';OUT.mkdir(parents=True,exist_ok=True)
import common as old
X,y,c,b=old.load_dataset_from_zip('uci270',ROOT/'data/raw/uci270.zip')
src=ROOT/'results/representation_controls'
p=np.load(src/'representation_predictions.npz')
assert np.array_equal(p['row_index'],np.where(b>=7)[0])
rows=[];ranges=[]
for gas in range(1,7):
    source=(b<=6)&(y==gas)
    for batch in range(7,11):
        target=(b==batch)&(y==gas)
        lo=max(c[source].min(),c[target].min());hi=min(c[source].max(),c[target].max())
        measured=(b==batch)&(y==gas)&(c>=lo)&(c<=hi) if lo<=hi else np.zeros(len(y),bool)
        r=dict(batch=batch,class_id=gas,n_source=int(source.sum()),source_min=c[source].min(),source_max=c[source].max(),source_distinct=int(len(np.unique(c[source]))),
            n_target=int(target.sum()),target_min=c[target].min(),target_max=c[target].max(),target_distinct=int(len(np.unique(c[target]))),common_min=lo,common_max=hi,n_common=int(measured.sum()))
        for method in ['RAW','RING']:
            pred=p['fixed__'+method];targetmask=target[b>=7];commonmask=measured[b>=7]
            r[method+'_recall']=float((pred[targetmask]==gas).mean())
            r[method+'_common_recall']=float((pred[commonmask]==gas).mean()) if commonmask.any() else np.nan
            r[method+'_class_f1']=float(f1_score(y[b==batch],pred[b[b>=7]==batch],labels=[gas],average='macro',zero_division=0))
        r['recall_delta']=r['RING_recall']-r['RAW_recall'];r['f1_delta']=r['RING_class_f1']-r['RAW_class_f1']
        ranges.append(r)
pd.DataFrame(ranges).to_csv(OUT/'class_concentration_diagnostics.csv',index=False)
# Recalculate saved target metrics from the prediction arrays.
checks=[]
for category,npz,csv in [('representation','representation_predictions.npz','representation_scores.csv'),('augmentation','augmentation_predictions.npz','augmentation_scores.csv')]:
    pp=np.load(src/npz);df=pd.read_csv(src/csv)
    diffs=[]
    for _,r in df.iterrows():
        key=f'{r.regime}__{r.method}' if category=='representation' else f'{r.method}__{int(r.seed)}'
        ix=np.ones(len(pp['y']),bool) if r.batch==0 else pp['batch']==r.batch
        value=f1_score(pp['y'][ix],pp[key][ix],labels=range(1,7),average='macro',zero_division=0)
        diffs.append(abs(value-r.macro_f1))
    checks.append(dict(category=category,rows=len(df),max_abs_difference=max(diffs)))
    assert max(diffs)<1e-12
# The default interpolation route has identical predictions in both result sets.
oldp=np.load(ROOT/'results/uci_baselines/predictions.npz')
newp=np.load(src/'augmentation_predictions.npz')
for seed in range(1,9):assert np.array_equal(oldp[f'matched__RING-MIX__{seed}'],newp[f'default__{seed}'])
penalty=pd.read_csv(src/'augmentation_penalty_counts.csv')
assert np.allclose(penalty[penalty.method.isin(['mass','equal_mix','equal_conc'])].penalty_mass,5933/2)
counts=penalty[penalty.method.isin(['equal_mix','equal_conc'])].pivot(index=['seed','class_id'],columns='method',values='n_synthetic')
assert (counts.equal_mix==counts.equal_conc).all()
result=dict(target_metric_checks=checks,original_augmented_predictions_identical=True,penalty_mass_and_equal_class_volumes='PASS')
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('DIAGNOSTICS PASS',json.dumps(result))
print(pd.DataFrame(ranges)[['batch','class_id','n_target','n_common','RAW_recall','RING_recall','RAW_common_recall','RING_common_recall']].to_string(index=False))
