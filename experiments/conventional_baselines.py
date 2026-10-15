from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uci_baselines import ROOT, OUT, metric, old, SEEDS
X,y,conc,b = old.load_dataset_from_zip('uci270',ROOT/'data/raw/uci270.zip')
inner, val, source, target = b<=5, b==6, b<=6, b>=7
lr_grid = [.1,1.,3.,10.]
tree_grid = [('sqrt',1),('sqrt',3),(1.,1),(1.,3)]

def model(name,g,seed):
    if name=='LINEAR':
        return make_pipeline(StandardScaler(),LogisticRegression(C=lr_grid[g],class_weight='balanced',solver='lbfgs',max_iter=3000))
    mf,leaf = tree_grid[g]
    return ExtraTreesClassifier(n_estimators=200,max_features=mf,min_samples_leaf=leaf,class_weight='balanced',random_state=seed,n_jobs=4)

rows=[]
for name in ['LINEAR','TREES']:
    for grid in range(4):
        for seed in (SEEDS if name=='TREES' else [1]):
            m=model(name,grid,seed).fit(X[inner],y[inner])
            rows.append(dict(method=name,grid=grid,seed=seed,**metric(y[val],m.predict(X[val]))))
        print('VALIDATION',name,grid,flush=True)
vf=pd.DataFrame(rows)
vf.to_csv(OUT/'analysis/conventional_validation.csv',index=False)
choices=vf.groupby(['method','grid']).macro_f1.mean().reset_index().sort_values(['method','macro_f1','grid'],ascending=[True,False,True]).drop_duplicates('method')
choices.to_csv(OUT/'analysis/conventional_parameters.csv',index=False)
scores=[]
oldnp=np.load(OUT/'analysis/predictions.npz')
preds={k:oldnp[k] for k in oldnp.files}
for name in ['LINEAR','TREES']:
    grid=int(choices[choices.method==name].grid.iloc[0])
    fixed=None
    for seed in SEEDS:
        if name=='LINEAR' and fixed is not None:
            p=fixed
        else:
            m=model(name,grid,seed).fit(X[source],y[source])
            p=m.predict(X[target])
            fixed=p
        preds[f'source_selected__{name}__{seed}']=p
        for scope,mask in [('pooled',np.ones(int(target.sum()),dtype=bool))]+[(f'batch{k}',b[target]==k) for k in range(7,11)]:
            scores.append(dict(regime='source_selected',method=name,seed=seed,grid=grid,scope=scope,n_synthetic=0,n_test=int(mask.sum()),**metric(y[target][mask],p[mask])))
    print('TARGET',name,'complete',flush=True)
pd.DataFrame(scores).to_csv(OUT/'analysis/conventional_scores.csv',index=False)
np.savez_compressed(OUT/'analysis/predictions.npz',**preds)
print(pd.DataFrame(scores).query('scope=="pooled"').groupby('method')[['macro_f1','accuracy']].agg(['mean','std']).to_string())

print('CONVENTIONAL BASELINES PASS', flush=True)
