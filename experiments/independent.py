import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools,json,time,hashlib
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,recall_score
import common as old
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'outputs/independent';OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(BASE/'data/independent/extracted_features.csv',sep=' ')
cols=[n for i in range(1,63) for n in [f'mean_R{i}[Ohm]',f'rel_diff_R{i}',f'sample_startslope_R{i}',f'rec_lvl_R{i}']]
X=d[cols].to_numpy();assert X.shape==(624,248) and np.isfinite(X).all()
classes=sorted(d.Class.unique());y=pd.Categorical(d.Class,categories=classes).codes+1
date=pd.to_datetime(d['Date[yyyy-mm-dd]'],format='%y-%m-%d');days=sorted(date.unique());batch=np.array([days.index(v)+1 for v in date])
GRID=[(3.,.003),(3.,'scale'),(10.,.003),(10.,'scale')]
KINDS=['RAW','RING','IDENTITY','DUPMEAN','SCALED-RING','PERM14','PERM15','PERM16','RAW-MIX','RING-MIX']

def features(kind,fit):
    kind=kind.replace('-MIX','')
    if kind=='RAW':return X
    values=StandardScaler().fit(X[fit]).transform(X) if kind=='SCALED-RING' else X
    m=values.reshape(-1,62,4).mean(2)
    if kind=='IDENTITY':return np.c_[X,m,np.zeros_like(m)]
    if kind=='DUPMEAN':return np.c_[X,m,m]
    L=old.build_ring_laplacian(62)
    if kind.startswith('PERM'):
        p=np.random.default_rng(20261000+int(kind[-2:])).permutation(62);L=L[np.ix_(p,p)]
    s=m@np.linalg.inv(np.eye(62)+.4*L)
    return np.c_[X,s,m-s]

def fitpredict(kind,g,seed,nfit,nval=0):
    fit=batch<=nfit;test=(batch>nfit)&(batch<=nval) if nval else batch>24
    ff=features(kind,fit)
    a,z=old.build_training_matrix_with_mixup(ff[fit],y[fit],batch[fit],np.zeros(fit.sum()),use_mixup=kind.endswith('-MIX'),use_concentration_mixup=False,seed=seed,per_pair=40,beta_alpha=.8)
    p=old.fit_predict_svm(a,z,ff[test],*GRID[g])
    return p,len(z)-fit.sum()

def metric(z,p):
    return dict(macro_f1=float(f1_score(z,p,labels=range(1,6),average='macro',zero_division=0)),accuracy=float(accuracy_score(z,p)))

def valjob(job):
    kind,nfit,g,seed=job;nval=nfit+4
    p,n=fitpredict(kind,g,seed,nfit,nval);ix=(batch>nfit)&(batch<=nval)
    return dict(method=kind,fold_end=nval,grid=g,seed=seed,n_fit=int((batch<=nfit).sum()),n_val=int(ix.sum()),n_synthetic=int(n),**metric(y[ix],p)),p

def main():
    t=time.time();jobs=[(k,f,g,s) for k in KINDS for f in [12,16,20] for g in range(4) for s in (range(1,9) if k.endswith('-MIX') else [1])]
    rows=[];vp={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for job,(r,p) in zip(jobs,pool.map(valjob,jobs)):
            rows.append(r);vp['__'.join(map(str,job))]=p
            if job[-1]==1:print('EXT VALID',job,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'validation.csv',index=False)
    np.savez_compressed(OUT/'validation_predictions.npz',**vp)
    choices=[];scores=[];pp={}
    for kind in KINDS:
        g=int(df[df.method==kind].groupby('grid').macro_f1.mean().sort_values(ascending=False,kind='stable').index[0])
        choices.append(dict(method=kind,grid=g))
        for seed in (range(1,9) if kind.endswith('-MIX') else [1]):
            cache={}
            for regime,gg in [('fixed',0),('rolling',g)]:
                if gg not in cache:cache[gg]=fitpredict(kind,gg,seed,24)
                p,n=cache[gg];pp[f'{regime}__{kind}__{seed}']=p
                for scope in ['pooled','early','late']:
                    mask=np.ones((batch>24).sum(),bool) if scope=='pooled' else (batch[batch>24]<=30 if scope=='early' else batch[batch>24]>30)
                    scores.append(dict(method=kind,regime=regime,grid=gg,seed=seed,scope=scope,n=int(mask.sum()),n_synthetic=int(n),**metric(y[batch>24][mask],p[mask])))
        print('EXT TARGET',kind,g,flush=True)
    pd.DataFrame(choices).to_csv(OUT/'choices.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',y=y[batch>24],batch=batch[batch>24],row_index=np.where(batch>24)[0],**pp)
    pd.crosstab(pd.Series(batch,name='day'),pd.Series(y,name='class')).to_csv(OUT/'day_class_counts.csv')
    for c in range(1,6):assert ((batch<=24)&(y==c)).any() and ((batch>24)&(y==c)).any()
    meta=dict(seconds=time.time()-t,shape=list(X.shape),source_records=int((batch<=24).sum()),target_records=int((batch>24).sum()),source_days=24,target_days=12,source_end=str(pd.Timestamp(days[23]).date()),target_start=str(pd.Timestamp(days[24]).date()),classes={str(i+1):v for i,v in enumerate(classes)},days=[str(pd.Timestamp(v).date()) for v in days],feature_columns=cols,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'data/independent/extracted_features.csv',BASE/'METHODS.md',Path(__file__)]})
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('INDEPENDENT PASS',json.dumps({k:v for k,v in meta.items() if k not in ['feature_columns','days']}),flush=True)
if __name__=='__main__':main()
