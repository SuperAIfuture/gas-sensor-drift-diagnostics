"""Index-ring representation and interpolation controls on the UCI trajectory."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json, hashlib, platform, time, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score
import common as old

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'outputs/representation_controls';OUT.mkdir(parents=True,exist_ok=True)
GRID=[(3.,.003),(3.,'scale'),(10.,.003),(10.,'scale')]
KINDS=['RAW','RING','IDENTITY','DUPMEAN','SCALED-RING','PERM14','PERM15','PERM16']
X,y,conc,batch=old.load_dataset_from_zip('uci270',BASE/'data/raw/uci270.zip')
LABELS=np.arange(1,7)

def metrics(z,p,labels=LABELS):
    return {'macro_f1':float(f1_score(z,p,labels=labels,average='macro',zero_division=0)),
            'fixed6_f1':float(f1_score(z,p,labels=LABELS,average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(z,p))}

def features(kind,fit):
    if kind=='RAW':return X
    values=StandardScaler().fit(X[fit]).transform(X) if kind=='SCALED-RING' else X
    m=values.reshape(-1,16,8).mean(2)
    if kind=='IDENTITY':return np.column_stack([X,m,np.zeros_like(m)])
    if kind=='DUPMEAN':return np.column_stack([X,m,m])
    L=old.build_ring_laplacian(16)
    if kind.startswith('PERM'):
        perm=np.random.default_rng(20261000+int(kind[-2:])).permutation(16)
        L=L[np.ix_(perm,perm)]
    smooth=m@np.linalg.inv(np.eye(16)+.4*L)
    return np.column_stack([X,smooth,m-smooth])

def repr_task(job):
    kind,fold,g=job
    fit=batch<fold;val=batch==fold
    F=features(kind,fit)
    p=old.fit_predict_svm(F[fit],y[fit],F[val],*GRID[g])
    return dict(method=kind,fold=fold,grid=g,n_fit=int(fit.sum()),n_val=int(val.sum()),
                present_labels=','.join(map(str,np.unique(y[val]))),**metrics(y[val],p,np.unique(y[val]))),p

def representation():
    jobs=list(itertools.product(KINDS,[3,4,5,6],range(4)))
    rows=[];pred={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for (job,(row,p)) in zip(jobs,pool.map(repr_task,jobs)):
            rows.append(row);pred[f'{job[0]}__fold{job[1]}__grid{job[2]}']=p
            print('REP VALID',job,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'representation_validation.csv',index=False)
    np.savez_compressed(OUT/'representation_validation_predictions.npz',**pred)
    choices=[]; scores=[];targetpred={}
    for kind in KINDS:
        d=df[df.method==kind]
        selections={'fixed':0,'batch6':int(d[d.fold==6].sort_values(['macro_f1','grid'],ascending=[False,True]).iloc[0].grid),
            'rolling':int(d.groupby('grid').macro_f1.mean().sort_values(ascending=False,kind='stable').index[0]),
            'rolling6':int(d.groupby('grid').fixed6_f1.mean().sort_values(ascending=False,kind='stable').index[0])}
        F=features(kind,batch<=6);cache={}
        for regime,g in selections.items():
            choices.append(dict(method=kind,regime=regime,grid=g))
            if g not in cache:cache[g]=old.fit_predict_svm(F[batch<=6],y[batch<=6],F[batch>=7],*GRID[g])
            p=cache[g];targetpred[f'{regime}__{kind}']=p
            for b in [0,7,8,9,10]:
                sel=np.ones((batch>=7).sum(),bool) if b==0 else batch[batch>=7]==b
                scores.append(dict(method=kind,regime=regime,grid=g,batch=b,n=int(sel.sum()),**metrics(y[batch>=7][sel],p[sel])))
        print('REP TARGET',kind,selections,flush=True)
    pd.DataFrame(choices).to_csv(OUT/'representation_choices.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'representation_scores.csv',index=False)
    np.savez_compressed(OUT/'representation_predictions.npz',y=y[batch>=7],batch=batch[batch>=7],row_index=np.where(batch>=7)[0],**targetpred)
    cov=[]
    for fold in [3,4,5,6]:
        for c in LABELS:
            cov.append(dict(fold=fold,class_id=c,n_fit=int(((batch<fold)&(y==c)).sum()),n_val=int(((batch==fold)&(y==c)).sum())))
    pd.DataFrame(cov).to_csv(OUT/'rolling_class_coverage.csv',index=False)

def augmentation_task(seed):
    fit=batch<=6;test=batch>=7;F=features('RING',fit)
    real=F[fit];yr=y[fit]
    syn={}
    for route in ['mix','conc']:
        syn[route]=old.generate_domain_mixup_samples(real,yr,batch[fit],conc[fit],use_concentration=route=='conc',seed=seed,per_pair=40,beta_alpha=.8)
    rng=np.random.default_rng(90000+seed)
    mins={c:min((syn['mix'][1]==c).sum(),(syn['conc'][1]==c).sum()) for c in LABELS}
    for route in ['mix','conc']:
        a,z=syn[route];ix=np.concatenate([rng.choice(np.where(z==c)[0],size=mins[c],replace=False) for c in LABELS])
        syn['equal_'+route]=(a[ix],z[ix])
    specs=[('default','mix',False,False,False),('real_scaler','mix',True,False,False),
           ('real_weights','mix',False,True,False),('both','mix',True,True,False),
           ('mass','mix',True,True,True),('equal_mix','equal_mix',True,True,True),('equal_conc','equal_conc',True,True,True)]
    rows=[];mass=[];pred={}
    for name,route,realscale,realweight,norm in specs:
        a,z=syn[route];A=np.vstack([real,a]);Y=np.r_[yr,z]
        scaler=StandardScaler().fit(real if realscale else A)
        cw={int(c):len(yr)/(6*(yr==c).sum()) for c in LABELS} if realweight else 'balanced'
        w=np.ones(len(Y))
        if norm:
            for c in LABELS:w[Y==c]=(yr==c).sum()/(Y==c).sum()
        model=SVC(C=3,gamma=.003,class_weight=cw,cache_size=400)
        model.fit(scaler.transform(A),Y,sample_weight=w)
        p=model.predict(scaler.transform(F[test]));pred[name]=p
        for b in [0,7,8,9,10]:
            m=np.ones(test.sum(),bool) if b==0 else batch[test]==b
            rows.append(dict(method=name,seed=seed,batch=b,n=int(m.sum()),n_synthetic=len(z),**metrics(y[test][m],p[m])))
        for c in LABELS:
            weight=cw[int(c)] if isinstance(cw,dict) else len(Y)/(6*(Y==c).sum())
            mass.append(dict(method=name,seed=seed,class_id=int(c),n_real=int((yr==c).sum()),n_synthetic=int((z==c).sum()),class_weight=float(weight),sample_weight=float(w[Y==c][0]),penalty_mass=float(3*weight*w[Y==c].sum())))
        print('AUG',seed,name,rows[-5]['macro_f1'],flush=True)
    return rows,mass,pred

def augmentation():
    rows=[];masses=[];pred={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for seed,(rr,mm,pp) in zip(range(1,9),pool.map(augmentation_task,range(1,9))):
            rows.extend(rr);masses.extend(mm)
            pred.update({f'{name}__{seed}':p for name,p in pp.items()})
    pd.DataFrame(rows).to_csv(OUT/'augmentation_scores.csv',index=False)
    pd.DataFrame(masses).to_csv(OUT/'augmentation_penalty_counts.csv',index=False)
    np.savez_compressed(OUT/'augmentation_predictions.npz',y=y[batch>=7],batch=batch[batch>=7],**pred)

if __name__=='__main__':
    t=time.time();representation();augmentation()
    meta=dict(seconds=time.time()-t,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'data/raw/uci270.zip',BASE/'METHODS.md',Path(__file__)]})
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('REPRESENTATION CONTROLS PASS',json.dumps(meta),flush=True)
