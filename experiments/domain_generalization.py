"""Tabular adaptation of the pinned DomainBed CORAL source-domain objective.
Mean/covariance discrepancy is extracted verbatim via AST from archived code.
The tabular MLP, optimizer, and sampling scheme are specified in METHODS.md.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import ast,json,time,platform,itertools,random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score
import common as old

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'outputs/domain_generalization';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
DEVICE='cuda'
X,y,conc,batch=old.load_dataset_from_zip('uci270',BASE/'data/raw/uci270.zip')
y=y-1
tree=ast.parse((BASE/'third_party/domainbed/algorithms.py').read_text(encoding='utf-8'))
cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AbstractMMD')
fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='mmd')
scope={'torch':torch}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_domainbed_mmd','exec'),scope)
class Penalty:
    kernel_type='mean_cov'
    mmd=scope['mmd']
penalty=Penalty()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.classifier=nn.Linear(128,6)
    def forward(self,x):return self.classifier(self.featurizer(x))

GRID={'ERM':[(.0001,0),(.0003,0),(.001,0),(.003,0)],
      'CORAL':[(.0001,.1),(.0001,1),(.001,.1),(.001,1)]}

def metric(z,p):
    return dict(macro_f1=float(f1_score(z,p,labels=np.unique(z),average='macro',zero_division=0)),
        fixed6_f1=float(f1_score(z,p,labels=range(6),average='macro',zero_division=0)),accuracy=float(accuracy_score(z,p)))

def train(method,g,seed,fold):
    torch.cuda.set_device(seed%torch.cuda.device_count())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    fit=batch<fold
    scaler=StandardScaler().fit(X[fit]); xx=torch.tensor(scaler.transform(X),dtype=torch.float32,device=DEVICE)
    yy=torch.tensor(y,dtype=torch.long,device=DEVICE)
    domains=[torch.tensor(np.where(batch==b)[0],device=DEVICE) for b in range(1,fold)]
    model=Model().to(DEVICE);lr,weight=GRID[method][g]
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0)
    model.train()
    for step in range(1000):
        ixs=[ix[torch.randint(len(ix),(32,),device=DEVICE)] for ix in domains]
        # Equal-size domain batches permit a single deterministic forward pass.
        joined=torch.cat(ixs)
        h=model.featurizer(xx[joined]).reshape(len(domains),32,128)
        objective=F.cross_entropy(model.classifier(h.reshape(-1,128)),yy[joined])
        if weight:
            means=h.mean(1);cent=h-means[:,None,:]
            cov=cent.transpose(1,2)@cent/31
            # Symmetric all-pair sum, excluding zero diagonal, equals pair mean.
            md=(means[:,None,:]-means[None,:,:]).square().mean(-1)
            cd=(cov[:,None,:,:]-cov[None,:,:,:]).square().mean((-1,-2))
            p=(md+cd).sum()/(len(domains)*(len(domains)-1))
            loss=objective+weight*p
        else:loss=objective
        optim.zero_grad();loss.backward();optim.step()
    model.eval()
    with torch.no_grad():pred=model(xx).argmax(1).cpu().numpy()
    return pred,model,scaler,dict(loss=float(loss.detach()),objective=float(objective.detach()))

def validation_job(job):
    method,fold,g,seed=job
    p,_,_,loss=train(method,g,seed,fold)
    ix=batch==fold
    return dict(method=method,fold=fold,grid=g,seed=seed,**metric(y[ix],p[ix]),**loss),p[ix]+1

def main():
    t=time.time();rows=[];predictions={}
    jobs=list(itertools.product(GRID,[3,4,5,6],range(4),[1,2,3]))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for job,(row,p) in zip(jobs,pool.map(validation_job,jobs)):
            rows.append(row);method,fold,g,seed=job
            predictions[f'{method}__fold{fold}__grid{g}__seed{seed}']=p
            print('DG VALID',job,flush=True)
            pd.DataFrame(rows).to_csv(OUT/'validation.csv',index=False)
    np.savez_compressed(OUT/'validation_predictions.npz',**predictions)
    df=pd.DataFrame(rows);choice=[];scores=[];pp={}
    for method in GRID:
        d=df[df.method==method]
        selected={}
        for regime in ['batch6','rolling']:
            v=d[d.fold==6] if regime=='batch6' else d
            selected[regime]=int(v.groupby('grid').macro_f1.mean().sort_values(ascending=False,kind='stable').index[0])
        for regime,g in selected.items():choice.append(dict(method=method,regime=regime,grid=g,lr=GRID[method][g][0],penalty=GRID[method][g][1]))
        for g in sorted(set(selected.values())):
            for seed in [1,2,3]:
                p,model,scaler,loss=train(method,g,seed,7)
                torch.save({'state_dict':model.state_dict(),'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_},OUT/f'{method}_grid{g}_seed{seed}.pt')
                for regime,gg in selected.items():
                    if g!=gg:continue
                    pp[f'{regime}__{method}__{seed}']=p[batch>=7]+1
                    for b in [0,7,8,9,10]:
                        m=batch>=7 if b==0 else batch==b
                        scores.append(dict(method=method,regime=regime,grid=g,seed=seed,batch=b,n=int(m.sum()),**metric(y[m],p[m])))
                print('DG TARGET',method,g,seed,flush=True)
    pd.DataFrame(choice).to_csv(OUT/'choices.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',y=y[batch>=7]+1,batch=batch[batch>=7],row_index=np.where(batch>=7)[0],**pp)
    # Verify exact extracted pinned penalty against an independent formula.
    a=torch.arange(80,dtype=torch.float32,device=DEVICE).reshape(10,8)/80
    b=torch.flip(a,[0,1])*.7
    expected=(a.mean(0)-b.mean(0)).square().mean()+(torch.cov(a.T)-torch.cov(b.T)).square().mean()
    assert torch.allclose(penalty.mmd(a,b),expected,rtol=1e-6,atol=1e-8)
    h=torch.stack([a,b,a*.8])
    means=h.mean(1);cent=h-means[:,None,:];cov=cent.transpose(1,2)@cent/(len(a)-1)
    vector=((means[:,None,:]-means[None,:,:]).square().mean(-1)+(cov[:,None,:,:]-cov[None,:,:,:]).square().mean((-1,-2))).sum()/6
    reference=torch.stack([penalty.mmd(aa,bb) for aa,bb in itertools.combinations(h,2)]).mean()
    assert torch.allclose(vector,reference,rtol=1e-6,atol=1e-8)
    meta=dict(seconds=time.time()-t,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),deterministic=True,python=platform.python_version(),numpy=np.__version__,grid=GRID,steps=1000,per_domain_batch=32,penalty_verification='PASS: vectorized penalty matches pinned pairwise function',adaptation='Two 128-unit ReLU layers; domain-balanced minibatches; DomainBed mean/covariance penalty vectorized without changing its definition; no target fitting; no class balancing; no early stopping')
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('DG PASS',json.dumps(meta),flush=True)
if __name__=='__main__':main()
