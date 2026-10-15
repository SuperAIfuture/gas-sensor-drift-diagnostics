from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/figures';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(O/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(O/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(O/f'{name}.tiff',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
order=['RAW','RING','IDENTITY','DUPMEAN','SCALED-RING','PERM14','PERM15','PERM16']
r=pd.read_csv(ROOT/'results/representation_controls/representation_scores.csv');r=r[r.batch==0]
a=pd.read_csv(ROOT/'results/representation_controls/augmentation_scores.csv');a=a[a.batch==0]
e=pd.read_csv(ROOT/'results/independent/scores.csv');e=e[e.scope=='pooled']
fig,ax=plt.subplots(2,1,figsize=(7,8.5),layout='constrained')
for j,(reg,c,m) in enumerate([('fixed','#0072B2','o'),('batch6','#D55E00','s'),('rolling','#009E73','^')]):
    vals=[r[(r.method==k)&(r.regime==reg)].iloc[0].macro_f1 for k in order]
    ax[0].plot(vals,np.arange(8)+(j-1)*.19,m,ms=5,color=c,label={'fixed':'Fixed','batch6':'Batch-6 selected','rolling':'Rolling selected'}[reg])
ax[0].set(yticks=range(8),yticklabels=order,xlabel='Pooled target Macro-F1',title='(a) Unaugmented representations',xlim=(.69,.79));ax[0].invert_yaxis();ax[0].legend(loc='lower left',fontsize=9)
names=['default','real_scaler','real_weights','both','mass','equal_mix','equal_conc']
labels=['Default','Measured scaler','Measured class weights','Both measured-only','Both + penalty mass','Equal-volume mix','Equal-volume concentration']
for i,k in enumerate(names):
    v=a[a.method==k].macro_f1.to_numpy()
    ax[1].plot([v.min(),v.max()],[i,i],color='#999999',lw=2)
    ax[1].scatter(v,i+np.linspace(-.09,.09,len(v)),s=12,color='#0072B2',alpha=.65)
    ax[1].plot(v.mean(),i,'D',color='#D55E00',ms=5)
ax[1].axvline(.7694873024,color='black',ls='--',lw=1,label='Unaugmented RING')
ax[1].set(yticks=range(7),yticklabels=labels,xlabel='Pooled target Macro-F1',title='(b) Interpolation fitting controls');ax[1].invert_yaxis();ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9,frameon=False)
save(fig,'Figure4_controls')
fig,ax=plt.subplots(2,1,figsize=(7,8),layout='constrained')
for j,(reg,c,m) in enumerate([('fixed','#0072B2','o'),('rolling','#D55E00','s')]):
    vals=[e[(e.method==k)&(e.regime==reg)].macro_f1.mean() for k in order]
    ax[0].plot(vals,np.arange(8)+(j-.5)*.2,m,color=c,label=reg.title(),ms=5)
ax[0].set(yticks=range(8),yticklabels=order,xlabel='Pooled target Macro-F1',title='(a) Independent unaugmented controls',xlim=(.435,.535));ax[0].invert_yaxis();ax[0].legend(fontsize=9)
for j,(reg,c,m) in enumerate([('fixed','#0072B2','o'),('rolling','#D55E00','s')]):
    for i,k in enumerate(['RAW','RAW-MIX','RING','RING-MIX']):
        v=e[(e.method==k)&(e.regime==reg)].macro_f1.to_numpy();yy=i+(j-.5)*.24
        ax[1].plot([v.min(),v.max()],[yy,yy],color=c,lw=1.5)
        if len(v)>1:ax[1].scatter(v,yy+np.linspace(-.05,.05,len(v)),s=12,color=c,alpha=.6)
        ax[1].plot(v.mean(),yy,m,color=c,ms=6,label=reg.title() if i==0 else None)
ax[1].set(yticks=range(4),yticklabels=['RAW','RAW-MIX','RING','RING-MIX'],xlabel='Pooled target Macro-F1',title='(b) Interpolation by selection regime',xlim=(.425,.54));ax[1].invert_yaxis();ax[1].legend(fontsize=9)
save(fig,'Figure5_independent')
# Chronological partitions for the two datasets.
fig,ax=plt.subplots(2,1,figsize=(7,6.4),layout='constrained')
counts=np.array([445,1244,1586,161,197,2300,3613,294,470,3600]);colors=['#0072B2']*6+['#D55E00']*4
ax[0].bar(range(1,11),counts,color=colors);ax[0].set(xticks=range(1,11),xlabel='UCI batch (categorical order)',ylabel='Measured records',title='(a) UCI: final source 1-6; target 7-10')
for i,n in enumerate(counts):ax[0].text(i+1,n+70,str(n),ha='center',fontsize=9)
ax[0].set_ylim(0,4300);ax[0].text(.02,.92,'Rolling validation: fit <3 / val 3; <4 / 4; <5 / 5; <6 / 6',transform=ax[0].transAxes,fontsize=8)
d=pd.read_csv(ROOT/'results/independent/day_class_counts.csv');nn=d[[str(i) for i in range(1,6)]].sum(axis=1)
ax[1].bar(d.day,nn,color=['#0072B2' if t<=24 else '#D55E00' for t in d.day]);ax[1].axvline(24.5,color='black',ls='--');ax[1].set(xticks=[1,12,16,20,24,30,36],xlabel='Independent acquisition date index (sorted; unequal elapsed spacing)',ylabel='Measured records',title='(b) Independent data: 431 source / 193 target records',ylim=(0,27))
ax[1].text(.02,.93,'Inner folds: fit 1-12 / val 13-16; 1-16 / 17-20; 1-20 / 21-24',transform=ax[1].transAxes,fontsize=8)
save(fig,'Figure1_protocol')
# Confusion counts from the saved predictions.
z=np.load(ROOT/'results/representation_controls/representation_predictions.npz');p=z['fixed__RING'];y=z['y'];b=z['batch']
fig,ax=plt.subplots(1,2,figsize=(7,3.6),layout='constrained')
from sklearn.metrics import confusion_matrix
for a,mask,t in [(ax[0],np.ones(len(y),bool),'(a) Pooled targets (n=7977)'),(ax[1],b==10,'(b) Batch 10 (n=3600)')]:
    cm=confusion_matrix(y[mask],p[mask],labels=range(1,7));norm=cm/cm.sum(axis=1,keepdims=True)
    im=a.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(6):
        for j in range(6):a.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.55 else 'black')
    a.set(xticks=range(6),yticks=range(6),xticklabels=range(1,7),yticklabels=range(1,7),xlabel='Predicted label',ylabel='True label',title=t)
fig.colorbar(im,ax=ax,label='Within-class proportion',shrink=.85)
save(fig,'FigureS1_confusion')
r.to_csv(O/'Figure4a_source.csv',index=False)
e.to_csv(O/'Figure5_source.csv',index=False)
for p in O.glob('*.tiff'):
    with Image.open(p) as img: assert img.size[0]>=3000 and img.info.get('dpi',(0,))[0]>=599
print('FIGURES PASS: four figures exported and TIFF resolution checked.')
