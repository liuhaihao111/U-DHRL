import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
SOURCE=Path(CFG['source'])

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def code_manifest():
    files=list(ROOT.glob('*.py'))+list((ROOT/'src').glob('*.py'))+[ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'plan.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
def verify(ds=None,seed=None):
    f=json.loads((ROOT/'frozen.json').read_text());assert f['code']==code_manifest(),'Frozen V26 code changed'
    for p,h in f['vendor'].items():assert sha(p)==h,p
    if ds:
        for p,h in f['inputs'][ds+'/'+str(seed)].items():assert sha(p)==h,p
    return f
def development_inputs(ds,seed,arm):
    p=ROOT/'data'/ds/str(seed);m=json.loads((p/'metadata.json').read_text())
    with np.load(p/(arm+'_development.npz')) as z:
        assert set(z.files)=={'train','validation','calibration'}
        parts={k:z[k] for k in z.files}
    ids=[set(m['split_ids'][k]) for k in ('train','validation','calibration','test')]
    assert sum(map(len,ids))==len(set.union(*ids))
    m=dict(m,vocab=m['arm_vocabs'][arm])
    return parts,m

def fit_inputs(ds,seed,arm,train):
    """The same train-only fitting sample is supplied to all six methods.

    Evidence fitting, baseline cutoffs and RL keep the original train array.
    Validation/calibration/test distributions are never reweighted.
    """
    p=ROOT/'data'/ds/str(seed)
    with np.load(p/'fit_indices.npz') as z:ix=z[arm]
    assert ix.shape==(len(train),) and ix.dtype.kind in 'iu'
    assert np.array_equal(ix,np.arange(len(train),dtype=ix.dtype))
    return train[ix],dict(indices_sha256=sha(p/'fit_indices.npz'),
        fitting_cases=len(ix),distinct_original_cases=int(len(np.unique(ix))),
        sampling_applied=False,
        applies_to='all fitting stages use the original training distribution',
        evidence_threshold_RL_original_distribution=True)
def metric(y,s,p):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray(y)>0;p=np.asarray(p,bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    f1=2*tp/max(1,2*tp+fp+fn);normal_f1=2*tn/max(1,2*tn+fp+fn)
    return dict(binary_f1=f1,macro_f1=(f1+normal_f1)/2,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
        fpr=fp/max(1,fp+tn),average_precision=float(average_precision_score(y,s)) if y.any() else None,roc_auc=float(roc_auc_score(y,s)) if np.unique(y).size==2 else None,tp=tp,fp=fp,fn=fn,tn=tn)
