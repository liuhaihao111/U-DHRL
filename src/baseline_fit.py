"""Existing V16 baseline architectures and fit rules, trained equally per arm."""
import time,collections,pickle
import numpy as np
import torch
from sklearn.cluster import KMeans
from common import CFG,seed_all,fingerprint,write
from native_models import FACTORIES
from tranad_training import train_tranad

class Naive:
    def __init__(self,train):
        self.counts=collections.Counter(row.tobytes() for row in train);self.n=len(train)
    def score(self,x):
        return np.array([-np.log((self.counts[row.tobytes()]+1)/(self.n+len(self.counts)+1)) for row in x],np.float64)

@torch.no_grad()
def scores(model,tokens,device):
    if isinstance(model,Naive):return model.score(tokens)
    model.eval()
    return np.concatenate([model.score(torch.as_tensor(tokens[i:i+64],device=device))[0].cpu().numpy()
                           for i in range(0,len(tokens),64)]).astype(np.float64)

def cutoff(score,seed):
    score=np.asarray(score,np.float64);assert np.isfinite(score).all()
    if len(np.unique(score))<3:return float(np.quantile(score,.95))
    centers=np.sort(KMeans(n_clusters=3,n_init=20,random_state=seed).fit(score[:,None]).cluster_centers_.ravel())
    return float((centers[0]+centers[1])/2)

def train(name,train,val,vocab,seed,device,out,epochs=None):
    if name=='Naive':
        model=Naive(train)
        with (out/'Naive.pkl').open('wb') as f:pickle.dump(model,f)
        return model,dict(training_cases=len(train),unique_training_variants=len(model.counts),learned_parameters=False)
    if name=='TranAD':return train_tranad(train,val,vocab,seed,device,out,epochs=epochs)
    assert name in ('DAE','BINet','GRASPED')
    best=float('inf');chosen=None;history=[];updates=0;started=time.time()
    rates=CFG['baseline_learning_rates'][name]
    for lr in rates:
        seed_all(seed);rng=np.random.default_rng(seed)
        model=FACTORIES[name](vocab,52).to(device);initial=fingerprint(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.99) if name=='DAE' else (.9,.999))
        local=float('inf');stale=0
        for ep in range(epochs or CFG['baseline_epochs']):
            model.train();order=rng.permutation(len(train));total=0.;n=0
            for st in range(0,len(order),64):
                ix=order[st:st+64];x=torch.as_tensor(train[ix],device=device)
                optimizer.zero_grad(set_to_none=True);loss=model.objective(x,ep)
                assert torch.isfinite(loss);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(gradient)
                optimizer.step();updates+=1;total+=float(loss.detach())*len(ix);n+=len(ix)
            model.eval();before=fingerprint(model);value=0.
            with torch.no_grad():
                for st in range(0,len(val),64):
                    x=torch.as_tensor(val[st:st+64],device=device);value+=float(model.objective(x,ep))*len(x)
            value/=len(val);assert np.isfinite(value) and fingerprint(model)==before
            history.append(dict(lr=lr,epoch=ep+1,training_loss=total/n,validation_loss=value))
            write(out/(name+'_history.json'),history)
            if value<best:
                best=value;chosen=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    lr=lr,epoch=ep+1,validation_loss=value,model=name,vocab=vocab,width=52)
            if value<local-1e-8:local=value;stale=0
            else:stale+=1
            if (ep+1)%5==0:print('BASELINE_PROGRESS',name,lr,ep+1,value,flush=True)
            if stale>=CFG['baseline_patience']:break
        assert fingerprint(model)!=initial and updates>0
        del model,optimizer
    model=FACTORIES[name](vocab,52).to(device);model.load_state_dict(chosen['state_dict']);model.eval().requires_grad_(False)
    torch.save(chosen,out/(name+'.pt'))
    info={k:v for k,v in chosen.items() if k!='state_dict'}
    info.update(seconds=time.time()-started,updates=updates,learning_rates=rates,maximum_epochs=CFG['baseline_epochs'],
        selection='original un-injected validation objective only',test_used=False,labels_used=False)
    return model,info

def load(name,vocab,device,out):
    if name=='Naive':
        with (out/'Naive.pkl').open('rb') as f:return pickle.load(f)
    m=FACTORIES[name](vocab,52).to(device);saved=torch.load(out/(name+'.pt'),map_location='cpu')
    m.load_state_dict(saved['state_dict']);return m.eval().requires_grad_(False)
