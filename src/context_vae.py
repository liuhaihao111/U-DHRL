"""Prefix-conditioned residual decoder for the historical sequence VAE.

The original latent encoder/decoder is frozen. A causal activity-prefix GRU
learns bounded logit corrections from original training sequences, without
anomaly labels. Reconstruction validation chooses checkpoint and mixing weight.
This changes the VAE decoder and must be described in the manuscript.
"""
import copy
import numpy as np
import torch
from torch import nn
from common import CFG,seed_all,fingerprint,write
from native_models import per_case_ce
from representation import load_vae


class ContextVAE(nn.Module):
    def __init__(self,base,vocab):
        super().__init__();self.base=base.eval().requires_grad_(False)
        self.embedding=copy.deepcopy(base.embedding).requires_grad_(True)
        self.dropout=nn.Dropout(CFG['context_dropout'])
        self.context=nn.GRU(self.embedding.embedding_dim,CFG['context_hidden'],batch_first=True)
        self.output=nn.Linear(CFG['context_hidden'],vocab)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('mix',torch.tensor(0.))
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def correction(self,x):
        # Output at position j sees input tokens 0..j-1 only. It never sees x[j].
        h,_=self.context(self.dropout(self.embedding(x[:,:-1])))
        cap=CFG['context_logit_bound'];delta=cap*torch.tanh(self.output(h)/cap)
        return torch.cat((torch.zeros_like(delta[:,:1]),delta),1)
    def forward(self,x):
        with torch.no_grad():logits,mu,logvar=self.base(x)
        if float(self.mix)!=0:logits=logits+self.mix*self.correction(x)
        return logits,mu,logvar


@torch.no_grad()
def cached_logits(base,tokens,device):
    return torch.cat([base(torch.as_tensor(tokens[i:i+128],device=device))[0].cpu()
                      for i in range(0,len(tokens),128)])


@torch.no_grad()
def validation_losses(model,tokens,old_logits,device):
    model.eval();losses=np.zeros(len(CFG['representation_mix_grid']),dtype=np.float64)
    for st in range(0,len(tokens),128):
        x=torch.as_tensor(tokens[st:st+128],device=device);old=old_logits[st:st+128].to(device)
        delta=model.correction(x)
        for j,mix in enumerate(CFG['representation_mix_grid']):
            ce=per_case_ce((old+mix*delta)[:,1:],x[:,1:])[0]
            losses[j]+=float(ce.sum())
    return losses/len(tokens)


def train_context(base,train,val,vocab,seed,device,out,epochs=None):
    """No calibration, test tokens, perturbations or anomaly labels accepted."""
    base.eval().requires_grad_(False);base_hash=fingerprint(base)
    cache_train=cached_logits(base,train,device);cache_val=cached_logits(base,val,device)
    baseline=float(per_case_ce(cache_val[:,1:],torch.as_tensor(val[:,1:]))[0].mean())
    best=baseline;chosen=None;history=[];total_updates=0;max_epochs=epochs or CFG['representation_epochs']
    for lr in CFG['representation_learning_rates']:
        seed_all(seed+1100);rng=np.random.default_rng(seed+1100)
        model=ContextVAE(base,vocab).to(device)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if chosen is None:chosen=dict(state=initial,epoch=0,lr=None,mix=0.,validation_nll=baseline)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=CFG['context_weight_decay'])
        stale=0;local_best=baseline
        for epoch in range(max_epochs):
            model.train();order=rng.permutation(len(train));loss_sum=0.;count=0
            for st in range(0,len(order),CFG['batch_size']):
                ix=order[st:st+CFG['batch_size']];x=torch.as_tensor(train[ix],device=device)
                logits=cache_train[ix].to(device)+model.correction(x)
                loss=per_case_ce(logits[:,1:],x[:,1:])[0].mean()
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
                optimizer.step();total_updates+=1;loss_sum+=float(loss.detach())*len(ix);count+=len(ix)
            if (epoch+1)%5 and epoch+1!=max_epochs:continue
            values=validation_losses(model,val,cache_val,device);j=int(np.argmin(values));value=float(values[j])
            row=dict(lr=lr,epoch=epoch+1,train_nll=loss_sum/count,validation_nlls=values.tolist(),mix=CFG['representation_mix_grid'][j])
            history.append(row);write(out/'context_history.json',history)
            if value<best:
                best=value;chosen=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                      epoch=epoch+1,lr=lr,mix=CFG['representation_mix_grid'][j],validation_nll=value)
            if value<local_best-1e-5:local_best=value;stale=0
            else:stale+=1
            if stale>=CFG['representation_patience']:break
    assert total_updates>0 and fingerprint(base)==base_hash
    gain=(baseline-best)/max(baseline,1e-12)
    accepted=gain>=CFG['representation_min_gain'] and chosen['mix']>0
    if not accepted:chosen['mix']=0.
    chosen['state']['mix']=torch.tensor(chosen['mix'])
    info=dict(epoch=chosen['epoch'],lr=chosen['lr'],mix=chosen['mix'],accepted=bool(accepted),
              baseline_validation_nll=baseline,candidate_validation_nll=best,
              deployed_validation_nll=best if accepted else baseline,relative_nll_gain=gain,
              attempted_updates=total_updates,base_unchanged=True,uses_anomaly_labels=False,
              selection='original validation reconstruction NLL only',vocab=vocab)
    model=ContextVAE(base,vocab).to(device);model.load_state_dict(chosen['state']);model.eval().requires_grad_(False)
    torch.save(dict(state=chosen['state'],**info),out/'context_vae.pt');write(out/'context_selection.json',info)
    return model,info


def load_context(ds,seed,vocab,device,path):
    base=load_vae(ds,seed,vocab,device);model=ContextVAE(base,vocab).to(device)
    saved=torch.load(path,map_location='cpu');model.load_state_dict(saved['state']);model.eval().requires_grad_(False)
    return model
