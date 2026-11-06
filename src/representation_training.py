"""Fresh VAE -> causal context -> masked refinement, then freeze. No labels."""
import time
import numpy as np
import torch
from common import CFG,seed_all,fingerprint,write
from native_models import VAE
from context_vae import train_context
from masked_vae import train_masked

def fit_base(train,val,vocab,seed,device,out,epochs=None):
    started=time.time();history=[];best=float('inf');chosen=None;total_updates=0
    for lr in CFG['base_learning_rates']:
        seed_all(seed);rng=np.random.default_rng(seed)
        model=VAE(vocab,train.shape[1]).to(device)
        initial=fingerprint(model); optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        local_best=float('inf');stale=0;encoder_updates=0
        for ep in range(epochs or CFG['base_epochs']):
            model.train();order=rng.permutation(len(train));loss_sum=0.
            for st in range(0,len(order),CFG['base_batch']):
                ix=order[st:st+CFG['base_batch']];x=torch.as_tensor(train[ix],device=device)
                optimizer.zero_grad(set_to_none=True);loss=model.objective(x,ep)
                assert torch.isfinite(loss);loss.backward()
                grad=model.encoder.weight_ih_l0.grad
                assert grad is not None and torch.isfinite(grad).all()
                encoder_updates+=int(float(grad.abs().sum())>0)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                total_updates+=1;loss_sum+=float(loss.detach())*len(ix)
            model.eval();before=fingerprint(model);total=0.
            with torch.no_grad():
                for st in range(0,len(val),CFG['base_batch']):
                    x=torch.as_tensor(val[st:st+CFG['base_batch']],device=device)
                    total+=float(model.objective(x,ep))*len(x)
            assert fingerprint(model)==before
            value=total/len(val);assert np.isfinite(value)
            history.append(dict(lr=lr,epoch=ep+1,train_loss=loss_sum/len(train),validation_loss=value))
            write(out/'base_history.json',history)
            if value<best:
                best=value;chosen=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                      lr=lr,epoch=ep+1,validation_loss=value,initialization_seed=seed)
            if value<local_best-1e-8:local_best=value;stale=0
            else:stale+=1
            if (ep+1)%5==0:print('BASE_PROGRESS',lr,ep+1,value,flush=True)
            if stale>=CFG['base_patience']:break
        assert fingerprint(model)!=initial and encoder_updates>0
        del optimizer,model
    model=VAE(vocab,train.shape[1]).to(device);model.load_state_dict(chosen['state_dict'])
    model.eval().requires_grad_(False);torch.save(chosen,out/'representation.pt')
    info={k:v for k,v in chosen.items() if k!='state_dict'}
    info.update(training_seconds=time.time()-started,attempted_updates=total_updates,uses_test=False,uses_labels=False)
    write(out/'base_selection.json',info)
    return model,info

def retrain(train,val,vocab,data_seed,device,out):
    seed=data_seed+CFG['representation_seed_offset']
    base,binfo=fit_base(train,val,vocab,seed,device,out)
    context,cinfo=train_context(base,train,val,vocab,seed,device,out)
    masked,minfo=train_masked(context,train,val,seed,device,out)
    assert not masked.training and not any(p.requires_grad for p in masked.parameters())
    return masked,dict(base=binfo,context=cinfo,masked=minfo,representation_seed=seed,
        policy_seed=data_seed,selection='Original unchanged validation reconstruction loss only; no anomaly labels',
        interpretation='Full representation pipeline retrained with a predeclared new initialization, not joint RL/VAE training')
