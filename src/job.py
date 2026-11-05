"""One dataset/seed/arm/method; immutable selection precedes all evaluation."""
import argparse,json,pickle,time
import numpy as np
import torch
from common import ROOT,CFG,write,sha,verify,seed_all,development_inputs,fit_inputs,fingerprint,metric
from native_models import VAE
from context_vae import ContextVAE
from masked_vae import MaskedContextVAE
from retrain_v22 import retrain
from evidence_v14 import EvidenceV14
from evidence_v15 import EvidenceV15
from evidence_v16 import EvidenceV16
from tail_v16 import extract_tail_views
from policy_v16 import Agent,train_policy,load_policy_weights
import baseline_fit

POLICY='U-DHRL-V16'
def folder(stage,ds,seed,arm,model):return ROOT/stage/ds/str(seed)/arm/model
def artifact_hashes(out):return {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='COMPLETE.json'}
def refit(packs,seed,enabled):
    e=EvidenceV14().fit(packs['train'],packs['validation'],seed,enabled)
    e=EvidenceV15().fit(e,packs['train'],packs['validation'],seed)
    return EvidenceV16().fit(e,packs['train'],packs['validation'],seed)
def load_ours(vocab,out,device):
    # Full checkpoint load; no historical VAE or test-dependent fit is used.
    vae=MaskedContextVAE(ContextVAE(VAE(vocab,52),vocab)).to(device)
    saved=torch.load(out/'masked_vae.pt',map_location='cpu');assert saved['output_vocabulary']==vocab
    vae.load_state_dict(saved['state']);vae.eval().requires_grad_(False)
    with (out/'evidence.pkl').open('rb') as f:evidence=pickle.load(f)
    state=torch.load(out/(POLICY+'.pt'),map_location='cpu')
    agent=Agent(state['gdim'],state['edim'],POLICY,device);load_policy_weights(agent,state);agent.eval().requires_grad_(False)
    return vae,evidence,agent

def development(ds,seed,arm,name,device):
    verify(ds,seed);seed_all(seed);start=time.time();out=folder('development',ds,seed,arm,name)
    out.mkdir(parents=True,exist_ok=True);assert not (out/'COMPLETE.json').exists()
    parts,meta=development_inputs(ds,seed,arm);vocab=max(meta['vocab'].values())+1
    fit_train,sampling=fit_inputs(ds,seed,arm,parts['train'])
    if name=='U-DHRL':
        vae,rinfo=retrain(fit_train,parts['validation'],vocab,seed,device,out)
        before=fingerprint(vae);packs={k:extract_tail_views(vae,v,device) for k,v in parts.items()}
        evidence=refit(packs,seed,bool(vae.masked_enabled));inputs={k:evidence.transform(v) for k,v in packs.items()}
        with (out/'evidence.pkl').open('wb') as f:pickle.dump(evidence,f)
        agent,pinfo=train_policy(inputs['train'],inputs['validation'],evidence,seed,device,POLICY,out)
        assert fingerprint(vae)==before and not any(p.requires_grad for p in vae.parameters())
        calibration=agent.predict(inputs['calibration'])[0]
        thresholds=dict(method_decision=0.,common_unlabeled_q95=float(np.quantile(calibration,CFG['quantile'])))
        info=dict(representation=rinfo,policy=pinfo,vae_frozen_during_RL=True,vae_fingerprint=before,agent_fingerprint=fingerprint(agent))
    else:
        model,info=baseline_fit.train(name,fit_train,parts['validation'],vocab,seed,device,out)
        before=None if name=='Naive' else fingerprint(model)
        train_scores=baseline_fit.scores(model,parts['train'],device);calibration=baseline_fit.scores(model,parts['calibration'],device)
        assert np.isfinite(train_scores).all() and np.isfinite(calibration).all()
        thresholds=dict(method_decision=baseline_fit.cutoff(train_scores,seed),common_unlabeled_q95=float(np.quantile(calibration,CFG['quantile'])))
        if before is not None:assert before==fingerprint(model)
        info.update(train_score_unique=int(len(np.unique(train_scores))),parameters_frozen=True)
        np.savez_compressed(out/'unlabeled_scores.npz',train=train_scores,calibration=calibration)
    write(out/'thresholds.json',thresholds)
    write(out/'selection.json',dict(model=name,arm=arm,seed=seed,train_cases=len(parts['train']),
       validation_cases=len(parts['validation']),test_read=False,anomaly_labels_used=False,sampling=sampling,details=info))
    write(out/'COMPLETE.json',dict(ok=True,seconds=time.time()-start,artifacts=artifact_hashes(out),code_hash=sha(ROOT/'frozen.json')))
    print('DEVELOPMENT_COMPLETE',ds,seed,arm,name,time.time()-start,flush=True)

def evaluation(ds,seed,arm,name,device):
    verify(ds,seed);seed_all(seed);start=time.time()
    dev=folder('development',ds,seed,arm,name);out=folder('results',ds,seed,arm,name)
    out.mkdir(parents=True,exist_ok=True);assert not (out/'COMPLETE.json').exists()
    decision=ROOT/'decisions'/(ds+'.json');lock=json.loads(decision.read_text());assert lock['code_hash']==sha(ROOT/'frozen.json')
    for n,h in lock['artifacts'][str(seed)+'/'+arm+'/'+name].items():assert sha(dev/n)==h,n
    parts,meta=development_inputs(ds,seed,arm);vocab=max(meta['vocab'].values())+1
    thresholds=json.loads((dev/'thresholds.json').read_text())
    if name=='U-DHRL':
        vae,evidence,agent=load_ours(vocab,dev,device);models={'vae':vae,'agent':agent}
    else:
        model=baseline_fit.load(name,vocab,device,dev);models={} if name=='Naive' else {name:model}
    before={k:fingerprint(m) for k,m in models.items()}
    write(out/'FROZEN_BEFORE_TEST.json',dict(fingerprints=before,thresholds=thresholds,decision_sha256=sha(decision),time=time.time()))
    rows=[]
    with np.load(ROOT/'data'/ds/str(seed)/(arm+'_test.npz')) as z:
        for rate in CFG['primary_rates']+CFG['extra_rates']:
            x,y=z['test_%02d'%rate],z['labels_%02d'%rate];payload={'y_three':y}
            if name=='U-DHRL':
                inputs=evidence.transform(extract_tail_views(vae,x,device));scores,labels,goals=agent.predict(inputs)
                assert np.array_equal(labels!=0,scores>0)
                payload.update(classes=labels,branches=goals,fine_margin=inputs['fine_margin'],coarse_margin=inputs['coarse_margin'])
            else:scores=baseline_fit.scores(model,x,device)
            assert np.isfinite(scores).all();payload['scores']=scores
            for track,cut in thresholds.items():
                pred=scores>cut;payload[track]=pred
                rows.append(dict(dataset=ds,seed=seed,arm=arm,model=name,rate=rate,track=track,**metric(y,scores,pred)))
            np.savez_compressed(out/('test_%02d.npz'%rate),**payload)
    assert before=={k:fingerprint(m) for k,m in models.items()}
    write(out/'metrics.json',rows);write(out/'COMPLETE.json',dict(ok=True,seconds=time.time()-start,fit_in_test=False))
    print('EVALUATION_COMPLETE',ds,seed,arm,name,time.time()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','evaluation'],required=True)
    p.add_argument('--dataset',choices=CFG['datasets'],required=True);p.add_argument('--seed',type=int,choices=CFG['seeds'],required=True)
    p.add_argument('--arm',choices=CFG['arms'],required=True);p.add_argument('--model',choices=CFG['models'],required=True)
    a=p.parse_args();assert torch.cuda.is_available()
    (development if a.stage=='development' else evaluation)(a.dataset,a.seed,a.arm,a.model,'cuda:0')
