"""Two decisions per case: choose evidence level, then terminal classification."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from common import CFG,seed_all,fingerprint,write

def hierarchical(variant):
    # Legacy names are inference-only frozen references. New fitting uses V13.
    return variant in ('U-DHRL-V13','U-DHRL-V12','U-DHRL-V11','U-DHRL-V10')

def mask_q(q,goal):
    allowed=torch.ones_like(q,dtype=torch.bool);allowed[goal==0,2]=False
    return q.masked_fill(~allowed,-1e9)

def coarse_fine_advantage(q,goal):
    return mask_q(q,goal)[:,1:].max(1).values-q[:,0]

class LowQ(nn.Module):
    def __init__(self,gdim,edim,mlp=False):
        super().__init__();self.mlp=mlp
        self.global_encoder=nn.Sequential(nn.Linear(gdim,64),nn.ReLU())
        self.local_stats=nn.Sequential(nn.Linear(edim,16),nn.ReLU())
        self.local_encoder=nn.Sequential(nn.Flatten(),nn.Linear(51*4,32),nn.ReLU()) if mlp else nn.LSTM(4,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(64+16+32+2,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,b,goal):
        global_h=self.global_encoder(b['global_state']);local=torch.zeros((len(goal),48),device=goal.device)
        fine=goal==1
        if fine.any():
            if self.mlp:h=self.local_encoder(b['seq'][fine])
            else:
                packed=pack_padded_sequence(b['seq'][fine],b['length'][fine].cpu(),batch_first=True,enforce_sorted=False)
                h=self.local_encoder(packed)[1][0][-1]
            local[fine]=torch.cat((self.local_stats(b['extra'][fine]),h),1)
        return self.head(torch.cat((global_h,local,F.one_hot(goal,2).float()),1))

class Agent(nn.Module):
    def __init__(self,gdim,edim,variant,device):
        super().__init__();self.variant=variant;self.device=device
        self.low=LowQ(gdim,edim,variant=='Pseudo-MLP')
        self.high=nn.Sequential(nn.Linear(gdim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        self.register_buffer('decision_offsets',torch.zeros(2))
        self.to(device)
    @torch.no_grad()
    def predict(self,p):
        self.eval();scores=[];classes=[];goals=[]
        for st in range(0,len(p['score']),128):
            b=to_tensor(p,np.arange(st,min(st+128,len(p['score']))),self.device)
            goal=self.high(b['global_state']).argmax(1) if hierarchical(self.variant) else torch.ones(len(b['score']),dtype=torch.long,device=self.device)
            q=mask_q(self.low(b,goal),goal)
            q[:,0]+=self.decision_offsets[goal]
            s=coarse_fine_advantage(q,goal);label=q.argmax(1)
            label[(goal==0)&(label>0)]=-1
            scores.append(s.cpu().numpy());classes.append(label.cpu().numpy());goals.append(goal.cpu().numpy())
        return np.concatenate(scores),np.concatenate(classes),np.concatenate(goals)


def load_policy_weights(agent, saved):
    """Load a shape-matched policy or historical reference without changing it."""
    result=agent.load_state_dict(saved['state'],strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    assert set(result.missing_keys).issubset({'decision_offsets'}), result.missing_keys
    if result.missing_keys:agent.decision_offsets.zero_()

def to_tensor(p,index,device):
    keys=('global_state','extra','seq','length','score','pseudo','ambiguity','coarse_margin','fine_margin')
    return {k:torch.as_tensor(p[k][index],device=device) for k in keys if k in p}

def teacher_inputs(p,index):
    """Keep all reward inputs in numpy; never expose benchmark labels."""
    keys=('score','pseudo','ambiguity','coarse_margin','fine_margin','coarse_score','fine_score','disagreement')
    return {k:p[k][index] for k in keys if k in p}

def subset(b,index):return {k:v[index] for k,v in b.items()}

class Replay:
    def __init__(self,rng):self.rows=[];self.cursor=0;self.rng=rng
    def add(self,*cols):
        for row in zip(*cols):
            if len(self.rows)<100000:self.rows.append(row)
            else:self.rows[self.cursor]=row
            self.cursor=(self.cursor+1)%100000
    def sample(self,n=128):return np.asarray([self.rows[i] for i in self.rng.choice(len(self.rows),n,replace=False)])

def validation_objective(agent,p,teacher):
    agent.eval();sums=0.;count=0
    with torch.no_grad():
        for st in range(0,len(p['score']),128):
            ix=np.arange(st,min(st+128,len(p['score'])));b=to_tensor(p,ix,agent.device);next_values=[]
            for g in ((0,1) if hierarchical(agent.variant) else (1,)):
                goal=torch.full((len(ix),),g,dtype=torch.long,device=agent.device)
                q=agent.low(b,goal);rr=torch.as_tensor(teacher.rewards(teacher_inputs(p,ix),np.full(len(ix),g)),device=agent.device)
                use=q[:,:2] if g==0 else q;target=rr[:,:2] if g==0 else rr
                sums+=float(F.mse_loss(use,target,reduction='sum'));count+=target.numel()
                ext=torch.as_tensor(teacher.external(teacher_inputs(p,ix),np.full(len(ix),g)),device=agent.device)
                selected=mask_q(q,goal).argmax(1)
                # Validate against the observed terminal reward of the greedy
                # second-stage action, avoiding a self-confirming high-Q target.
                actual=rr.gather(1,selected[:,None]).squeeze(1)
                next_values.append(ext+CFG['gamma']*actual)
            if hierarchical(agent.variant):
                high=agent.high(b['global_state']);tar=torch.stack(next_values,1)
                sums+=.1*float(F.mse_loss(high,tar,reduction='sum'));count+=.1*tar.numel()
    return sums/max(1,count)

def train_policy(train,val,teacher,seed,device,variant,out,epochs=None,warm_path=None):
    best=float('inf');chosen=None;history=[];max_epochs=epochs or CFG['policy_epochs'];attempted_low=0;attempted_high=0
    assert variant in CFG['policies'], 'Frozen references cannot be fitted by V13'
    assert warm_path is None or CFG['warm_start'], 'V13 defaults to fresh, equal-budget policy fitting'
    assert all(k in p for p in (train,val) for k in ('coarse_margin','fine_margin'))
    initial=torch.load(warm_path,map_location='cpu') if warm_path is not None else None
    def snapshot(agent,row,before):
        return dict(state={k:v.detach().cpu().clone() for k,v in agent.state_dict().items()},
                    gdim=train['global_state'].shape[1],edim=train['extra'].shape[1],variant=variant,
                    **row,low_changed=fingerprint(agent.low)!=before[0],high_changed=fingerprint(agent.high)!=before[1])
    for lr in CFG['learning_rates']:
        seed_all(seed);rng=np.random.default_rng(seed)
        agent=Agent(train['global_state'].shape[1],train['extra'].shape[1],variant,device)
        if initial is not None:
            assert initial['gdim']==train['global_state'].shape[1] and initial['edim']==train['extra'].shape[1]
            load_policy_weights(agent,initial)
            # New evidence needs fresh calibration. The frozen reference
            # separately retains its original offsets during evaluation.
            agent.decision_offsets.zero_()
        low_target=copy.deepcopy(agent.low).eval().requires_grad_(False)
        opt=torch.optim.Adam(agent.low.parameters(),lr=lr);high_opt=torch.optim.Adam(agent.high.parameters(),lr=lr)
        replay=Replay(rng);high_replay=Replay(rng);updates=0;high_updates=0;stale=0;local_best=float('inf')
        before=[fingerprint(agent.low),fingerprint(agent.high)];all_b=to_tensor(train,np.arange(len(train['score'])),device)
        if initial is not None:
            value=validation_objective(agent,val,teacher);local_best=value
            row=dict(lr=lr,epoch=0,training_loss=None,validation_objective=value,low_updates=0,high_updates=0)
            history.append(row)
            if value<best:best=value;chosen=snapshot(agent,row,before)
        for epoch in range(max_epochs):
            agent.train();order=rng.permutation(len(train['score']));eps=max(.05,1-epoch/max(1,.7*max_epochs));losses=[]
            for st in range(0,len(order),CFG['batch_size']):
                ix=order[st:st+CFG['batch_size']]
                # Offline exploration covers both evidence levels. No true anomaly label is used.
                idx=np.r_[ix,ix] if hierarchical(variant) else ix
                goals=np.r_[np.zeros(len(ix),int),np.ones(len(ix),int)] if hierarchical(variant) else np.ones(len(ix),int)
                b=subset(all_b,idx);goal=torch.as_tensor(goals,device=device)
                rewards=teacher.rewards(teacher_inputs(train,idx),goals)
                with torch.no_grad():
                    a=mask_q(agent.low(b,goal),goal).argmax(1).cpu().numpy();explore=rng.random(len(idx))<eps
                    for g in (0,1):
                        use=explore&(goals==g);a[use]=rng.integers(2 if g==0 else 3,size=int(use.sum()))
                replay.add(idx,goals,a,rewards[np.arange(len(idx)),a])
                if hierarchical(variant):
                    # Both accessible sub-goals have observable label-free rewards.
                    # Balanced offline coverage prevents early routing collapse.
                    hi=np.r_[ix,ix];hg=np.r_[np.zeros(len(ix),int),np.ones(len(ix),int)]
                    high_replay.add(hi,hg,teacher.external(teacher_inputs(train,hi),hg))
                if len(replay.rows)<128:continue
                z=replay.sample();ii=z[:,0].astype(int);go=torch.as_tensor(z[:,1],dtype=torch.long,device=device)
                ac=torch.as_tensor(z[:,2],dtype=torch.long,device=device);reward=torch.as_tensor(z[:,3],dtype=torch.float32,device=device)
                batch=subset(all_b,ii);q=agent.low(batch,go)
                # Low decision terminates this case; its target has no next-case bootstrap.
                if variant=='Pseudo-MLP':
                    # Direct full-information reward distillation: no chosen-action TD target.
                    direct=subset(all_b,ix);dg=torch.ones(len(ix),dtype=torch.long,device=device)
                    target=torch.as_tensor(teacher.rewards(teacher_inputs(train,ix),np.ones(len(ix),int)),device=device)
                    loss=F.smooth_l1_loss(agent.low(direct,dg),target)
                else:
                    loss=F.smooth_l1_loss(q.gather(1,ac[:,None]).squeeze(1),reward)
                    rr=torch.as_tensor(teacher.rewards(teacher_inputs(train,ii),go.cpu().numpy()),device=device)
                    valid=torch.ones_like(q,dtype=torch.bool);valid[go==0,2]=False
                    loss=loss+CFG['full_reward_weight']*F.smooth_l1_loss(q[valid],rr[valid])
                    adv=coarse_fine_advantage(q,go)
                    desired=coarse_fine_advantage(rr,go)
                    for g in (0,1):
                        use=go==g
                        if int(use.sum())>1:
                            a=adv[use];d=desired[use]
                            loss=loss+CFG['ranking_weight']*F.smooth_l1_loss(a-a.roll(1),d-d.roll(1))
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(agent.low.parameters(),5);opt.step();updates+=1;losses.append(float(loss.detach()))
                if hierarchical(variant) and len(high_replay.rows)>=128:
                    hz=high_replay.sample();hi=hz[:,0].astype(int);gg=torch.as_tensor(hz[:,1],dtype=torch.long,device=device)
                    ext=torch.as_tensor(hz[:,2],dtype=torch.float32,device=device);hb=subset(all_b,hi)
                    with torch.no_grad():
                        # Double-Q bootstrap across two stages of THIS case, not unrelated traces.
                        next_a=mask_q(agent.low(hb,gg),gg).argmax(1)
                        next_v=low_target(hb,gg).gather(1,next_a[:,None]).squeeze(1)
                        actual=torch.as_tensor(teacher.rewards(teacher_inputs(train,hi),gg.cpu().numpy()),device=device).gather(1,next_a[:,None]).squeeze(1)
                        mix=CFG['high_return_mix']
                        target=ext+CFG['gamma']*((1-mix)*next_v+mix*actual)
                    hq=agent.high(hb['global_state']).gather(1,gg[:,None]).squeeze(1)
                    hl=F.smooth_l1_loss(hq,target);high_opt.zero_grad();hl.backward();nn.utils.clip_grad_norm_(agent.high.parameters(),5);high_opt.step();high_updates+=1
                if updates%CFG['target_every']==0:low_target.load_state_dict(agent.low.state_dict())
            if (epoch+1)%5 and epoch+1!=max_epochs:continue
            value=validation_objective(agent,val,teacher)
            row=dict(lr=lr,epoch=epoch+1,training_loss=float(np.mean(losses)),validation_objective=value,low_updates=updates,high_updates=high_updates)
            history.append(row)
            if value<best:
                best=value;chosen=snapshot(agent,row,before)
            if value<local_best-1e-6:local_best=value;stale=0
            else:stale+=1
            write(out/(variant+'_history.json'),history)
            if stale>=CFG['policy_patience']:break
        attempted_low+=updates;attempted_high+=high_updates;del all_b
    assert chosen and attempted_low>0
    if hierarchical(variant):assert attempted_high>0
    chosen.update(warm_source=str(warm_path) if warm_path else None,
                  validation_fallback_to_reference_weights=bool(initial is not None and chosen['epoch']==0),
                  attempted_low_updates=attempted_low,attempted_high_updates=attempted_high,
                  selection_basis='unlabeled validation reward loss; no test F1',
                  warm_offsets_reset=bool(initial is not None),
                  initial_validation_objective=history[0]['validation_objective'] if initial is not None else None)
    if initial is not None:assert chosen['validation_objective']<=history[0]['validation_objective']+1e-12
    agent=Agent(chosen['gdim'],chosen['edim'],variant,device);agent.load_state_dict(chosen['state']);agent.eval()
    torch.save(chosen,out/(variant+'.pt'))
    return agent,{k:v for k,v in chosen.items() if k!='state'}
