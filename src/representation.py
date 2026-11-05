"""One historical unsupervised sequence VAE; no auxiliary encoder or labels."""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from common import CFG,SOURCE
from native_models import VAE

def load_vae(ds,seed,vocab,device):
    path=SOURCE/'backbones'/ds/str(seed)/'VAE/representation.pt'
    saved=torch.load(path,map_location='cpu')
    model=VAE(vocab,52).to(device)
    model.load_state_dict(saved['state_dict']);model.eval();model.requires_grad_(False)
    return model

@torch.no_grad()
def extract(model,tokens,device):
    chunks={k:[] for k in ('latent','event','entropy','length','stats','native')}
    for st in range(0,len(tokens),128):
        x=torch.as_tensor(tokens[st:st+128],device=device)
        logits,mu,_=model(x);logp=logits[:,1:].log_softmax(-1);target=x[:,1:];mask=target!=0
        error=-logp.gather(-1,target[...,None]).squeeze(-1)
        entropy=-(logp.exp()*logp).sum(-1)
        error=error.masked_fill(~mask,0);entropy=entropy.masked_fill(~mask,0)
        e,h,m=error.cpu().numpy(),entropy.cpu().numpy(),mask.cpu().numpy()
        stats=[]
        for a,b,ok in zip(e,h,m):
            a=a[ok].astype(np.float64);b=b[ok].astype(np.float64);top=np.sort(a)[::-1]
            stats.append([a.mean(),a.std(),top[:3].mean(),top[:5].mean(),a.max(),
                          np.quantile(a,.9),b.mean(),np.maximum(a-b,0).mean(),
                          np.maximum(a-b,0).max()])
        chunks['latent'].append(mu.cpu().numpy());chunks['event'].append(e);chunks['entropy'].append(h)
        chunks['length'].append(m.sum(1));chunks['stats'].append(np.asarray(stats))
        chunks['native'].append((error.sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy())
    return {k:np.concatenate(v) for k,v in chunks.items()}

def robust_fit(a):
    a=np.asarray(a,dtype=np.float64)
    center=np.median(a,axis=0)
    scale=np.maximum((np.quantile(a,.9,axis=0)-np.quantile(a,.1,axis=0))/2.563,.05)
    return center,scale

def robust_apply(a,fit):
    return np.clip((a-fit[0])/fit[1],-10,20)

def robust_thresholds(score,seed):
    score=np.asarray(score,np.float64);lo,hi=np.quantile(score,CFG['teacher_winsor'])
    trimmed=np.clip(score,lo,hi)
    if len(np.unique(trimmed))<3:
        x,y=np.quantile(score,[.8,.95]);return float(x),float(max(y,x+1e-4)),None
    c=np.sort(KMeans(n_clusters=3,n_init=20,random_state=seed).fit(trimmed[:,None]).cluster_centers_.ravel())
    x,y=(c[:-1]+c[1:])/2
    return float(x),float(max(y,x+1e-4)),c.tolist()

def adaptive_thresholds(score,seed):
    """Unlabeled modal reconstruction stratum, with explicit degenerate fallback.

    Low error is not sufficient to identify the entire normal population.
    Merge lower-error strata into the dominant stratum only when its mass
    exceeds the lowest stratum beyond the fixed sampling-uncertainty guard.
    This is a heuristic normal-bulk assumption, not identification of true labels.
    """
    score=np.asarray(score,dtype=np.float64)
    assert score.ndim==1 and len(score)>0 and np.isfinite(score).all()
    old_x,old_y,centers=robust_thresholds(score,seed)
    info=dict(old_x=old_x,old_y=old_y,centers=centers,changed=False)
    if centers is None:return old_x,old_y,dict(info,reason='degenerate_keep_v8')
    c=np.asarray(centers);bounds=(c[:-1]+c[1:])/2
    clipped=np.clip(score,*np.quantile(score,CFG['teacher_winsor']))
    labels=np.searchsorted(bounds,clipped);counts=np.bincount(labels,minlength=3)
    mass=counts/len(score);mode=int(np.argmax(counts))
    uncertainty=CFG['modal_uncertainty_z']*np.sqrt((mass[mode]+mass[0])/len(score))
    info.update(counts=counts.tolist(),modal_cluster=mode,mass_gap=float(mass[mode]-mass[0]),uncertainty=float(uncertainty))
    if mode==0 or mass[mode]-mass[0]<=uncertainty:
        return old_x,old_y,dict(info,reason='lowest_is_dominant_or_uncertain')
    if mode==1:x=float(bounds[1])
    else:
        # No upper cluster remains. Use a within-modal robust upper-tail rule.
        bulk=clipped[labels==2];center=float(np.median(bulk))
        spread=max(float(np.median(np.abs(bulk-center)))*1.4826,float(c[2]-c[1])*.25,1e-6)
        x=center+spread
    tail=clipped[clipped>x]
    gap=max(float(np.median(np.diff(c)))*.5,1e-4)
    if len(tail)>=6 and len(np.unique(tail))>=2:
        cc=np.sort(KMeans(n_clusters=2,n_init=20,random_state=seed).fit(tail[:,None]).cluster_centers_.ravel())
        y=max(float(cc.mean()),x+1e-4)
    else:y=x+gap
    return float(x),float(y),dict(info,changed=True,reason='merge_lower_strata_into_modal_bulk')

def pair_entropy(weights):
    top=np.sort(weights,axis=1)[:,-2:]
    top=top/np.maximum(top.sum(1,keepdims=True),1e-30)
    return -(top*np.log(np.maximum(top,1e-300))).sum(1)/np.log(2)

class Evidence:
    def fit(self,p,seed):
        stats=np.log1p(p['stats']);core=stats[:,[0,2,4]]
        self.core_fit=robust_fit(core);self.edges=np.unique(np.quantile(p['length'],[1/3,2/3]))
        bins=np.searchsorted(self.edges,p['length'],side='right');self.local=[]
        for b in range(len(self.edges)+1):
            values=core[bins==b];n=len(values)
            if n<30:self.local.append(self.core_fit)
            else:
                loc=robust_fit(values);w=n/(n+100.)
                self.local.append(tuple(w*a+(1-w)*g for a,g in zip(loc,self.core_fit)))
        self.latent_fit=robust_fit(p['latent']);latent=robust_apply(p['latent'],self.latent_fit)
        km=KMeans(n_clusters=3,n_init=20,random_state=seed).fit(latent)
        self.centers=km.cluster_centers_
        dist=np.sqrt(((latent[:,None]-self.centers[None])**2).sum(2)).min(1)
        separation=[np.linalg.norm(a-b) for i,a in enumerate(self.centers) for b in self.centers[i+1:]]
        # Within-cluster scale and center separation replace duplicate-sensitive NN distance.
        self.sigma=max(float(np.median(dist[dist>1e-8])) if (dist>1e-8).any() else 0,
                       .25*float(np.median(separation)),.05)
        self.global_fit=robust_fit(self.global_raw(p));self.stats_fit=robust_fit(stats)
        score=self.evidence_score(p);self.x,self.y,self.threshold_diagnostics=adaptive_thresholds(score,seed)
        self.cluster_centers=self.threshold_diagnostics['centers']
        self.reward_scale=max(float(np.quantile(score,.9)-np.quantile(score,.1)),.1)
        self.native_x,self.native_y,self.native_threshold_diagnostics=adaptive_thresholds(np.log1p(p['native']),seed)
        return self

    def core(self,p):
        raw=np.log1p(p['stats'][:,[0,2,4]]);g=robust_apply(raw,self.core_fit)
        bins=np.searchsorted(self.edges,p['length'],side='right');loc=np.empty_like(g)
        for b,fit in enumerate(self.local):
            ix=bins==b;loc[ix]=robust_apply(raw[ix],fit)
        return g,loc

    def evidence_score(self,p):
        g,loc=self.core(p)
        # Global context is retained so length deviations are not completely normalized away.
        return (.5*g+.5*loc)@np.array([.25,.5,.25])

    def global_raw(self,p):
        return np.column_stack((p['latent'],np.log1p(p['length']),np.log1p(p['stats'][:,[0,6]])))

    def ambiguity(self,p):
        z=robust_apply(p['latent'],self.latent_fit)
        logits=-((z[:,None]-self.centers[None])**2).sum(2)/(2*self.sigma**2)
        logits-=logits.max(1,keepdims=True);weights=np.exp(logits);weights/=weights.sum(1,keepdims=True)
        return pair_entropy(weights)

    def transform(self,p):
        u=self.ambiguity(p);g=np.column_stack((robust_apply(self.global_raw(p),self.global_fit),u))
        a,b=self.core(p);extra=np.column_stack((robust_apply(np.log1p(p['stats']),self.stats_fit),a,b))
        n,width=p['event'].shape;mask=np.arange(width)[None]<p['length'][:,None]
        position=np.arange(width)[None]/np.maximum(p['length'][:,None]-1,1)
        seq=np.stack((np.log1p(p['event']),np.log1p(p['entropy']),
                      np.log1p(np.maximum(p['event']-p['entropy'],0)),np.broadcast_to(position,(n,width))),2)
        seq[~mask]=0
        score=self.evidence_score(p);severity=np.searchsorted([self.x,self.y],score)
        return dict(global_state=g.astype(np.float32),extra=extra.astype(np.float32),seq=seq.astype(np.float32),
                    length=p['length'].astype(np.int64),score=score.astype(np.float32),
                    pseudo=severity.astype(np.int64),ambiguity=u.astype(np.float32))

    def rewards(self,score,goal):
        # Both branches share binary reward units. Fine action 0 can reverse a coarse anomaly.
        # Smoothly compress large errors without V8's flat clipped reward tail.
        d=np.arcsinh((score-self.x)/self.reward_scale)
        r=np.column_stack((-d,d,d)).astype(np.float32)
        fine=goal==1;simple=score<self.y
        r[fine&simple,2]-=.1
        r[fine&~simple,1]-=.1
        return r

    def external(self,ambiguity,goal):
        return (goal*(CFG['ambiguity_weight']*ambiguity-CFG['refine_cost'])).astype(np.float32)


def stable_thresholds(score, seed):
    """Bounded upper uncertainty correction to the existing adaptive k-means rule.

    The cap is a change in TRAIN tail mass, not an assumed anomaly prevalence.
    No test distribution, rate or label is an input. Unstable cluster regimes
    retain the original threshold instead of silently selecting a new regime.
    """
    score = np.asarray(score, dtype=np.float64)
    x, y, base = adaptive_thresholds(score, seed)
    info = dict(base, base_x=x, base_y=y, bootstrap_consistent=0, tightened=False)
    repeats = int(CFG['threshold_bootstrap_repeats'])
    if base['centers'] is None or len(score) < 64 or repeats == 0:
        return x, y, dict(info, stability_reason='insufficient_or_degenerate')
    rng = np.random.default_rng(seed + 1010)
    estimates = []
    for _ in range(repeats):
        sample = score[rng.integers(0, len(score), len(score))]
        bx, _, bi = adaptive_thresholds(sample, seed)
        if bi['reason'] == base['reason'] and bi['centers'] is not None:
            estimates.append(bx)
    info['bootstrap_consistent'] = len(estimates)
    if len(estimates) < int(np.ceil(0.75 * repeats)):
        return x, y, dict(info, stability_reason='cluster_regime_unstable')
    candidate = float(np.quantile(estimates, CFG['threshold_bootstrap_quantile']))
    # The maximum allowed increase discards at most 2.5 percentage points of
    # train cases; ties cannot cause an uncontrolled jump across a score mass.
    max_trim = CFG['threshold_max_tail_trim']
    before = float(np.mean(score > x))
    proposed = min(max(x, candidate), y - 1e-4)
    if proposed <= x:
        return x, y, dict(info, bootstrap_x=candidate, stability_reason='no_upward_correction')
    if before - float(np.mean(score > proposed)) > max_trim + 1e-12:
        allowed = np.unique(score[(score > x) & (score <= proposed)])
        ordered = np.sort(score)
        drops = (np.searchsorted(ordered, allowed, side='right') - np.searchsorted(ordered, x, side='right')) / len(score)
        allowed = allowed[drops <= max_trim + 1e-12]
        proposed = float(allowed[-1]) if len(allowed) else x
    after = float(np.mean(score > proposed))
    info.update(bootstrap_x=candidate, train_tail_before=before, train_tail_after=after,
                tightened=bool(proposed > x), stability_reason='bounded_bootstrap_correction')
    return float(proposed), float(y), info


class EvidenceV10(Evidence):
    """Same VAE features and dimensions, with a bounded length-bias correction.

    Keep Evidence unchanged so frozen V9 pickles retain exactly their original
    transform and reward behavior for the reference evaluation.
    """
    def fit(self, p, seed):
        super().fit(p, seed)
        legacy = Evidence.evidence_score(self, p)
        from scipy.stats import spearmanr
        correlations = []
        for j in (0, 2, 4):
            a = np.log1p(p['stats'][:, j])
            rho = float(spearmanr(a, p['length']).statistic) if np.ptp(a) > 1e-8 and np.ptp(p['length']) else 0.
            correlations.append(rho if np.isfinite(rho) else 0.)
        rho = max(0., float(np.dot(correlations, [.25, .5, .25])))
        floor = CFG['length_correlation_floor']
        strength = np.clip((rho - floor) / (1. - floor), 0., 1.)
        self.local_weight = float(.5 + (CFG['length_local_weight_max'] - .5) * strength)
        self.correction_limit = float(CFG['evidence_correction_fraction'] *
                                      max(np.quantile(legacy, .9) - np.quantile(legacy, .1), .1))
        score = self.evidence_score(p)
        self.x, self.y, self.threshold_diagnostics = stable_thresholds(score, seed)
        self.cluster_centers = self.threshold_diagnostics['centers']
        self.reward_scale = max(float(np.quantile(score, .9) - np.quantile(score, .1)), .1)
        self.optimization_diagnostics = dict(length_correlations=correlations, local_weight=self.local_weight,
            correction_limit=self.correction_limit, max_train_score_change=float(np.max(np.abs(score - legacy))),
            train_tail_fraction=float(np.mean(score > self.x)), extra_features_added=False)
        return self

    def evidence_score(self, p):
        g, loc = self.core(p)
        weights = np.array([.25, .5, .25])
        original = (.5 * g + .5 * loc) @ weights
        local_weight = getattr(self, 'local_weight', .5)
        correction = (local_weight - .5) * ((loc - g) @ weights)
        limit = getattr(self, 'correction_limit', 0.)
        return original + np.clip(correction, -limit, limit)
