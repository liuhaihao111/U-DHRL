"""Reconstruction evidence for V14; fit on original train/validation only."""
import numpy as np
from common import CFG
from evidence_v12 import EvidenceV12, _Background, _width, local_surprise


def masked_pack(p):
    return dict(p,event=p['masked_event'],entropy=p['masked_entropy'],
                stats=p['masked_stats'],native=p['masked_native'])


class EvidenceV14:
    def fit(self, train, validation, seed, masked_enabled):
        assert not any(k in p for p in (train,validation) for k in ('y','labels','y_three','test'))
        # Restore the V12 entropy-excess evidence, with calibration kept separate.
        self.clean=EvidenceV12().fit(train,validation,seed)
        self.clean.threshold_diagnostics['reference']='original validation only; unlabeled'
        self.compat=self.clean.compat;self.centers=self.clean.centers;self.sigma=self.clean.sigma
        self.masked_enabled=bool(masked_enabled)
        self.masked_weight=float(CFG['masked_evidence_weight']) if self.masked_enabled else 0.
        assert 0<=self.masked_weight<=.5
        c,l,cm,base,lm=self.clean._scores(validation)
        self.masked_background=None;self.masked_x=0.;self.masked_y=1.;self.masked_scale=1.
        m_info=dict(enabled=False,reason='representation validation guard retained original parameters')
        mm=base.copy();mseverity=self.clean.fine_severity_margin
        if self.masked_enabled:
            modes=self.clean._modes(validation)
            self.masked_background=_Background(self.compat.edges).fit(local_surprise(masked_pack(validation)),validation['length'],modes)
            ms=self.masked_background.transform(local_surprise(masked_pack(validation)),validation['length'],modes)
            self.masked_x,self.masked_y,m_info=self.clean._cutoff(ms,CFG['fine_tail_budget'],seed)
            self.masked_scale=_width(ms);mm=(ms-self.masked_x)/self.masked_scale
            mseverity=(self.masked_y-self.masked_x)/self.masked_scale
        combined=(1-self.masked_weight)*base+self.masked_weight*mm
        # The existing validation evidence guard precedes all policy fitting;
        # there is no extra calibration-set alert-rate cap after the Q decision.
        self.fine_shift=max(0.,float(np.quantile(combined,1-CFG['fine_tail_budget'],method='higher')))
        fm=combined-self.fine_shift
        self.x=0.;self.y=max(1e-4,(1-self.masked_weight)*self.clean.fine_severity_margin+self.masked_weight*mseverity-self.fine_shift)
        self.fine_severity_margin=self.y;self.reward_scale=1.
        self.native_x=self.clean.native_x;self.native_y=self.clean.native_y
        self.cluster_centers=dict(clean=self.clean.cluster_centers,masked=m_info.get('centers'))
        self.threshold_diagnostics=dict(reference='original validation only',
          reference_cases=len(fm),clean=self.clean.threshold_diagnostics,masked=m_info,
          masked_enabled=self.masked_enabled,masked_weight=self.masked_weight,
          fine_shift=self.fine_shift,reference_positive_fraction=float((fm>0).mean()),
          calibration_used_for_evidence=False,post_policy_alert_cap=False)
        self.optimization_diagnostics=dict(clean=self.clean.optimization_diagnostics,
          masked_background=None if self.masked_background is None else self.masked_background.diagnostics,
          masked_weight=self.masked_weight,local_evidence='positive reconstruction NLL minus predictive entropy',
          state_margin_transform='asinh, preserving ordering above old clip bound',
          global_excludes_masked_evidence=True,single_shared_VAE=True)
        return self

    def transform(self,p):
        result=self.compat.transform(p)
        cs,ls,cm,base,lm=self.clean._scores(p);mm=base.copy()
        if self.masked_enabled:
            ms=self.masked_background.transform(local_surprise(masked_pack(p)),p['length'],self.clean._modes(p))
            mm=(ms-self.masked_x)/self.masked_scale
        fm=(1-self.masked_weight)*base+self.masked_weight*mm-self.fine_shift
        disagreement=np.tanh(np.abs(fm-cm))
        # Unbounded logarithmic compression preserves score-tail ordering.
        result['global_state']=np.column_stack((result['global_state'],np.arcsinh(cm))).astype(np.float32)
        local=np.column_stack((np.arcsinh(cm),np.arcsinh(fm),np.arcsinh(lm),np.arcsinh(mm),disagreement))
        result['extra']=np.column_stack((result['extra'],local)).astype(np.float32)
        source=masked_pack(p) if self.masked_enabled else p
        n,width=p['event'].shape;mask=np.arange(width)[None]<p['length'][:,None]
        position=np.arange(width)[None]/np.maximum(p['length'][:,None]-1,1)
        seq=np.stack((np.log1p(source['event']),np.log1p(source['entropy']),
          np.log1p(np.maximum(source['event']-source['entropy'],0)),np.broadcast_to(position,(n,width))),2)
        seq[~mask]=0;result['seq']=seq.astype(np.float32)
        result.update(score=fm.astype(np.float32),coarse_score=cs.astype(np.float32),local_score=ls.astype(np.float32),
          fine_score=(fm+self.fine_shift).astype(np.float32),coarse_margin=cm.astype(np.float32),
          fine_margin=fm.astype(np.float32),local_margin=lm.astype(np.float32),masked_margin=mm.astype(np.float32),
          disagreement=disagreement.astype(np.float32),pseudo=np.searchsorted([0.,self.y],fm).astype(np.int64))
        assert all(np.isfinite(v).all() for v in result.values())
        return result

    def rewards(self,p,goal):
        d=np.arcsinh(p['fine_margin']);r=np.column_stack((-d,d,d)).astype(np.float32)
        fine=np.asarray(goal)==1;simple=np.asarray(p['fine_margin'])<self.y
        r[fine&simple,2]-=.1;r[fine&~simple,1]-=.1
        return r

    def external(self,p,goal):return (-CFG['refine_cost']*np.asarray(goal)).astype(np.float32)
