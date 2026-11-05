"""Case-preserving time-tie sensitivity, applied BEFORE coupled test injection."""
import collections,gzip,json,datetime
import xml.etree.ElementTree as ET
import numpy as np

def normalize_ties(events):
    # Preserve the original observed 50-activity window. Never admit later
    # events, change the vocabulary, or erase an injected anomaly afterward.
    events=list(events[:50]);out=[];start=0
    while start<len(events):
        end=start+1;t=events[start].get('time:timestamp')
        while end<len(events) and t is not None and events[end].get('time:timestamp')==t:end+=1
        group=events[start:end]
        if all(e.get('action_code') for e in group):group=sorted(group,key=lambda e:e['action_code'])
        out.extend(group);start=end
    assert len(out)==len(events)
    assert collections.Counter(e['concept:name'] for e in out)==collections.Counter(e['concept:name'] for e in events)
    return out

def read_log(path):
    result={};op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rb') as f:
        for _,tr in ET.iterparse(f,events=('end',)):
            if tr.tag.rsplit('}',1)[-1]!='trace':continue
            name=None;events=[]
            for c in tr:
                if c.tag.rsplit('}',1)[-1]=='event':events.append({a.get('key'):a.get('value') for a in c})
                elif c.get('key')=='concept:name':name=c.get('value')
            assert events and all(e.get('concept:name') for e in events)
            key='%d:%s'%(len(result),name)
            result[key]=dict(raw=[e['concept:name'] for e in events[:50]],
                             tie_order=[e['concept:name'] for e in normalize_ties(events)],full_length=len(events))
            tr.clear()
    return result

def encode(sequences,vocab):
    out=np.zeros((len(sequences),52),np.int64)
    for i,seq in enumerate(sequences):
        assert 0<len(seq)<=50
        ids=[1]+[vocab.get(a,3) for a in seq]+[2];out[i,:len(ids)]=ids
    return out
def enc_key(seq,vocab):return tuple(vocab.get(a,3) for a in seq)

def apply_operation(seq,op):
    out=list(seq);kind=op['type'];pos=int(op['position'])
    if not 0<=pos<len(out):return None
    if kind=='skip':
        if len(out)<=1:return None
        out.pop(pos)
    elif kind in ('insert','rework'):
        if len(out)>=50:return None
        out.insert(pos,op['value'] if kind=='insert' else out[pos])
    elif kind in ('early','late'):
        target=int(op['target'])
        if not (0<=target<len(out)) or (kind=='early' and target>=pos) or (kind=='late' and target<=pos):return None
        out.insert(target,out.pop(pos))
    else:raise ValueError(kind)
    return out

def random_operation(kind,n,vocab,rng):
    pos=int(rng.integers(n));op=dict(type=kind,position=pos)
    if kind=='insert':op['value']=str(rng.choice(vocab))
    if kind=='early':
        if pos==0:return None
        op['target']=int(rng.integers(pos))
    if kind=='late':
        if pos+1>=n:return None
        op['target']=int(rng.integers(pos+1,n))
    return op

def coupled_mutate(raw,ordered,old_ops,vocab,rng):
    """Same types/positions/targets in both arms; same original per-case budget.

    Try historical operation parameters first; replace jointly only if they
    yield an invisible/no-op/cyclic edit in either arm. The model never enters
    this routine. Resampling counts and realized operations are all retained.
    """
    assert len(raw)==len(ordered)
    total_draws=0
    for restart in range(64):
        current=[list(raw),list(ordered)];seen=[{enc_key(s,vocab)} for s in current];realized=[]
        for step,historical in enumerate(old_ops):
            accepted=False
            for attempt in range(512):
                total_draws+=1
                if attempt==0 and restart==0:
                    op={k:v for k,v in historical.items() if k in ('type','position','target','value')}
                else:op=random_operation(historical['type'],len(current[0]),list(vocab),rng)
                if op is None:continue
                updated=[apply_operation(s,op) for s in current]
                if any(s is None for s in updated):continue
                keys=[enc_key(s,vocab) for s in updated]
                if any(k in old for k,old in zip(keys,seen)):continue
                assert len(updated[0])==len(updated[1])
                current=updated
                for k,old in zip(keys,seen):old.add(k)
                realized.append(dict(operation=op,attempt=attempt,restart=restart));accepted=True;break
            if not accepted:break
        else:
            assert len(realized)==len(old_ops)
            assert all(enc_key(a,vocab)!=enc_key(b,vocab) for a,b in zip(current,[raw,ordered]))
            return current,dict(realized=realized,restarts=restart,draws=total_draws,
                changed_historical_recipe=any(r['attempt'] or r['restart'] for r in realized))
    raise RuntimeError('Cannot construct a jointly visible paired anomaly; refuse invalid label')

def selfcheck():
    e=[dict(**{'concept:name':a,'action_code':code,'time:timestamp':t}) for a,code,t in [('B','02','t1'),('A','01','t1'),('C','03','t2')]]
    n=normalize_ties(e);assert [a['concept:name'] for a in n]==['A','B','C']
    assert normalize_ties(n)==n
    e2=[dict(x) for x in e];e2[0].pop('action_code');assert normalize_ties(e2)==e2
    # Sorting never admits the 51st event into the fixed observation window.
    many=[dict(**{'concept:name':str(i),'action_code':'%03d'%(60-i),'time:timestamp':'same'}) for i in range(60)]
    assert set(x['concept:name'] for x in normalize_ties(many))==set(str(i) for i in range(50))
    vocab={'A':4,'B':5,'C':6,'D':7};rng=np.random.default_rng(23)
    for kind in ('skip','insert','rework','early','late'):
        recipe=dict(type=kind,position=1,target=0 if kind=='early' else 2,value='D')
        _,audit=coupled_mutate(['A','B','C'],['B','A','C'],[recipe],vocab,rng)
        assert audit['realized'][0]['operation']['type']==kind
    for i in range(100):
        a,b=coupled_mutate(['A','B','C'],['A','B','C'],[dict(type='early',position=1,target=0)],vocab,rng)[0]
        assert a==b
    return dict(ok=True,stable_ties=True,missing_code_preserved=True,window_membership_preserved=True,
                no_post_injection_sorting=True,coupled_visible_operations=True,identity_treatment_same_injections=True)
