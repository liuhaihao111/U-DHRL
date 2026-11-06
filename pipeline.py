"""Standalone raw-log -> preprocessing/injection -> training -> evaluation pipeline."""
import argparse,collections,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def prepare(raw,out,seed,cfg):
 import numpy as np
 from standalone_data import read_xes,encode as activity_encode,mutate
 from paired_data import coupled_mutate
 from preprocessing import read_events,choose_representation,vocabulary,encode,replay
 raw=Path(raw).resolve();cases=read_xes(raw);events=read_events(raw)
 rng=np.random.default_rng(seed);perm=rng.permutation(len(cases));nt=int(.7*len(cases));nv=int(.1*len(cases))
 parts={'train':perm[:nt],'validation':perm[nt:nt+nv//2],'calibration':perm[nt+nv//2:nt+nv],'test':perm[nt+nv:]}
 if min(map(len,parts.values()))<2:raise ValueError('At least 40 cases are required for nonempty validation/calibration partitions.')
 split={k:[cases[i]['id'] for i in v] for k,v in parts.items()}
 seqs={k:[events[i] for i in ids] for k,ids in split.items()}
 av={a:i+4 for i,a in enumerate(sorted({a for ix in parts['train'] for a in cases[ix]['activities'][:50]}))}
 compound,decision=choose_representation(seqs['train']);vocab=vocabulary(seqs['train'],compound)
 arrays={k:encode(v,vocab,compound) for k,v in seqs.items()}
 train_events=collections.defaultdict(list)
 for sequence in seqs['train']:
  for e in sequence:train_events[e['concept:name']].append(e)
 order=rng.permutation(len(parts['test']));ranks=np.empty(len(order),int);ranks[order]=np.arange(len(order))
 modified=[];labels=[];recipes=[]
 for i,ix in enumerate(parts['test']):
  original=cases[ix]['activities'][:50];count=1 if ranks[i]%2==0 else int(rng.integers(2,4))
  historical,ops=mutate(original,list(av),count,rng)
  if np.array_equal(activity_encode([original],av),activity_encode([historical],av)):
   raise ValueError('Historical injection invisible after encoding; refusing invalid labels.')
  ordered=[e['concept:name'] for e in seqs['test'][i]]
  paired,info=coupled_mutate(original,ordered,ops,av,np.random.default_rng(np.random.SeedSequence([23023,seed,i])))
  changed,realized=replay(seqs['test'][i],info['realized'],train_events,np.random.default_rng(np.random.SeedSequence([cfg['preprocessing_seed_offset'],seed,i])))
  assert [e['concept:name'] for e in changed]==paired[1]
  modified.append(changed);labels.append(1 if count==1 else 2)
  recipes.append(dict(case_id=cases[ix]['id'],selection_rank=int(ranks[i]),label=labels[-1],operations=realized))
 altered=encode(modified,vocab,compound)
 if not np.all(np.any(altered!=arrays['test'],axis=1)):raise ValueError('Injection invisible in final representation.')
 payload={}
 for rate in cfg['primary_rates']+cfg['extra_rates']:
  selected=ranks<int(round(rate/100*len(order)));x=arrays['test'].copy();y=np.zeros(len(order),np.int64)
  x[selected]=altered[selected];y[selected]=np.asarray(labels)[selected]
  payload[f'test_{rate:02d}']=x;payload[f'labels_{rate:02d}']=y
 out.mkdir(parents=True)
 np.savez_compressed(out/'final_development.npz',**{k:arrays[k] for k in ['train','validation','calibration']})
 np.savez_compressed(out/'final_test.npz',**payload)
 np.savez_compressed(out/'fit_indices.npz',final=np.arange(len(arrays['train']),dtype=np.int64))
 write(out/'metadata.json',dict(seed=seed,source=str(raw),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),split_ids=split,arm_vocabs={'final':vocab},vocab=vocab,representation=decision,standalone_reconstruction=True))
 write(out/'injections.json',recipes)

def worker(work,ds,seed,model,device,stage):
 sys.path.insert(0,str(work/'src'))
 import torch
 # PyTorch >=2.6 changes the default; load only locally generated trusted checkpoints.
 original_load=torch.load
 def trusted_load(*a,**kw):
  kw.setdefault('weights_only',False);return original_load(*a,**kw)
 torch.load=trusted_load
 import job
 if stage=='train':job.development(ds,seed,'final',model,device)
 else:job.evaluation(ds,seed,'final',model,device)

def summarize(work):
 import numpy as np
 rows=[]
 for p in sorted((work/'results').glob('*/*/final/*/metrics.json')):rows.extend(json.loads(p.read_text()))
 dest=work/'summary';dest.mkdir()
 with (dest/'all_metrics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups=collections.defaultdict(list)
 for r in rows:
  if r['track']=='method_decision' and r['rate'] in range(5,46,5):groups[r['dataset'],r['model'],r['seed']].append(r['binary_f1'])
 seedmeans={k:float(np.mean(v)) for k,v in groups.items()};records=[]
 for ds,model in sorted({k[:2] for k in groups}):
  values=[v for k,v in seedmeans.items() if k[:2]==(ds,model)]
  records.append(dict(dataset=ds,model=model,n_seeds=len(values),binary_f1_mean=float(np.mean(values)),binary_f1_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
 write(dest/'binary_summary.json',records)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--logs',help='JSON mapping dataset names to local XES/XES.GZ paths')
 p.add_argument('--output',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[42,123,456,789,2024])
 p.add_argument('--models',nargs='+',default=['U-DHRL'],choices=['U-DHRL','Naive','DAE','BINet','TranAD','GRASPED'])
 p.add_argument('--device',default='cpu');p.add_argument('--smoke',action='store_true',help='1 epoch per stage; functionality only, not paper settings')
 p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',nargs=5,metavar=('WORK','DATASET','SEED','MODEL','STAGE'))
 a=p.parse_args()
 if a.worker:
  w,d,s,m,stage=a.worker;worker(Path(w),d,int(s),m,a.device,stage);return
 if not a.logs or not a.output:p.error('--logs and --output are required')
 logs=json.loads(Path(a.logs).read_text(encoding='utf-8-sig'));cfg=json.loads((ROOT/'config.json').read_text())
 if not logs:p.error('No event logs specified')
 if len(set(a.seeds))!=len(a.seeds) or len(set(a.models))!=len(a.models):p.error('Duplicate seeds or models')
 for ds,path in logs.items():
  if ds not in cfg['datasets']:p.error('Unknown dataset name: '+ds)
  if not Path(path).is_file():p.error('Missing raw log: '+path)
 if 'GRASPED' in a.models and not (ROOT/'vendor/guanwei49__GRASPED/model/GRU_AE.py').is_file():
  p.error('GRASPED external implementation missing; see docs/THIRD_PARTY.md. U-DHRL runs without it.')
 work=a.output.resolve()
 if work.exists():p.error('Output already exists; use a new directory')
 work.mkdir(parents=True);shutil.copytree(ROOT/'src',work/'src')
 if (ROOT/'vendor').exists():shutil.copytree(ROOT/'vendor',work/'vendor')
 cfg.update(datasets=list(logs),seeds=a.seeds,models=a.models,source=str(work),execution_order=list(logs),expected_jobs=len(logs)*len(a.seeds)*len(a.models))
 if a.smoke:
  for k in ['base_epochs','representation_epochs','masked_epochs','policy_epochs','baseline_epochs','tranad_epochs']:cfg[k]=1
  for k in ['base_learning_rates','representation_learning_rates','masked_learning_rates','learning_rates','tranad_learning_rates']:cfg[k]=cfg[k][:1]
  cfg['representation_mix_grid']=[0.0];cfg['baseline_learning_rates']={k:v[:1] for k,v in cfg['baseline_learning_rates'].items()}
 write(work/'config.json',cfg);(work/'PROTOCOL.md').write_text('Standalone run. Smoke='+str(a.smoke),encoding='utf-8')
 sys.path.insert(0,str(work/'src'))
 for ds,raw in logs.items():
  for seed in a.seeds:
   prepare(raw,work/'data'/ds/str(seed),seed,cfg);print('PREPARED',ds,seed,flush=True)
 if a.prepare_only:return
 import common
 jobs=[dict(dataset=ds,seed=s,arm='final',model=m) for ds in logs for s in a.seeds for m in a.models]
 inputs={f'{ds}/{s}':{str(f):common.sha(f) for f in (work/'data'/ds/str(s)).iterdir()} for ds in logs for s in a.seeds}
 write(work/'plan.json',{'jobs':jobs,'inputs':inputs})
 vendor={str(f):common.sha(f) for f in (work/'vendor').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
 write(work/'frozen.json',dict(code=common.code_manifest(),inputs=inputs,vendor=vendor))
 for ds in logs:
  artifacts={}
  for s in a.seeds:
   for m in a.models:
    subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'--worker',str(work),ds,str(s),m,'train','--device',a.device],check=True)
    done=json.loads((work/'development'/ds/str(s)/'final'/m/'COMPLETE.json').read_text());artifacts[f'{s}/final/{m}']=done['artifacts']
  write(work/'decisions'/(ds+'.json'),{'code_hash':common.sha(work/'frozen.json'),'artifacts':artifacts})
  for s in a.seeds:
   for m in a.models:subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'--worker',str(work),ds,str(s),m,'evaluate','--device',a.device],check=True)
 summarize(work);write(work/'RUN_COMPLETE.json',dict(smoke=a.smoke,jobs=len(jobs),device=a.device));print('COMPLETE',work,flush=True)

if __name__=='__main__':main()
