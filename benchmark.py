"""Compare eager and on-demand inference using a completed standalone run."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--rate',type=int,default=25);p.add_argument('--device',default='cpu');p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
 torch.set_num_threads(2)
 work=a.run.resolve();sys.path.insert(0,str(work/'src'))
 from job import load_ours
 from tail_features import extract_tail_views
 from inference_lazy import predict
 from common import metric
 load=torch.load
 def trusted(*x,**kw):kw.setdefault('weights_only',False);return load(*x,**kw)
 torch.load=trusted
 m=json.loads((work/'data'/a.dataset/str(a.seed)/'metadata.json').read_text());vocab=max(m['arm_vocabs']['final'].values())+1
 vae,evidence,agent=load_ours(vocab,work/'development'/a.dataset/str(a.seed)/'final/U-DHRL',a.device)
 with np.load(work/'data'/a.dataset/str(a.seed)/'final_test.npz') as z:x=z[f'test_{a.rate:02d}'];y=z[f'labels_{a.rate:02d}']
 def eager():return agent.predict(evidence.transform(extract_tail_views(vae,x,a.device)))
 operations={'eager':eager,'on_demand':lambda:predict(x,vae,evidence,agent,a.device),'all_fine':lambda:predict(x,vae,evidence,agent,a.device,force_goal=1)}
 results={};outputs={}
 for name,fn in operations.items():
  with torch.no_grad():
   fn();times=[]
   for _ in range(a.repeats):
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter();o=fn()
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    times.append(time.perf_counter()-start)
  outputs[name]=o;results[name]=dict(test_cases=len(x),full_test_pass_seconds_median=float(np.median(times)),**metric(y,o[0],o[0]>0))
 results['eager_vs_on_demand']=dict(binary_equal=bool(np.array_equal(outputs['eager'][0]>0,outputs['on_demand'][0]>0)),classes_equal=bool(np.array_equal(outputs['eager'][1],outputs['on_demand'][1])),max_score_difference=float(np.max(np.abs(outputs['eager'][0]-outputs['on_demand'][0]))))
 results['protocol']='Standalone full-test-pass timing; not the paper single-case/128-case protocol.'
 dest=work/f'benchmark_{a.dataset}_{a.seed}_{a.rate}.json';dest.write_text(json.dumps(results,indent=2));print(dest)
if __name__=='__main__':main()
