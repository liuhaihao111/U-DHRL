"""Generate a synthetic log and exercise the standalone pipeline on CPU."""
import argparse,json,random,subprocess,sys
import xml.etree.ElementTree as ET
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use a new directory')
 a.output.mkdir(parents=True);rng=random.Random(42);root=ET.Element('log')
 for i in range(160):
  trace=ET.SubElement(root,'trace');ET.SubElement(trace,'string',key='concept:name',value=str(i))
  for j in range(rng.randrange(4,18)):
   e=ET.SubElement(trace,'event');ET.SubElement(e,'string',key='concept:name',value=rng.choice('ABCDE'))
   ET.SubElement(e,'string',key='lifecycle:transition',value=rng.choice(['start','complete']))
   ET.SubElement(e,'date',key='time:timestamp',value=f'2020-01-01T00:00:{j:02d}Z')
 raw=a.output/'synthetic.xes';ET.ElementTree(root).write(raw,encoding='utf-8',xml_declaration=True)
 mapping=a.output/'logs.json';mapping.write_text(json.dumps({'BPIC13_C':str(raw.resolve())}))
 (a.output/'SYNTHETIC_ONLY.txt').write_text('Synthetic functional test. BPIC13_C is used solely as an accepted configuration key; these are not BPIC data or paper results.')
 subprocess.run([sys.executable,str(Path(__file__).with_name('pipeline.py')),'--logs',str(mapping),'--output',str(a.output/'run'),'--seeds','42','--models','U-DHRL','Naive','--device','cpu','--smoke'],check=True)
 assert (a.output/'run'/'RUN_COMPLETE.json').is_file()
 print('Synthetic end-to-end smoke test passed.')
if __name__=='__main__':main()
