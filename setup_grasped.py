"""Fetch the optional GRASPED dependency at the recorded upstream revision."""
from pathlib import Path
import urllib.request,hashlib
root=Path(__file__).resolve().parent
url='https://raw.githubusercontent.com/guanwei49/GRASPED/14fb81df161a952067e13154c4148af87b969b14/model/GRU_AE.py'
p=root/'vendor/guanwei49__GRASPED/model/GRU_AE.py'
if p.exists():raise SystemExit('Dependency already exists; refusing to overwrite.')
data=urllib.request.urlopen(url,timeout=60).read()
compile(data,str(p),'exec');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print('Downloaded',p,'SHA256',hashlib.sha256(data).hexdigest())
