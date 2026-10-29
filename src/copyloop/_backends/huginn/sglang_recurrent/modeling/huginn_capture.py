import os
from pathlib import Path
import torch
COUNTS={}
def capture(role,**values):
 p=os.environ.get('HUGINN_CAPTURE_DIR')
 if not p:return
 p=Path(p);p.mkdir(exist_ok=True,parents=True)
 i=COUNTS.get(role,0);COUNTS[role]=i+1
 torch.save({k:v.detach().cpu() for k,v in values.items()},p/f'{role}_{i:04d}.pt')
