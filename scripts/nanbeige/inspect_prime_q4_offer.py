"""Read advertised image compatibility using the existing authenticated Prime client."""
import json
import argparse
from pathlib import Path
from prime_cli.client import APIClient

parser=argparse.ArgumentParser()
parser.add_argument('--gpu',default='RTX6000Ada_48GB')
parser.add_argument('--provider',default='massedcompute')
parser.add_argument('--output',type=Path,default=Path('/mnt/c/projects/VPD/artifacts/nanbeige_reasoning/prime_q4_offer_details.json'))
args=parser.parse_args()
data=APIClient().get('/availability/gpus',params={'gpu_type':args.gpu,'gpu_count':1,'page_size':100})
rows=[r for r in data['items'] if r.get('provider')==args.provider]
path=args.output
path.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
