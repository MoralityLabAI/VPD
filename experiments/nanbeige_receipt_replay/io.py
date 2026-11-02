"""Small dependency-free artifact helpers for inference and capture processes."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]


def read_json(path):
    """Read an atomic checkpoint despite brief Windows replacement/share locks."""
    path=Path(path)
    for attempt in range(20):
        try:return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.01*(attempt+1),.1))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    # Windows readers can briefly deny replacement while holding the old file.
    # Keep the old checkpoint intact and retry the atomic rename; never truncate it.
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.01*(attempt+1),.1))
