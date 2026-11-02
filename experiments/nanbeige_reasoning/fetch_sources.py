"""Download public, revision-pinned inputs; never load executable dataset code."""
from __future__ import annotations
import hashlib
import json
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('artifacts/nanbeige_reasoning/sources')
REVISIONS = {
    'arc_agi_1': ('fchollet/ARC-AGI', '399030444e0ab0cc8b4e199870fb20b863846f34'),
    'arc_agi_2': ('arcprize/ARC-AGI-2', 'f3283f727488ad98fe575ea6a5ac981e4a188e49'),
    'hermes_skills': ('MoralityLabAI/Hermes-Skills', '07a177f2751c2adb1cc16dc819c1173c0cd57ad5'),
}
HF_REV = 'a9eb9183224cec63fe9a71e7a3a80cff14868624'
ENV_REV = '48cb24baecee7dc76bb55be86bf4a181ffcb60e8'

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'Nanbeige-Reasoning-Research/1.0'}), timeout=180)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix(target.suffix + '.part')
        with request(url) as r, part.open('wb') as f:
            for chunk in iter(lambda: r.read(1024*1024), b''):
                f.write(chunk)
        part.replace(target)
    return {'url': url, 'path': target.as_posix(), 'bytes': target.stat().st_size,
            'sha256': sha(target)}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = {'schema_version': 1, 'sources': {}}
    for name, (repo, rev) in REVISIONS.items():
        if name == 'hermes_skills':
            # The matching local git checkout is separately audited, not downloaded wholesale.
            lock['sources'][name] = {'repository': repo, 'revision': rev}
            continue
        archive = ROOT / (name + '.tar.gz')
        receipt = download(f'https://codeload.github.com/{repo}/tar.gz/{rev}', archive)
        files = []
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                rel = Path(*Path(member.name).parts[1:])
                if not member.isfile() or '..' in rel.parts or rel.is_absolute():
                    continue
                if not (rel.as_posix().startswith(('data/training/', 'data/evaluation/'))
                        and rel.suffix == '.json') and rel.name not in ('LICENSE', 'README.md'):
                    continue
                if member.size > 5_000_000:
                    raise ValueError('Unexpected large ARC source file')
                target = ROOT / name / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                files.append({'path': rel.as_posix(), 'sha256': sha(target)})
        lock['sources'][name] = dict(repository=repo, revision=rev, archive=receipt, files=files)
        print(name, len(files), 'files', flush=True)
    for subset in ('logic', 'math'):
        url = f'https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-RL/resolve/{HF_REV}/{subset}/train-00000-of-00001.parquet'
        receipt = download(url, ROOT / 'intellect3' / (subset + '.parquet'))
        lock['sources']['intellect3_' + subset] = dict(revision=HF_REV, subset=subset,
            dataset='PrimeIntellect/INTELLECT-3-RL', split='train', file=receipt)
        print(subset, receipt['bytes'], 'bytes', flush=True)
    repo='PrimeIntellect-ai/research-environments'
    with request(f'https://api.github.com/repos/{repo}/git/trees/{ENV_REV}?recursive=1') as response:
        tree=json.load(response)
    if tree.get('truncated'):
        raise ValueError('Incomplete official verifier source inventory')
    paths=[x['path'] for x in tree['tree'] if x['type']=='blob' and (
        x['path'].startswith(('environments/reasoning/i3_logic/','environments/math/i3_math/'))
        or x['path']=='LICENSE')]
    def fetch_environment(path):
        return download(f'https://raw.githubusercontent.com/{repo}/{ENV_REV}/{path}',
            ROOT/'research_environments'/path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts=list(pool.map(fetch_environment,paths))
    lock['sources']['research_environments']=dict(repository=repo,revision=ENV_REV,files=receipts)
    (ROOT.parent / 'sources.lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
