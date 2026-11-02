"""Verify completed prompt captures independently of the capture process."""
import json
from pathlib import Path
import numpy as np
from .fetch_sources import sha


def main():
    root=Path('artifacts/nanbeige_reasoning/local_prompt_capture_v2')
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='prompt_capture_complete'
    records=manifest['records']
    assert len(records)==8 and len({(r['lane'],r['split']) for r in records})==8
    arrays=0
    sites={f'model.layers.{i}' for i in range(22)}|{'model.norm'}
    for name,expected in manifest['files'].items():
        assert sha(root/name)==expected,(name,'hash mismatch')
    for record in records:
        assert set(record['calls'])==sites and set(record['calls'].values())=={2}
        assert sha(root/record['archive'])==record['sha256']
        with np.load(root/record['archive'],allow_pickle=False) as archive:
            positions=archive['token_positions']
            assert len(positions)==32 and np.all(np.diff(positions)>0)
            assert positions[0]>=0 and positions[-1]<len(archive['token_ids'])
            for site in sites:
                for invocation in (0,1):
                    value=archive[f'{site}__pass_{invocation}']
                    assert value.shape==(32,3072) and np.isfinite(value).all()
                    arrays+=1
    analysis=json.loads((root/'frozen_sae_transfer/manifest.json').read_text())
    assert analysis['capture_run_complete'] and analysis['examples']==8
    for name,expected in analysis['files'].items():
        assert sha(root/'frozen_sae_transfer'/name)==expected
    report=dict(status='verified',examples=8,sites=23,passes=2,activation_arrays=arrays,
        retained_token_positions=8*32,prompt_tokens=sum(r['prompt_tokens'] for r in records),
        forward_seconds=sum(r['seconds'] for r in records),
        manifest_sha256=sha(root/'manifest.json'),analysis_manifest_sha256=sha(root/'frozen_sae_transfer/manifest.json'))
    (root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
