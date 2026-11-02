"""Read only connection/status fields from both Prime API response endpoints."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from prime_cli.core import APIClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pod_id')
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if not re.fullmatch('[a-f0-9]{32}', args.pod_id):
        raise ValueError('Expected an observed Prime pod ID')
    client = APIClient()
    details = client.get('/pods/' + args.pod_id)
    statuses = client.get('/pods/status', params={'pod_ids': [args.pod_id]})
    allowed = ('id', 'podId', 'status', 'ip', 'sshConnection', 'installationStatus',
               'installationProgress', 'installationFailure', 'primePortMapping')
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), pod_id=args.pod_id,
        details={key:details.get(key) for key in allowed},
        statuses=[{key:row.get(key) for key in allowed} for row in statuses.get('data', [])],
        detail_field_names=sorted(details))
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
