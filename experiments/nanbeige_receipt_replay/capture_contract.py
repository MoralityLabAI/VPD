"""Validate captured rows before fitting probes or assessing frozen dictionaries."""
import json
from pathlib import Path

import numpy as np

from .io import digest


def load_captures(folder):
    """Reject ambiguous identities and misaligned states rather than pooling them."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'decision_captures_complete':
        raise ValueError('Capture run is not complete')
    if digest(folder / 'records.json') != manifest['files']['records.json']:
        raise ValueError('Capture records hash mismatch')
    records = json.loads((folder / 'records.json').read_text(encoding='utf-8'))
    if not records:
        raise ValueError('No captured episodes')
    identities = set()
    worlds = {}
    templates = {}
    arrays = {}
    hidden_width = None
    required = {f'model.layers.{layer}__pass_{p}' for layer in range(22) for p in range(2)}
    for record in records:
        episode = record['episode_id']
        if not isinstance(episode, str) or not episode or episode in identities:
            raise ValueError('Episode IDs must be nonempty and unique')
        identities.add(episode)
        split = record['split']
        if split not in ('dev', 'holdout'):
            raise ValueError(f'Unknown split: {split}')
        world, template = record['world_id'], record['template_id']
        if world in worlds and worlds[world] != (split, template):
            raise ValueError('World spans splits or template identities')
        if template in templates and templates[template] != split:
            raise ValueError('Template leaks across development and holdout')
        worlds[world] = (split, template)
        templates[template] = split
        for target in ('receipt_already_credited', 'will_attempt_duplicate_credit'):
            label = record['labels'][target]
            if label is not None and type(label) is not bool:
                raise ValueError(f'{target} must be boolean or unknown')
        archive = (folder / record['archive']).resolve()
        if not archive.is_relative_to(folder):
            raise ValueError('Capture archive escapes its directory')
        if digest(archive) != record['sha256']:
            raise ValueError('Capture archive hash mismatch')
        with np.load(archive, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        if not required.issubset(data):
            raise ValueError('Capture lacks one or more of the 44 layer/pass arrays')
        tokens, positions = data['token_ids'], data['token_positions']
        if (tokens.ndim != 1 or positions.ndim != 1 or not len(positions)
                or tokens.dtype.kind not in 'iu' or positions.dtype.kind not in 'iu'
                or not np.array_equal(positions, np.unique(positions))
                or positions[0] < 0 or positions[-1] >= len(tokens)):
            raise ValueError('Invalid token IDs or captured token positions')
        for key, state in data.items():
            if '__pass_' not in key:
                continue
            if state.ndim != 2 or state.shape[0] != len(positions) or not np.isfinite(state).all():
                raise ValueError(f'Misaligned or nonfinite states: {key}')
            if key in required:
                hidden_width = state.shape[1] if hidden_width is None else hidden_width
                if not hidden_width or state.shape[1] != hidden_width:
                    raise ValueError('Physical layer hidden widths differ')
        anchor_names = set()
        for anchor in record['anchors']:
            name, row = anchor['anchor'], anchor['array_row']
            if name not in ('prompt_end', 'pre_action') or name in anchor_names:
                raise ValueError('Invalid or repeated anchor name')
            anchor_names.add(name)
            if type(row) is not int or not 0 <= row < len(positions):
                raise ValueError('Anchor array row is out of bounds')
            pos = int(positions[row])
            if anchor['token_index'] != pos or anchor['token_id'] != int(tokens[pos]):
                raise ValueError('Anchor token does not match captured state')
        if 'prompt_end' not in anchor_names:
            raise ValueError('Capture lacks the primary prompt-end anchor')
        arrays[episode] = {key: value for key, value in data.items() if '__pass_' in key}
    return manifest, records, arrays
