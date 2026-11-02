"""Synthetic regression fixtures; these are never behavioral model evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .capture_contract import load_captures
from .io import ROOT, digest, save


class CaptureContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'captures'
        self.folder.mkdir()
        self.records = []
        for world in range(4):
            for case, (variant, credited, choice) in enumerate([
                    ('main', True, True), ('main', True, False),
                    ('pending', False, False), ('pending', False, False)]):
                episode = f'synthetic-w{world}-c{case}'
                # Two independently decodable variables, including legitimate POST.
                state = np.array([[2 * credited - 1, 2 * choice - 1, .1 * world, .1 * case]] * 2,
                                 dtype=np.float32)
                data = {f'model.layers.{layer}__pass_{p}': state
                        for layer in range(22) for p in range(2)}
                data.update(token_ids=np.array([10, 11, 12]), token_positions=np.array([0, 2]))
                path = self.folder / (episode + '.npz')
                np.savez_compressed(path, **data)
                self.records.append(dict(
                    episode_id=episode, world_id=f'world{world}', template_id=f'template{world}',
                    split='dev' if world < 2 else 'holdout', variant=variant,
                    chosen_action='POST' if choice or variant == 'pending' else 'CLOSE',
                    labels=dict(receipt_already_credited=credited, will_attempt_duplicate_credit=choice),
                    anchors=[dict(anchor='prompt_end', array_row=0, token_index=0, token_id=10),
                             dict(anchor='pre_action', array_row=1, token_index=2, token_id=12)],
                    archive=path.name, sha256=digest(path)))
        self.seal()

    def seal(self):
        save(self.folder / 'records.json', self.records)
        save(self.folder / 'manifest.json', dict(status='decision_captures_complete',
             files={'records.json': digest(self.folder / 'records.json')}))

    def change_array(self, modify):
        record = self.records[0]
        path = self.folder / record['archive']
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        modify(data)
        np.savez_compressed(path, **data)
        record['sha256'] = digest(path)
        self.seal()

    def test_valid_archive_keeps_all_physical_passes(self):
        _, records, states = load_captures(self.folder)
        self.assertEqual(len(records), 16)
        self.assertTrue(all(len(value) == 44 for value in states.values()))

    def test_duplicate_episode_cannot_overwrite_states(self):
        self.records.append(copy.deepcopy(self.records[0]))
        self.seal()
        with self.assertRaisesRegex(ValueError, 'unique'):
            load_captures(self.folder)

    def test_template_leakage_is_rejected(self):
        self.records[8]['template_id'] = self.records[0]['template_id']
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Template leaks'):
            load_captures(self.folder)

    def test_unknown_split_is_not_silently_dropped(self):
        self.records[0]['split'] = 'test'
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Unknown split'):
            load_captures(self.folder)

    def test_nonbinary_labels_are_rejected(self):
        self.records[0]['labels']['will_attempt_duplicate_credit'] = 'false'
        self.seal()
        with self.assertRaisesRegex(ValueError, 'boolean'):
            load_captures(self.folder)

    def test_anchor_cannot_read_another_tokens_state(self):
        self.records[0]['anchors'][1]['array_row'] = 0
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Anchor token'):
            load_captures(self.folder)

    def test_missing_pass_is_rejected(self):
        self.change_array(lambda data: data.pop('model.layers.21__pass_1'))
        with self.assertRaisesRegex(ValueError, '44 layer/pass'):
            load_captures(self.folder)

    def test_nonfinite_state_is_rejected(self):
        self.change_array(lambda data: data['model.layers.21__pass_1'].__setitem__((0, 0), np.nan))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            load_captures(self.folder)

    def test_modified_archive_requires_a_matching_hash(self):
        path = self.folder / self.records[0]['archive']
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_captures(self.folder)

    def test_no_behavior_variation_yields_missing_cells_not_scores(self):
        from . import probes
        for record in self.records:
            record['labels']['will_attempt_duplicate_credit'] = False
            record['chosen_action'] = 'POST' if record['variant'] == 'pending' else 'CLOSE'
        self.seal()
        output = Path(self.temp.name) / 'missing_probes'
        with patch.object(sys, 'argv', ['probes', '--captures', str(self.folder),
                                       '--output', str(output), '--permutations', '1']):
            probes.main()
        rows = json.loads((output / 'heatmap_data.json').read_text(encoding='utf-8'))
        behavior = [row for row in rows if row['target'] != 'receipt_already_credited']
        self.assertEqual(len(behavior), 176)
        self.assertTrue(all(row['status'] == 'insufficient_class_variation' and row['auroc'] is None
                            and row['balanced_accuracy'] is None for row in behavior))
        candidates = json.loads((output / 'intervention_candidates.json').read_text(encoding='utf-8'))
        self.assertEqual(candidates['status'], 'no_qualified_behavior_candidate')
        self.assertEqual(candidates['candidates'], [])

    def test_probe_command_writes_all_cells_and_hashed_models(self):
        from . import probes
        output = Path(self.temp.name) / 'probes'
        with patch.object(sys, 'argv', ['probes', '--captures', str(self.folder),
                                       '--output', str(output), '--permutations', '3']):
            probes.main()
        rows = json.loads((output / 'heatmap_data.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 22 * 2 * 2 * 3)
        for row in rows:
            self.assertEqual(row['status'], 'evaluated')
            self.assertEqual(row['auroc'], 1.)
            self.assertEqual(row['balanced_accuracy'], 1.)
            self.assertEqual(row['permutation_baseline']['repeats'], 3)
            self.assertEqual(row['confidence_interval']['status'], 'not_reported_fewer_than_four_test_worlds')
        contract = json.loads((output / 'contract.json').read_text(encoding='utf-8'))
        self.assertFalse(contract['semantic_localization_claim'])
        self.assertFalse(set(contract['train_templates']) & set(contract['test_templates']))
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['evaluated'], 264)
        for name, sha256 in manifest['files'].items():
            self.assertEqual(digest(output / name), sha256)


if __name__ == '__main__':
    unittest.main()
