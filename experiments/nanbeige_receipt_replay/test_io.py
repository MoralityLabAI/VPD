"""Atomic checkpoint behavior under Windows file-sharing conflicts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .io import ROOT,save,read_json


class CheckpointTests(unittest.TestCase):
    def test_transient_read_lock_preserves_completed_checkpoint(self):
        with patch.object(Path,'read_text',side_effect=[PermissionError('Replacing'),PermissionError('Replacing'),'{"completed":true,"tokens":[1,2]}']),patch('experiments.nanbeige_receipt_replay.io.time.sleep'):
            self.assertEqual(read_json('checkpoint.json'),dict(completed=True,tokens=[1,2]))

    def test_persistent_read_lock_is_bounded(self):
        with patch.object(Path,'read_text',side_effect=PermissionError('Locked')) as reader,patch('experiments.nanbeige_receipt_replay.io.time.sleep'):
            with self.assertRaises(PermissionError):read_json('checkpoint.json')
            self.assertEqual(reader.call_count,20)

    def test_corrupt_json_is_not_hidden_by_retries(self):
        with patch.object(Path,'read_text',return_value='{') as reader:
            with self.assertRaises(json.JSONDecodeError):read_json('checkpoint.json')
            reader.assert_called_once()

    def test_transient_lock_retries_atomic_replace(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'checkpoint.json'
            save(path,{'tokens':[1]})
            original=Path.replace
            calls=[]
            def replace(source,target):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(path.read_text()),{'tokens':[1]})
                    raise PermissionError('Simulated sharing violation')
                return original(source,target)
            with patch.object(Path,'replace',replace),patch('experiments.nanbeige_receipt_replay.io.time.sleep'):
                save(path,{'tokens':[1,2]})
            self.assertEqual(len(calls),3)
            self.assertEqual(json.loads(path.read_text()),{'tokens':[1,2]})

    def test_persistent_lock_remains_an_error_and_preserves_both_snapshots(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'checkpoint.json'
            save(path,{'tokens':[1]})
            with patch.object(Path,'replace',side_effect=PermissionError('Locked')) as replace,patch('experiments.nanbeige_receipt_replay.io.time.sleep'):
                with self.assertRaises(PermissionError):save(path,{'tokens':[1,2]})
            self.assertEqual(replace.call_count,20)
            self.assertEqual(json.loads(path.read_text()),{'tokens':[1]})
            self.assertEqual(json.loads(path.with_suffix('.json.tmp').read_text()),{'tokens':[1,2]})


if __name__=='__main__':unittest.main()
