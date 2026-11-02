import json
import tempfile
import unittest
from pathlib import Path

from experiments.nanbeige_reasoning.run_q4_feature_answers import read_record,save_record


class AnswerCheckpointTests(unittest.TestCase):
    def test_saved_prefix_corruption_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'answer.json'
            row=dict(id='q',generated=[10,20],complete=False,mutation_count=24)
            save_record(path,row)
            self.assertEqual(read_record(path)['generated'],[10,20])
            altered=json.loads(path.read_text());altered['generated'][1]=21
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'checkpoint hash'):read_record(path)

    def test_advancing_checkpoint_rehashes_mutable_token_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'answer.json'
            row=dict(generated=[10],complete=False)
            save_record(path,row);old_hash=row['payload_sha256']
            row['generated'].append(20);row['complete']=True
            save_record(path,row)
            self.assertNotEqual(old_hash,row['payload_sha256'])
            self.assertEqual(read_record(path)['generated'],[10,20])
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__=='__main__':unittest.main()
