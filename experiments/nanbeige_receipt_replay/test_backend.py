"""Mock transport tests for logging and native-reasoning preservation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from .backend import NanbeigeBackend,ROOT


class BackendTests(unittest.TestCase):
    def test_relative_output_and_raw_reasoning(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            b=NanbeigeBackend(Path(d).relative_to(ROOT))
            b.render=lambda messages:'<think>\n'
            b.tokenizer=Mock();b.tokenizer.encode.return_value=[1]
            b.tokenizer.decode.return_value='Consider the ledger.</think>{"action":"M"}<|im_end|>'
            b.tokenizer.eos_token='<|im_end|>'
            with patch('experiments.nanbeige_receipt_replay.backend.request',side_effect=[{'tokens':[1]},{'tokens':[5,6], 'stop_type':'eos'}]):
                reply=b(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)
            self.assertEqual(reply['reasoning_content'],'Consider the ledger.')
            self.assertEqual(json.loads(reply['content']),{'action':'M'})
            self.assertEqual(reply['output_ids'],[5,6])
            saved=json.loads((Path(d)/'calls/00001.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['server_response']['tokens'],[5,6])

    def test_decoding_failure_keeps_raw_tokens(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            b=NanbeigeBackend(d);b.render=lambda messages:'<think>\n'
            b.tokenizer=Mock();b.tokenizer.encode.return_value=[1];b.tokenizer.decode.side_effect=ValueError('fixture decoder failure')
            with patch('experiments.nanbeige_receipt_replay.backend.request',side_effect=[{'tokens':[1]},{'tokens':[9], 'stop_type':'eos'}]):
                with self.assertRaises(ValueError):b(messages=[],seed=1,metadata={},temperature=.6,max_new_tokens=32)
            saved=json.loads((Path(d)/'calls/00001.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['server_response']['tokens'],[9])
            self.assertFalse(saved['completed'])


if __name__=='__main__':unittest.main()
