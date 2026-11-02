"""Verify decode-only recovery retains correct and incorrect completed answers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .backend import Q4_SHA256
from .capability_check import build_checks
from .io import ROOT,digest,save
from .recover_native_capability import main
from .native_text import clean_native_spaces


class RecoveryTests(unittest.TestCase):
    def test_all_completed_answers_retained_without_starting_worker(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            source=Path(folder)/'source';output=Path(folder)/'recovered';runtime=source/'runtime'
            checks=build_checks()
            save(source/'protocol.json',dict(checks=checks,transport='native',max_new_tokens=4096,temperature=.6))
            save(runtime/'server_stopped.json',dict(stopped=True));save(runtime/'worker_stopped.json',dict(stopped=True))
            save(runtime/'runtime_contract.json',dict(context=8192,model_sha256=Q4_SHA256,loops=2,enable_thinking=True))
            tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Nanbeige4.2-3B-tokenizer',use_fast=False,local_files_only=True)
            decoder=Tokenizer.from_file(str(ROOT/'models/Nanbeige4.2-3B-tokenizer/tokenizer.json'))
            for index,check in enumerate(checks[:2]):
                rendered=tokenizer.apply_chat_template(check['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True,preserve_thinking=True)
                request=dict(messages=check['messages'],metadata=dict(case_id=check['id'],phase='capability'),seed=709+index,
                    temperature=.6,max_new_tokens=4096,input_ids=tokenizer.encode(rendered,add_special_tokens=False),rendered_prompt=rendered)
                name=f'{index+1:05d}.json';save(runtime/'requests'/name,request)
                answer=dict(check['expected_DO_NOT_SEND'])
                if index:answer['physical_after']+=1
                tokenizer_raw='Check .</think>'+json.dumps(answer)+'<|im_end|>'
                raw=clean_native_spaces(tokenizer_raw)
                ids=decoder.encode(tokenizer_raw,add_special_tokens=False).ids
                save(runtime/'results'/name,dict(completed=True,request_sha256=digest(runtime/'requests'/name),
                    raw_decoded_output=raw,output_ids=ids,finish_reason='stop',elapsed_seconds=1))
            save(runtime/'results/00003.json',dict(completed=False,output_ids=[1]))
            with patch.object(sys,'argv',['recover','--source',str(source),'--output',str(output)]),patch('subprocess.Popen') as launch:
                main()
            launch.assert_not_called()
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['model_responses'],2);self.assertEqual(summary['correct'],1)
            self.assertEqual(summary['new_model_calls'],0);self.assertFalse(summary['gate_passed'])
            self.assertFalse((output/'question_02.json').exists())
            for index in range(2):
                record=json.loads((output/f'question_{index:02d}.json').read_text())
                original=json.loads((runtime/'results'/f'{index+1:05d}.json').read_text())
                self.assertEqual(record['response']['output_ids'],original['output_ids'])


if __name__=='__main__':unittest.main()
