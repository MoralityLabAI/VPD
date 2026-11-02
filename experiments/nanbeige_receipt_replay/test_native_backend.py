"""IPC identity and native-reasoning preservation without GPU inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from .native_backend import NativeNanbeigeBackend,save,ROOT,digest


class NativeBackendTests(unittest.TestCase):
    def test_forced_exit_cleans_up_owned_windows_worker_child(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=NativeNanbeigeBackend(folder);backend.process=Mock();backend.process.pid=123
            backend.process.poll.side_effect=[None,0]
            backend.process.wait.side_effect=[subprocess.TimeoutExpired('fixture',10),0]
            child=Mock();child.cmdline.return_value=['python','-m','experiments.nanbeige_receipt_replay.native_worker','--output',str(backend.output)]
            with patch('experiments.nanbeige_receipt_replay.native_backend.psutil.Process') as owner:
                owner.return_value.children.return_value=[child]
                backend.__exit__(None,None,None)
            child.terminate.assert_called_once()
            backend.process.terminate.assert_called_once()

    def fixture(self,folder,wrong_hash=False):
        backend=NativeNanbeigeBackend(folder);backend.render=lambda messages:'<think>\n'
        backend.tokenizer=Mock();backend.tokenizer.encode.return_value=[1]
        raw='Ledger check.</think>{"action":"M"}<|im_end|>'
        backend.tokenizer.decode.return_value=raw;backend.tokenizer.eos_token='<|im_end|>'
        backend.output_decoder=Mock();backend.output_decoder.decode.return_value=raw
        request=Path(folder)/'requests/00001.json'
        save(request,dict(messages=[],metadata={'phase':'act'},seed=1,temperature=.6,max_new_tokens=32,input_ids=[1],rendered_prompt='<think>\n'))
        save(Path(folder)/'results/00001.json',dict(completed=True,output_ids=[7],request_sha256='wrong' if wrong_hash else digest(request),
            raw_decoded_output=raw,finish_reason='stop',elapsed_seconds=.1))
        return backend

    def test_native_reply_preserves_thought_and_exact_tokens(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=self.fixture(folder)
            reply=backend(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)
            self.assertEqual(reply['reasoning_content'],'Ledger check.')
            self.assertEqual(reply['output_ids'],[7])
            self.assertEqual(json.loads(reply['content']),{'action':'M'})
            self.assertTrue(reply['native_hf_decoded_text_match'])

    def test_transient_result_read_lock_does_not_abandon_reply(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=self.fixture(folder);original=Path.read_text;attempts=[]
            def locked(path,*args,**kwargs):
                if path.parent.name=='results':
                    attempts.append(1)
                    if len(attempts)<3:raise PermissionError('Checkpoint being replaced')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',locked),patch('experiments.nanbeige_receipt_replay.io.time.sleep'):
                reply=backend(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)
            self.assertEqual(reply['output_ids'],[7])
            self.assertEqual(backend.calls,1)
            self.assertIsNone(backend.thermal_error)

    def test_stale_worker_reply_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=self.fixture(folder,True)
            with self.assertRaisesRegex(RuntimeError,'different request'):
                backend(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)

    def test_native_cleanup_retains_both_text_forms(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=self.fixture(folder)
            tokenizer_text=backend.output_decoder.decode.return_value.replace('check.','check .')
            backend.output_decoder.decode.return_value=tokenizer_text
            reply=backend(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)
            self.assertEqual(reply['tokenizer_decoded_output'],tokenizer_text)
            self.assertEqual(reply['reasoning_content'],'Ledger check.')
            self.assertFalse(reply['native_hf_decoded_text_match'])
            self.assertEqual(reply['native_tokenizer_text_relation'],'native_space_cleanup')

    def test_non_cleanup_text_difference_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=self.fixture(folder)
            backend.output_decoder.decode.return_value='Different answer'
            with self.assertRaisesRegex(ValueError,'beyond pinned'):
                backend(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)

    def test_answer_only_response_remains_visible(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            backend=self.fixture(folder);backend.enable_thinking=False
            path=Path(folder)/'results/00001.json'
            result=json.loads(path.read_text(encoding='utf-8'))
            result['raw_decoded_output']='{"action":"M"}<|im_end|>'
            save(path,result);backend.output_decoder.decode.return_value=result['raw_decoded_output']
            reply=backend(messages=[],seed=1,metadata={'phase':'act'},temperature=.6,max_new_tokens=32)
            self.assertEqual(reply['content'],'{"action":"M"}')
            self.assertIsNone(reply['reasoning_content'])
            self.assertFalse(reply['generation_context']['enable_thinking'])
            self.assertEqual(reply['reasoning_status'],'not_requested')

    def test_pinned_decoder_preserves_special_token_spacing(self):
        from tokenizers import Tokenizer
        decoder=Tokenizer.from_file(str(ROOT/'models/Nanbeige4.2-3B-tokenizer/tokenizer.json'))
        text='<|im_start|>assistant\n<think>\nLedger.\n</think>\n{"a":1}<|im_end|>'
        ids=decoder.encode(text,add_special_tokens=False).ids
        self.assertEqual(decoder.decode(ids,skip_special_tokens=False),text)


if __name__=='__main__':unittest.main()
