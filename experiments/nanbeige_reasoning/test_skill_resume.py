"""Synthetic provenance and no-resampling checks; no model inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.nanbeige_receipt_replay.io import ROOT,save,digest
from .skill_resume import SETTINGS,load_reusable_run


class ResumeTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);self.prep=base/'inputs';self.source=base/'run'
        self.prep.mkdir();self.source.mkdir()
        self.cfg=dict(model='Nanbeige4.2-3B',max_new_tokens=2,context=8192,thinking_enabled=False,
                      seed=42,temperature=0,repeat_penalty=1.,kv_cache='q8_0',weights='Q4_K_M',
                      unique_model_calls=2,tokenizer_assets={},tokenizer_root='.')
        self.calls=[dict(call_id=str(i),id=str(i),arm='vanilla',token_ids=[10+i],
                         rendered_prompt='prompt'+str(i),messages=[dict(role='user',content=str(i))]) for i in range(2)]
        (self.prep/'calls.jsonl').write_text('\n'.join(map(json.dumps,self.calls))+'\n',encoding='utf-8')
        save(self.prep/'cases.json',[])
        self.cfg.update(calls_sha256=digest(self.prep/'calls.jsonl'),cases_sha256=digest(self.prep/'cases.json'))
        save(self.prep/'protocol.json',self.cfg)
        save(self.source/'contract.json',dict(model=self.cfg['model'],expected_records=2,
             protocol_sha256=digest(self.prep/'protocol.json'),calls_sha256=self.cfg['calls_sha256'],
             generation_settings={k:self.cfg[k] for k in SETTINGS}))
        runtime=self.source/'runtime'
        save(runtime/'runtime_contract.json',dict(model_sha256='fixture',loops=2,enable_thinking=False,
             context=8192,quantization='Q4_K_M',kv_cache='q8_0'))
        for name in ('worker_stopped.json','server_stopped.json'):save(runtime/name,dict(stopped=True))
        for i,call in enumerate(self.calls):
            request=dict(input_ids=call['token_ids'],messages=call['messages'],metadata=dict(call_id=call['call_id']),
                         seed=42,temperature=0,max_new_tokens=2)
            request_path=runtime/'requests'/f'{i+1:05d}.json';save(request_path,request)
            path=runtime/'results'/request_path.name
            tokens=[20+i,30+i];finish='stop' if i==0 else 'length';text='wrong answer' if i==0 else 'unfinished answer'
            save(path,dict(completed=True,output_ids=tokens,raw_decoded_output=text,finish_reason=finish,
                           request_sha256=digest(request_path)))
            response=dict(input_ids=call['token_ids'],rendered_prompt=call['rendered_prompt'],
                 output_ids=tokens,raw_decoded_output=text,finish_reason=finish,call_record=str(path.relative_to(ROOT)))
            save(self.source/f'{i:03d}.json',dict(call_id=call['call_id'],id=call['id'],arm=call['arm'],
                 contract_sha256=digest(self.source/'contract.json'),response=response,generated_token_ids=tokens,
                 generated_text=text,native_text=text,generated_tokens=2,max_token_censored=finish=='length'))
        save(self.source/'cleanup.json',dict(server_stopped=True))
        self.manifest()

    def manifest(self):
        save(self.source/'manifest.json',dict(status='interrupted',records=2,files={
             str(p.relative_to(self.source)):digest(p) for p in self.source.rglob('*') if p.is_file() and p.name!='manifest.json'}))

    def load(self):return load_reusable_run(self.source,self.prep,self.calls,self.cfg,'fixture')

    def test_wrong_and_capped_answers_are_both_retained(self):
        result=self.load()
        self.assertEqual(result[0]['row']['generated_text'],'wrong answer')
        self.assertTrue(result[1]['row']['max_token_censored'])

    def test_completed_unrecorded_answer_cannot_be_resampled(self):
        save(self.source/'runtime/results/00003.json',dict(completed=True))
        self.manifest()
        with self.assertRaisesRegex(ValueError,'decode-only recovery'):self.load()

    def test_both_decoder_forms_are_preserved(self):
        path=self.source/'000.json';row=json.loads(path.read_text(encoding='utf-8'))
        native_path=self.source/'runtime/results/00001.json';native=json.loads(native_path.read_text(encoding='utf-8'))
        native['raw_decoded_output']='wrong answer.';save(native_path,native)
        row['native_text']='wrong answer.';row['generated_text']='wrong answer .'
        row['response'].update(raw_decoded_output='wrong answer.',tokenizer_decoded_output='wrong answer .')
        save(path,row);self.manifest()
        result=self.load()[0]['row']
        self.assertEqual(result['generated_text'],'wrong answer .')
        self.assertEqual(result['native_text'],'wrong answer.')

    def test_tampered_response_rejected(self):
        save(self.source/'000.json',{})
        with self.assertRaisesRegex(ValueError,'hash'):self.load()

    def test_live_worker_rejected(self):
        save(self.source/'runtime/worker_stopped.json',dict(stopped=False));self.manifest()
        with self.assertRaisesRegex(ValueError,'remains live'):self.load()

    def test_complete_reuse_starts_no_backend_and_preserves_source(self):
        from . import run_nanbeige_skill_comparison as runner
        output=Path(self.temp.name)/'continued';before=digest(self.source/'manifest.json')
        argv=['runner','--inputs',str(self.prep),'--output',str(output),'--reuse-run',str(self.source)]
        with patch.object(runner,'Q4_SHA256','fixture'),patch.object(runner,'NativeNanbeigeBackend') as backend,patch('sys.argv',argv):
            runner.main();backend.assert_not_called()
        result=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual((result['records'],result['reused_records'],result['new_model_responses']),(2,2,0))
        self.assertEqual(digest(self.source/'manifest.json'),before)
        self.assertEqual(json.loads((output/'001.json').read_text(encoding='utf-8'))['generated_text'],'unfinished answer')

    def test_only_missing_call_is_generated(self):
        from . import run_nanbeige_skill_comparison as runner
        call=dict(call_id='2',id='2',arm='vanilla',token_ids=[12],rendered_prompt='prompt2',messages=[dict(role='user',content='2')])
        self.calls.append(call);self.cfg['unique_model_calls']=3
        (self.prep/'calls.jsonl').write_text('\n'.join(map(json.dumps,self.calls))+'\n',encoding='utf-8')
        self.cfg['calls_sha256']=digest(self.prep/'calls.jsonl');save(self.prep/'protocol.json',self.cfg)
        path=self.source/'contract.json';contract=json.loads(path.read_text(encoding='utf-8'))
        contract.update(expected_records=3,calls_sha256=self.cfg['calls_sha256'],protocol_sha256=digest(self.prep/'protocol.json'))
        save(path,contract)
        for i in range(2):
            path=self.source/f'{i:03d}.json';row=json.loads(path.read_text(encoding='utf-8'))
            row['contract_sha256']=digest(self.source/'contract.json');save(path,row)
        self.manifest();output=Path(self.temp.name)/'continued'
        argv=['runner','--inputs',str(self.prep),'--output',str(output),'--reuse-run',str(self.source)]
        with patch.object(runner,'Q4_SHA256','fixture'),patch.object(runner,'NativeNanbeigeBackend') as factory,patch('sys.argv',argv):
            backend=factory.return_value.__enter__.return_value
            backend.render.side_effect=lambda messages:'prompt'+messages[0]['content']
            backend.tokenizer.encode.side_effect=lambda prompt,**kwargs:[10+int(prompt[-1])]
            backend.return_value=dict(input_ids=[12],output_ids=[99],raw_decoded_output='new answer',tokenizer_decoded_output='new answer',finish_reason='stop',elapsed_seconds=1)
            factory.return_value.__exit__.side_effect=lambda *args:save(output/'runtime/server_stopped.json',dict(stopped=True))
            runner.main()
            backend.assert_called_once()
            self.assertEqual(backend.call_args.kwargs['metadata']['call_id'],'2')
        result=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual((result['records'],result['reused_records'],result['new_model_responses']),(3,2,1))


if __name__=='__main__':unittest.main()
