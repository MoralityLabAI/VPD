"""Validate completed native benchmark replies before reusing an interrupted run."""
import json
from pathlib import Path

from experiments.nanbeige_receipt_replay.io import ROOT,digest
from experiments.nanbeige_receipt_replay.native_text import verify_native_text


SETTINGS=('max_new_tokens','context','thinking_enabled','seed','temperature','repeat_penalty','kv_cache','weights')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def load_reusable_run(source,preparation,calls,cfg,model_sha256):
    if source is None:return {}
    source=source.resolve();manifest=read(source/'manifest.json');contract=read(source/'contract.json')
    if manifest['status'] not in ('interrupted','logic_skill_generation_complete'):
        raise ValueError('Source run has no terminal manifest')
    if not read(source/'cleanup.json')['server_stopped']:
        raise ValueError('Source worker cleanup is unverified')
    if (source/'runtime').exists():
        for name in ('worker_stopped.json','server_stopped.json'):
            if not read(source/'runtime'/name)['stopped']:
                raise ValueError('Native worker remains live')
    for name,checksum in manifest['files'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or digest(path)!=checksum:
            raise ValueError('Source manifest hash or path differs')
    required={'contract.json','cleanup.json'}|{f'{i:03d}.json' for i in range(manifest['records'])}
    if not required.issubset(manifest['files']):raise ValueError('Source manifest omits required files')
    if (contract['protocol_sha256']!=digest(preparation/'protocol.json')
            or contract['calls_sha256']!=digest(preparation/'calls.jsonl')
            or contract['generation_settings']!={k:cfg[k] for k in SETTINGS}
            or contract['model']!=cfg['model'] or contract['expected_records']!=len(calls)):
        raise ValueError('Source inference protocol differs')
    paths=sorted(source.glob('[0-9][0-9][0-9].json'))
    if [p.name for p in paths]!=[f'{i:03d}.json' for i in range(manifest['records'])]:
        raise ValueError('Completed calls must be a unique uninterrupted prefix')
    if len(paths)>len(calls):raise ValueError('Too many source calls')
    reused={};represented=set()
    for index,path in enumerate(paths):
        row=read(path);call=calls[index];reply=row['response']
        if any(row[k]!=call[k] for k in ('call_id','id','arm')):
            raise ValueError('Completed call identity differs')
        if row['contract_sha256']!=digest(source/'contract.json'):
            raise ValueError('Completed call contract differs')
        if reply['input_ids']!=call['token_ids'] or reply['rendered_prompt']!=call['rendered_prompt']:
            raise ValueError('Completed call prompt differs')
        native=(ROOT/reply['call_record']).resolve()
        if not native.is_relative_to(ROOT):raise ValueError('Native response path escapes workspace')
        runtime=native.parent.parent
        for name in ('worker_stopped.json','server_stopped.json'):
            if not read(runtime/name)['stopped']:raise ValueError('Native worker remains live')
        rc=read(runtime/'runtime_contract.json')
        if (rc['model_sha256']!=model_sha256 or rc['loops']!=2 or rc['enable_thinking'] is not False
                or rc['context']!=cfg['context'] or rc['quantization']!=cfg['weights'] or rc['kv_cache']!=cfg['kv_cache']):
            raise ValueError('Native model or execution settings differ')
        result=read(native);request_path=runtime/'requests'/native.name;request=read(request_path)
        if not result['completed'] or result.get('error') or result['request_sha256']!=digest(request_path):
            raise ValueError('Native response is incomplete or mismatched')
        if (request['input_ids']!=call['token_ids'] or request['messages']!=call['messages']
                or request['metadata']['call_id']!=call['call_id']
                or any(request[k]!=cfg[k] for k in ('seed','temperature','max_new_tokens'))):
            raise ValueError('Native request differs')
        ids=result['output_ids'];text=result['raw_decoded_output'];finish=result['finish_reason']
        if (finish not in ('stop','length') or reply['output_ids']!=ids or row['generated_token_ids']!=ids
                or reply['raw_decoded_output']!=text or row['native_text']!=text
                or reply['finish_reason']!=finish or row['max_token_censored']!=(finish=='length')
                or row['generated_tokens']!=len(ids) or not 0<len(ids)<=cfg['max_new_tokens']):
            raise ValueError('Completed native tokens or censoring differ')
        verify_native_text(row['generated_text'],text)
        if row['generated_text']!=reply.get('tokenizer_decoded_output',reply['raw_decoded_output']):
            raise ValueError('Tokenizer-decoded response text differs')
        represented.add(native)
        reused[index]=dict(row=row,source=str(path),sha256=digest(path),native_sha256=digest(native))
    # A completed reply lost between worker and adapter must be recovered, never resampled.
    for native in (source/'runtime/results').glob('*.json'):
        if read(native).get('completed') and native.resolve() not in represented:
            raise ValueError('Unrecorded completed native reply requires decode-only recovery')
    return reused
