"""Prepare a fresh answer-only 27B Campsite comparison with frozen helpers."""
import ast
from collections import Counter
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import types
from transformers import AutoTokenizer
from .fetch_sources import sha
from .build_campsite_support import observation, symmetry_key

ROOT = Path('artifacts/nanbeige_reasoning')
OUT = ROOT/'logic_skill_comparison_v1'


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def public_puzzle(prompt):
    dims = re.search(r'(\d+) x (\d+)', prompt.splitlines()[0])
    assert dims
    n, m = map(int, dims.groups())
    cells = [line.split() for line in prompt.split('row_constraints')[0].splitlines()
             if re.fullmatch(r'\s*[TX](?:\s+[TX])+\s*', line)]
    assert len(cells) == n and all(len(row) == m for row in cells)
    assert all(v in ('T','X') for row in cells for v in row)
    vectors = {name:ast.literal_eval(re.search(r'^'+name+r'\s*=\s*(\[[^\n]+\])',prompt,re.M)[1])
               for name in ('row_constraints','col_constraints')}
    assert len(vectors['row_constraints']) == n and len(vectors['col_constraints']) == m
    return dict(grid=cells,n=n,m=m,**vectors)


def main():
    if (OUT/'protocol.json').exists(): raise ValueError('Completed preparation exists')
    OUT.mkdir(exist_ok=True)
    selected = ROOT/'selected_logic_skill_v1'
    receipt = json.loads((selected/'verification.json').read_text())
    assert sha(selected/'public_inputs_and_routes.jsonl') == receipt['input_routes_sha256']
    assert sha(selected/'skill_prompt.txt') == receipt['prompt_sha256']
    questions = [json.loads(x) for x in (selected/'public_inputs_and_routes.jsonl').read_text(encoding='utf-8').splitlines()]
    support_dir = ROOT/'hermes_synthetic_campsite_v1'
    support_manifest = json.loads((support_dir/'manifest.json').read_text())
    for name,h in support_manifest['files'].items(): assert sha(support_dir/name) == h
    bank = [json.loads(x) for x in (support_dir/'logic_support.jsonl').read_text().splitlines()]
    puzzles = [json.loads(x) for x in (support_dir/'puzzles.jsonl').read_text().splitlines()]
    training = [x for x in puzzles if x['split']=='train']
    assert {x['id'] for x in training} == {x['id'] for x in bank} and len(bank)==128
    keys = {symmetry_key(x['public']['grid'],x['public']['row_constraints'],x['public']['col_constraints']) for x in training}
    source = OUT/'helper_source'; source.mkdir(exist_ok=True)
    archive = Path('D:/research_engine/trmObservabilityLogs/trm_observability_harness_snapshot/harness')
    audit = json.loads((ROOT/'hermes_audit/controller_implementation_audit.json').read_text())
    for name in ('trm_retrieval.py','trm_critic.py','trm_logic_corrector.py'):
        assert sha(archive/name) == audit['sources']['harness/'+name]['sha256']
        shutil.copyfile(archive/name, source/name)
    package = types.ModuleType('selected_hermes_helpers'); package.__path__=[str(source.resolve())]
    sys.modules[package.__name__]=package
    retrieval = importlib.import_module(package.__name__+'.trm_retrieval')
    critic_module = importlib.import_module(package.__name__+'.trm_critic')
    repair = importlib.import_module(package.__name__+'.trm_logic_corrector')
    retriever = retrieval.TRMRetriever.from_rows(bank,min_supervision_weight=.4)
    critic = critic_module.TRMCritic.from_rows(bank,k=5)
    assert len(retriever.examples)==128
    sources = ROOT/'comparator_sources/qwen35_27b'
    tok = AutoTokenizer.from_pretrained(sources/'official',local_files_only=True,use_fast=False)
    skill = (selected/'skill_prompt.txt').read_text(encoding='utf-8')
    calls=[]; cases=[]
    for q in questions:
        public=public_puzzle(q['prompt'])
        assert symmetry_key(public['grid'],public['row_constraints'],public['col_constraints']) not in keys
        obs=observation(public)
        parsed=repair.parse_logic_observation(obs)
        assert parsed and parsed['base_grid']==public['grid']
        assert parsed['row_constraints']==public['row_constraints'] and parsed['col_constraints']==public['col_constraints']
        case=dict(id=q['id'],public=public,helper_observation=obs,route=q['route'],pilot_split=q['pilot_split'])
        variants={'vanilla':q['prompt'],'logic_skill':skill+'\n'+q['prompt']}
        if q['route']=='logic_skill_trm':
            query=dict(task_family='logic',task='campsite',observation=obs,
                       support_signature=q['signature'],response_mode='action_only',skill_name='intellect3-logic-hermes')
            assert not {'target_action','solution','expected_action'} & query.keys()
            action, exemplar, similarity = retriever.predict(query)
            crit=critic.predict(query)
            workspace=("\n\nTRM_WORKSPACE:\n"+f"EXEMPLAR={action}\n"
                       +f"CRITIC_BUCKET={crit['predicted_bucket']}\n"
                       +f"CRITIC_SCORE={crit['predicted_bucket_score']}\n"
                       +f"VISIBLE_OUTPUT_PRED={crit.get('predicted_visible_output_emitted')}\n\n"
                       +'Use the TRM workspace as advisory evidence only. Do not copy it blindly. '
                       +'Return only the final committed Python-style 2D grid literal.')
            variants['logic_skill_trm']=variants['logic_skill']+workspace
            case['workspace']=dict(retrieved_action=action,retrieved_row_id=exemplar.row_id,
                similarity=similarity,critic_prediction=crit,query=query)
        for arm,content in variants.items():
            messages=[dict(role='user',content=content)]
            rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==rendered
            assert rendered.endswith('<think>\n\n</think>\n\n')
            calls.append(dict(call_id=f"{q['id']}::{arm}",id=q['id'],arm=arm,
                              rendered_prompt=rendered,token_ids=ids))
        cases.append(case)
    assert len(calls)==259 and len(cases)==109
    # Interleave arms per question; the fixed order is independent of outcomes.
    (OUT/'calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in calls),encoding='utf-8')
    write(OUT/'cases.json',cases)
    shutil.copyfile(support_dir/'logic_support.jsonl',OUT/'synthetic_support.jsonl')
    config=dict(status='prepared',puzzles=109,unique_model_calls=259,
        reported_arms=['vanilla','logic_skill','routed_candidate','routed_final','plain_candidate_routed_repair'],
        model='Qwen3.5-27B',model_lock_sha256=sha(sources/'model_lock.json'),
        calls_sha256=sha(OUT/'calls.jsonl'),cases_sha256=sha(OUT/'cases.json'),
        selected_skill_receipt_sha256=sha(selected/'verification.json'),
        helper_sources={f.name:sha(f) for f in source.glob('*.py')},
        support_sha256=sha(OUT/'synthetic_support.jsonl'),support_train_rows=128,
        exact_and_dihedral_support_overlap=0,reference_answers_used_in_queries=False,
        skill_revision=receipt['revision'],max_new_tokens=1024,context=8192,thinking_enabled=False,
        seed=42,temperature=0,repeat_penalty=1.,weights='Q4_K_M',kv_cache='q8_0',
        tokenizer_assets=json.loads((ROOT/'qwen_comparator_pilot_inputs_v1/manifest.json').read_text())['tokenizer_assets'],
        max_prompt_tokens=max(len(c['token_ids']) for c in calls),
        route_counts=dict(Counter(q['route'] for q in questions)),
        repair=dict(max_passes=8,exact_search=True,exact_node_limit=200000),
        limitations=['Fresh integration of selected skill and hash-pinned archived helper implementation, not historical replay.',
            'Synthetic support replaces exposed historical support; all 128 support labels are positive, so critic discrimination is not tested.',
            'Frozen signature gate was selected on historical outcomes; this is not a clean generalization estimate.',
            'Answer-only Qwen generation with a 1024-token cap differs from Nanbeige thinking-mode runs.',
            'TRM auxiliary retrieval/critic/search paths are not neural recurrent transformer passes.'],
        preparer_sha256=sha(Path(__file__)))
    assert config['max_prompt_tokens']+1024<=config['context']
    write(OUT/'protocol.json',config)
    print(json.dumps({k:config[k] for k in ('puzzles','unique_model_calls','max_prompt_tokens','route_counts')}))


if __name__=='__main__':main()
