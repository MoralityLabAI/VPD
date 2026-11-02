"""Integrate only verified completed Campsite results into the methods draft."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--grading-name',default='grading');args=p.parse_args()
    run=args.run;grading=run/args.grading_name
    verify=json.loads((grading/'verification.json').read_text())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    for name,h in verify['files'].items():assert sha(grading/name)==h
    s=json.loads((grading/'summary.json').read_text())
    assert s['status']=='fresh_logic_skill_comparison_graded' and s['unique_model_calls']==259
    t=s['totals'];n=lambda arm:int(t[arm]['official_correct'])
    source=Path('artifacts/nanbeige_reasoning/PAPER.md')
    text=source.read_text(encoding='utf-8')
    assert '### 4.18 ' not in text
    shutil.copyfile(source,run/'PAPER_before_logic_comparison.md')
    for suffix in ('.pdf','.receipt.json'):
        f=Path('output/pdf/nanbeige_hermes_research_draft'+suffix)
        if f.exists():shutil.copyfile(f,run/('paper_before_logic_comparison'+suffix))
    section=['### 4.18 Fresh model-assisted Campsite comparison','',
        'A fresh Qwen3.5-27B Q4_K_M run evaluates the same 109 public Campsite puzzles across '
        'a vanilla prompt, the selected `intellect3-logic-hermes` prompt, and a routed TRM workspace. '
        'The skill and its latest prompt imprint are frozen at Hermes-Skills revision `07a177f2751c2adb1cc16dc819c1173c0cd57ad5`. '
        'The packaged gate routes 41 constraint signatures to the augmented path and leaves 68 on the plain skill. '
        'There are 259 unique model calls: 109 vanilla, 109 plain skill and 41 augmented calls. '
        'Unrouted candidates are reused exactly across routed arms.','',
        'All calls use the publisher template with thinking disabled, temperature zero, seed 42, '
        'a 1,024-token output cap, 8,192-token context and Q8 KV cache. The longest prompt is 792 tokens. '
        'The pinned llama.cpp runtime verifies full CUDA offload and GGUF/Hugging Face tokenizer agreement for all 259 inputs. '
        'Native detokenization and independently decoded token IDs are both retained. '
        f'The run generates {s["actual_generated_tokens"]:,} tokens, with {s["total_capped_calls"]} capped calls.','',
        'The auxiliary implementation is the previously audited, hash-pinned retrieval/nearest-neighbor critic/constraint-search path. '
        'It uses the existing 128 synthetic training puzzles instead of the exposed historical bank; '
        'none matches an evaluation puzzle under the checked rotations or reflections. '
        'The positive-only support labels make this a test of the integrated workspace, not a test of critic discrimination. '
        'The external corrector retains an eight-pass local-search limit, exact search enabled and a 200,000-node cap. '
        'These are controller search settings, not measured neural transformer invocations. '
        'No evaluation target enters a retrieval query or correction call.','',
        '| Arm | Official correct / 109 | Canonical exact | Literal rule valid | Capped candidates |',
        '|---|---:|---:|---:|---:|']
    labels=dict(vanilla='Vanilla',logic_skill='Plain skill',routed_candidate='Routed model candidate',
                routed_final='Routed final with correction',plain_candidate_routed_repair='Plain candidate with routed correction')
    for arm,label in labels.items():
        row=t[arm];section.append(f'| {label} | {n(arm)} | {row["canonical_exact"]} | {row["literal_valid"]} | {row["capped_candidates"]} |')
    repair=next(x for x in s['paired'] if x['a']=='routed_final' and x['b']=='routed_candidate')
    section+=['',f'Within the routed pipeline, correction fixes {repair["a_only"]} previously incorrect candidates '
        f'and regresses {repair["b_only"]} previously correct candidates under the official verifier. '
        'The plain-candidate correction control separates the effect of commit-stage search from the additional retrieval workspace. '
        'All corrections were produced before loading grading sidecars. The model candidates remain immutable.','',
        'Official verifier success and the stricter literal adjacency rule are reported separately because of the '
        'documented dataset-contract discrepancy in Section 4.3. '
        f'Candidate grades disagree between raw-token and native text views in {s["candidate_native_grading_disagreements"]} '
        'unique model calls. Arm-level token totals share calls and must not be added as independent compute.','',
        'This is a fresh integration, not a historical reproduction. The signature policy was selected using historical outcomes '
        'on this public cohort, so its result is not a clean generalization estimate. Synthetic support changes the helper distribution. '
        'The matched comparisons here are within Qwen on identical puzzles and decoding budgets. Nanbeige used different thinking and output budgets; '
        'this experiment does not isolate model scale, recurrence, or an architectural advantage. '
        'Disabling the template thinking channel does not prevent visible explanations. Vanilla explanations can exhaust the fixed cap, '
        'while the skill explicitly requests a grid only. Consequently, differences can reflect formatting and budget use as well as problem solving; '
        'a separate generic grid-only instruction control would be needed to isolate the specialized skill content. '
        'It also does not evaluate ARC-AGI or the other INTELLECT logic families. '
        'The completed source, routing, token, helper and grading receipts are retained in `logic_skill_comparison_v1` '
        f'and `{run.name}/{args.grading_name}`.','']
    text=text.replace('## 5. Prospective tests and limits','\n'.join(section)+'\n## 5. Prospective tests and limits')
    text=text.replace('Working methods draft, 8 September 2026.','Working methods draft, 9 September 2026.')
    text=text.replace('Tables report completed measurements only; broader performance comparisons remain uncompleted.',
                      'Tables report completed measurements, including a fresh model-assisted Campsite control; broader benchmark comparisons remain uncompleted.')
    text=text.replace('Audited Hermes controls provide context rather than a matched scale comparison.',
        f'A fresh 109-puzzle Qwen3.5-27B Campsite control scores {n("vanilla")}/109 with the vanilla prompt, '
        f'{n("logic_skill")}/109 with the plain skill and {n("routed_final")}/109 after routed correction. '
        'The frozen gate, synthetic support and different decoding budgets limit cross-model interpretation.')
    text=text.replace('The weights have not been downloaded or evaluated.',
        'The weights are now evaluated on the 109-puzzle Campsite comparison in Section 4.18; the full prepared benchmark remains unrun.')
    text=text.replace('These controls have not yet been exercised on the 27B weights; no fresh 27B inference or TRM comparison score is reported.',
        'The separate 109-puzzle answer-only experiment in Section 4.18 exercises weight/runtime identity, full GPU offload and per-input tokenizer checks. '
        'The original 18-record and 1,223-record thinking-mode runs and resume-prefix checks remain unexecuted.')
    text=text.replace('they do not reproduce historical training or complete the model-assisted comparison.',
        'they do not reproduce historical training. Section 4.18 now adds a fresh model-assisted Campsite comparison using the synthetic bank.')
    text=text.replace('A separate comparison could run the same questions through a pinned Hermes/TRM system with an auditable retrieval corpus and full tool-call costs.',
        'Beyond the completed Campsite control, a broader comparison could run the same ARC and INTELLECT questions through applicable pinned helpers with an auditable retrieval corpus and full tool-call costs.')
    source.write_text(text,encoding='utf-8')
    (run/'paper_integration.json').write_text(json.dumps(dict(status='completed_logic_results_integrated_source',
        source_sha256=sha(source),grading_verification_sha256=sha(grading/'verification.json'),
        pdf_render_and_visual_review_pending=True),indent=2)+'\n')
    print('Completed Campsite results integrated; PDF rendering and visual review remain.')


if __name__=='__main__':main()
