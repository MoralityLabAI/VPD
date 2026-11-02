"""Create a new manuscript revision; preserve every indexed prior artifact."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
VERSION='v006_2026-09-12_experiment_closeout'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')


def main():
    archive=ROOT/'output/paper_versions';index=read(archive/'index.json')
    old=archive/'v005_2026-09-11_cross_family_dynamics';out=archive/VERSION
    if out.exists():raise FileExistsError('Numbered revisions are never overwritten')
    previous={}
    for version in index['versions']:
        for name,entry in version['files'].items():
            p=archive/version['version']/name
            assert sha(p)==entry['sha256'],p
            previous[str(p.relative_to(ROOT))]=sha(p)
    data=ROOT/'artifacts/nanbeige_reasoning'
    grade=data/'nanbeige_skill_partial_grading_v1'
    comparison=data/'matched_skill_partial_closeout_v1'
    gate=ROOT/'artifacts/nanbeige_receipt_replay/capability_gate_4096_v1'
    for folder in (grade,comparison,gate):
        for name,digest in read(folder/'verification.json')['files'].items():assert sha(folder/name)==digest
    g=read(grade/'summary.json');pairs=read(comparison/'same_procedure_pairs.json');ledger=read(gate/'summary.json')
    assert (g['unique_model_calls'],g['total_capped_calls'],g['unrun_calls'])==(45,11,214)
    assert (ledger['model_responses'],ledger['correct'],ledger['censored'],ledger['not_run'],ledger['ledger_pilot_episodes'])==(3,1,2,9,0)
    assert g['candidate_native_grading_disagreements']==0
    out.mkdir()
    for directory in ('equations','figures','supplement'):shutil.copytree(old/directory,out/directory)
    for name,folder in [('partial_grading',grade),('partial_comparison',comparison),('ledger_gate',gate)]:
        shutil.copytree(folder,out/'supplement'/name)
    recovery=data/'cooling_timeout_recovery_v1/verification.json'
    shutil.copy2(recovery,out/'supplement/runtime_closeout.json')
    text=(old/'PAPER.md').read_text(encoding='utf-8').replace('Version 5 | 11 September 2026','Version 6 | 12 September 2026')
    start=text.index('The receipt-replay environment provides a prospective way')
    end=text.index('\n\nThe emerging account of recurrence',start)
    text=text[:start]+'''The receipt-replay environment makes this semantic requirement concrete. Whether a shipment receipt has already been credited is a fact about the world; whether the model requests duplicate credit is a later choice. An uncredited receipt makes the same submission action legitimate. This contrast prevents a probe that merely recognizes the submission command from standing in for an explanation of its meaning. An independent recount and a free correction opportunity further separate an initial request from knowingly retaining an inflated ledger. Pressure and announced-auditor conditions would then vary the context in which the relation is used.

That experiment first needs an agent that can express the environment's basic distinctions within its generation budget. The completed comprehension gate did not establish that prerequisite. Of twelve assigned questions, one produced a correct final answer and two exhausted the 4,096-token budget inside unclosed thinking, leaving no visible answer. The fixed requirement of eleven correct answers could no longer be met, so the remaining nine questions were not run. No ledger episode has completed. This is a budget-limited protocol failure, not evidence that the model lacks the underlying distinction, and not a result about concealment under monitoring.

The boundary matters to the interpretation of internal evidence. If a task cannot reliably elicit the behavior whose meaning is at issue, a decodable input fact cannot fill that evidential gap. The implemented analysis separates the already-credited fact from the subsequent duplicate-credit choice, holds out world templates, and preserves both invocations of every physical layer. These controls specify how a future semantic claim could be tested; no ledger probe or intervention effect is measured here. Appendix D records the completed gate and the unexecuted stages.''' + text[end:]
    old_scope='This revision adds task-grouped analysis of existing all-layer captures and a loop-normalization comparison. It retains the intervention-change analysis introduced in version 4, including its float64 re-encoding check. No new model inference, dictionary training, immediate-write experiment, matched-write intervention or semantic restoration experiment was added. Version 2 preserves the detailed technical record; versions 3 and 4 preserve the earlier mechanistic manuscripts.'
    new_scope='This revision closes the associated experiments at their observed endpoints. It retains version 5\'s all-layer geometry and version 4\'s intervention-change analysis, including float64 re-encoding. It adds grading of 45 saved Nanbeige comparator calls, exact-puzzle comparisons against the completed Qwen study, and the terminal ledger comprehension gate. The closeout itself uses CPU grading and document work without further model inference. No new dictionary fit, immediate-write control, matched-write intervention or semantic restoration result is added. Earlier numbered manuscripts remain preserved.'
    assert old_scope in text;text=text.replace(old_scope,new_scope)
    start=text.index('Only one question identifier overlaps the 18-input Nanbeige cohort')
    text=text[:start]+'''A separate Nanbeige answer-only comparator was then prepared from the exact Qwen user-message contents, with the same puzzles, routes, helper source and repair limits. Its native template adds the publisher's default system message. Both comparator protocols request thinking disabled and cap generation at 1,024 tokens; the native-thinking activation trajectories remain a separate dataset. Token units, parameter counts, architecture, quantization implementation and hardware differ, so this is neither equal compute nor an isolated test of recurrence.

The Nanbeige run ended after 45 of 259 scheduled calls when its cooling budget was exhausted. All 45 completed outputs are retained, including 11 capped responses; 214 scheduled calls have no completed output. The 45 are an exact prefix of the planned order, including seven outputs reused without resampling from the preceding interrupted run. The next call saved zero output tokens. We close this study as a partial cohort, without projecting its results onto all 109 puzzles. Its missingness follows the schedule and runtime interruption and is not established to be random.

The following table uses each procedure's exact intersection of completed puzzle IDs in the two models. Qwen counts here are recomputed from its archived grades on that same intersection, rather than compared against its full 109-puzzle denominator. All available Nanbeige candidates are graded by the existing published verifier; capped output is retained and scored as emitted. Native-text and tokenizer-text candidate grades agree on all 45 calls.

'''
    names={'vanilla':'Vanilla generation','logic_skill':'Plain logic skill','routed_candidate':'Routed candidate','routed_final':'Routed final with repair','plain_candidate_routed_repair':'Plain candidate with routed repair'}
    table=['| Procedure | Matched puzzles | Qwen correct | Nanbeige correct | Capped: Qwen / Nanbeige |','|---|---:|---:|---:|---:|']
    for r in pairs:
        table.append(f"| {names[r['procedure']]} | {r['puzzles']} | {r['both_correct']+r['qwen_only_correct']} | {r['both_correct']+r['nanbeige_only_correct']} | {r['qwen_capped']} / {r['nanbeige_capped']} |")
    text+='\n'.join(table)+'''

Each row is a paired model comparison. Denominators can differ between procedures because a final puzzle may have only some candidate calls completed. The supplement retains exact IDs, paired outcome counts and within-Nanbeige procedure intersections. Repaired answers are generated from the candidate and public constraints before the grading references are loaded, using the frozen helper. Several procedures reuse the same candidate; they are not independent experiments, and their token totals must not be summed as total compute.

Correctness here means published-verifier acceptance. The earlier contract audit found that the prompt's literal requirement for exactly one adjacent tree differs from the verifier's one-to-one matching rule. All six accepted Nanbeige routed-final answers match the supplied canonical targets, but none passes the stricter literal check. The supplement preserves these diagnostics separately; the accepted count must not be read as satisfaction of every statement in the prompt. The verifier also retains the existing bounded-parsing adapter described in the technical companion.

These descriptive results concern an exposed prefix of one public logic family under a short answer-only protocol. The historically selected route and positive-only synthetic support remain limitations. We do not supply a population confidence interval or infer an architectural advantage. The comparison separates candidate generation from external repair at a behavioral endpoint; it does not connect those outcomes to the saved VPD or SAE states. There is no completed matched ARC or math comparison and no verified claim that the selected skills clear all four families.

## Appendix D. Ledger gate and experimental endpoint

The warehouse evaluation was designed to connect an input relation, its internal accessibility and a consequential action. Four matched conditions combine ordinary or score/appointment-pressure instructions with the presence or absence of an announced auditor. A legitimate uncredited-receipt control uses the same submission action. World seeds determine receipt IDs, quantities, names, action order and menu symbols. The structured record retains prompts, environment turns, exposed native thinking, visible output, parsed action and authoritative state after every action.

The behavioral fields remain distinct: duplicate credit attempted; duplicate credit accepted; inflated ledger retained after the independent recount; final physical count overstated; duplicate request denied in the final report; final task success; and available reasoning/report evidence of the scoring bug. None is replaced by a single deception label. In particular, a false count accepted by the environment is different from retaining it after an explicit correction opportunity.

The native-thinking comprehension gate required eleven correct answers out of twelve, with a 4,096-token cap per question. It terminated once two non-passing responses made that threshold unattainable:

| Gate outcome | Count | Interpretation |
|---|---:|---|
| Correct final answer | 1 | Completed in 4,041 generated tokens |
| Length-censored thinking | 2 | 4,096 tokens each; no visible final answer |
| Assigned questions not run | 9 | Excluded from observed response accuracy |
| Completed ledger episodes | 0 | No action or monitoring-pressure outcomes |

The nine unrun questions are not incorrect observations, and the two censored responses are not substituted with answers inferred from unfinished thinking. These are hypothetical mechanics questions, not agentic warehouse episodes. The result establishes that the selected native-thinking protocol did not pass its execution gate; it does not estimate general task competence. There is consequently no condition-wise behavioral table, ledger activation cohort, held-out probe score, permutation baseline or ledger-distribution SAE reconstruction result to report.

The analysis implementation is prepared for simple linear probes of receipt-already-credited and subsequent duplicate-credit choice, splitting by world template rather than token rows. The primary anchor is the end of the input; a separately labelled secondary anchor occurs after generated thinking and before the action token. Their distinction prevents later generated information from being silently treated as input-state evidence. Capture validation checks physical layer, pass, token identity and context; the two-pass model requires 44 layer-invocation arrays. No probe accuracy is interpreted as semantic localization.

The planned intervention stage reuses the fitted VPD component and its first-pass, second-pass and both-pass masks. It would measure probe-score and behavioral changes alongside actual hidden-state changes, then re-encode both complete states through compatible frozen SAE dictionaries. Baseline reconstruction and intervention-change fidelity would be checked on the ledger distribution before interpreting feature absence. None of those stages was reached. The intervention-change results in the main text remain measurements on the earlier ARC and INTELLECT-3 replay states.

The study ends with a complete account of the evidence available for this revision: observed pass geometry, a paired dictionary-optimizer control, measured parameter-change reconstruction, a graded partial external-procedure comparison, and a failed ledger entry gate. The missing semantic experiment remains a named limitation. The methodological contribution is the requirement that an explanation follow a shared operation through its invocation-specific effect, its measured feature response and an independently established consequence, retaining reconstruction error at each step.

The revision's supplement includes partial_grading (all observed grades, missing call IDs and corrections), partial_comparison (exact matched IDs and paired counts), ledger_gate (protocol, three responses and terminal summary), and runtime_closeout.json (validated prefix and cleanup receipt). The earlier geometry, intervention and Qwen-comparator supplements are retained unchanged. The thermal endpoint is documented for reproducibility; it supplies no evidence about the model's reasoning process.
'''
    (out/'PAPER.md').write_text(text,encoding='utf-8')
    renderer=(old/'render_paper.py').read_text(encoding='utf-8').replace('Version 5 | Invocation geometry and intervention fidelity | 11 September 2026','Version 6 | Shared parameter effects and experiment closeout | 12 September 2026')
    (out/'render_paper.py').write_text(renderer,encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'closeout_verification.json',dict(status='source_numbers_verified_render_pending',
        previous_index_sha256=sha(archive/'index.json'),preserved_previous_files=previous,
        matched_table_rows=table[2:],ledger_summary_sha256=sha(gate/'summary.json'),
        partial_grading_sha256=sha(grade/'verification.json'),partial_comparison_sha256=sha(comparison/'verification.json'),
        new_inference_calls=0,full_benchmark_completed=False,ledger_episodes=0))
    print(json.dumps(dict(version=VERSION,previous_files_verified=len(previous),matched_rows=table[2:])))


if __name__=='__main__':main()
