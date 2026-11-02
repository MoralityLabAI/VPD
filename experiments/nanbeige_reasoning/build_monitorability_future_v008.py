"""Add prospective monitorability work without altering sealed experiment reports."""
import json
from pathlib import Path
import shutil
from .build_closeout_v006 import ROOT, read, sha, save

VERSION = 'v008_2026-09-12_monitorability_future_work'
SECTION = '''## 9. Future work: internal evidence when reasoning becomes less informative

The longer-term question is whether internal measurements can recover behaviorally relevant information when generated reasoning ceases to expose it. The present study supplies measurement controls for that question, not a demonstrated replacement for chain-of-thought monitoring. State carry changes the responses an interpreter must describe; sparse reconstruction can miss those responses or introduce its own selection effects. An internal monitor must therefore be validated against independently observed behavior, with reconstruction error retained as part of the evidence. Greater access to activations alone does not establish greater access to the model's reasons.

A follow-up would first establish the loss of textual evidence itself. The receipt-replay environment provides matched ordinary-task and score-pressure conditions, each with and without an announced auditor, alongside the legitimate uncredited-receipt control. The study would ask whether auditor exposure reduces explicit reasoning or report evidence of duplicate credit while the relevant behavior persists. Behavior and disclosure should be measured jointly: reduced discussion can accompany less exploitation, shorter answers or changed reporting conventions. A textual change alone would not demonstrate concealment. Duplicate attempts, acceptance, retention after recount, inaccurate reporting and task success would remain separate outcomes. A revised execution protocol would need to pass the comprehension gate before collecting these episodes, with token budgets, censoring rules and stopping criteria fixed before the main study.

The internal analysis would distinguish an input-state fact, whether the receipt was already credited, from a subsequent decision, whether duplicate credit will be attempted. Simple regularized linear probes would test local accessibility across physical layer and pass. Input-end captures would be separated from captures after generated reasoning and immediately before the action, preventing later information from leaking into an earlier decision claim. Evaluation would hold out world templates and vary receipt identities, quantities, names and action encodings. Permutation baselines, sample counts and uncertainty estimated over world groups would accompany discrimination and calibration measures. Comparisons with prompt-only and text-only predictors would test whether internal states add information beyond task wording, action identity or available reasoning. Probe accuracy alone would not establish a semantic location or causal mediator.

For a reproducible internal signal, the next step would reuse the fitted VPD interventions on the first pass, second pass and both passes, measuring probe-score changes alongside actual duplicate-credit choices. Identity controls, irrelevant directions and measured write magnitudes would help distinguish a specific relation from nonspecific disruption. Frozen SAE dictionaries would provide a candidate feature account only after checking reconstruction on the ledger distribution and fidelity to the intervention-induced change. Full-state effects, feature effects and unexplained residual changes would remain separate; fixed-support analyses would test whether apparent feature transitions arise from the encoder's selection rule. Restoration of a proposed intermediate relation and its predicted behavioral consequence would provide stronger evidence than decodability alone.

The practical success criterion is an improvement in held-out behavioral detection over the available text monitor at a matched false-positive rate, with uncertainty reported, specifically where textual evidence has weakened. Such a result would support compensating for a measured loss of monitorability within that setting; it would not establish a general replacement for reasoning traces. Failure to recover a signal would likewise need to be distinguished from insufficient data, poor measurement or an inappropriate capture position. This behavioral and causal study was outside the computational budget of the present revision. No ledger episode completed, so recovery of lost chain-of-thought monitorability remains a future-work hypothesis.

'''


def main():
    archive = ROOT / 'output/paper_versions'
    old = archive / 'v007_2026-09-12_state_carry_controls'
    out = archive / VERSION
    if out.exists():
        raise FileExistsError('Preserve numbered versions')
    index = read(archive / 'index.json')
    prior = {}
    for entry in index['versions']:
        for name, item in entry['files'].items():
            p = archive / entry['version'] / name
            assert sha(p) == item['sha256'], p
            prior[str(p.relative_to(ROOT))] = item['sha256']
    source = (old / 'PAPER.md').read_text(encoding='utf-8')
    assert source.count('## Appendix A.') == 1
    revised = source.replace('Version 7 | 12 September 2026', 'Version 8 | 12 September 2026')
    revised = revised.replace('## Appendix A.', SECTION + '## Appendix A.', 1)
    assert revised.replace(SECTION, '').replace('Version 8 | 12 September 2026', 'Version 7 | 12 September 2026') == source
    out.mkdir()
    for folder in ('equations', 'figures', 'supplement'):
        shutil.copytree(old / folder, out / folder)
    (out / 'PAPER.md').write_text(revised, encoding='utf-8')
    renderer = (old / 'render_paper.py').read_text(encoding='utf-8').replace(
        'Version 7 | State carry and intervention fidelity | 12 September 2026',
        'Version 8 | Monitorability future work | 12 September 2026')
    (out / 'render_paper.py').write_text(renderer, encoding='utf-8')
    shutil.copy2(__file__, out / Path(__file__).name)
    save(out / 'content_verification.json', dict(status='source_verified_render_pending',
        previous_index_sha256=sha(archive / 'index.json'), preserved_previous_files=prior,
        source_version=old.name, source_sha256=sha(old / 'PAPER.md'),
        only_content_addition='Section 9: prospective internal-monitorability study',
        existing_manuscript_text_preserved=True, new_experiments=0,
        ledger_episodes=0, monitorability_recovery_demonstrated=False))
    print(json.dumps(dict(version=VERSION, preserved_files=len(prior), added_words=len(SECTION.split()))))


if __name__ == '__main__':
    main()
