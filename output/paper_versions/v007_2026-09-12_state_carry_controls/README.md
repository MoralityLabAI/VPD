# Version 7: state carry and intervention fidelity

The 18-page manuscript adds Section 7 and Appendix E, with revised abstract and framing. Earlier numbered manuscripts are unchanged.

The new CPU experiment completed 450 conditions on ten complete held-out prompts from eight task groups: one pass, two passes carrying state, and two executed passes with every token reset at the boundary. All 150 reset/one-pass pairs agree exactly in saved states and logits. Five focused analysis tests passed; 1,680 primary metric values were independently checked.

Reconstruction failures persist without state carry. Freezing the SAE's baseline support removes a large selection-induced error spike. State carry nevertheless changes propagated responses: all six tested layer-12 writes have positive Muon layer-22 change fidelity under reset and negative fidelity under carry. This is a descriptive result for this cohort, dictionary and intervention direction, not a universal recurrence penalty or a separately trained architecture comparison.

The supplement contains analysis and both feature-encoding streams, independent checks, experiment protocol, records, native-archive manifest, figures and source snapshots. Native activation archives remain in artifacts/nanbeige_reasoning/recurrence_state_control_v1. Source hashes link the manuscript to those files. The existing inference and pass-specific intervention setup is reused; the new residual writes are distinguished from the earlier fixed-coefficient VPD weight edits.

No ledger episode completed, and no concealed chain-of-thought or semantic mediation result is claimed. The full benchmark remains unfinished. No GPU inference or paid cloud allocation was used for this control; its native model is closed.
