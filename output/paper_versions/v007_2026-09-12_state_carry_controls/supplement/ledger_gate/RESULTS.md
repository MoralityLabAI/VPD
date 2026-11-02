# Native-thinking comprehension gate at 4,096 tokens

The fixed 11/12 gate stopped after one correct response and two length-censored responses. Nine questions were not run because the gate could no longer be reached. This measures failure to complete the required response under this budget; it does not establish that the model lacked the underlying ledger knowledge.

| Question | Outcome | Tokens | Reasoning status | Visible answer characters |
|---|---|---:|---|---:|
| s0-dev-world-00000-main-DISPATCH | correct | 4041 | complete | 73 |
| s0-dev-world-00000-main-POST | censored | 4096 | partial_unclosed | 0 |
| s0-dev-world-00000-pending-POST | censored | 4096 | partial_unclosed | 0 |

Completed correct and censored responses were retained through runtime resumptions. Partial outputs interrupted by checkpoint I/O were preserved separately and were not counted as model answers. The latest run ended normally at the gate rule, with verified worker cleanup.

The ledger pilot has zero completed episodes. No condition-level behavioral rate, ledger probe accuracy, monitoring effect, SAE response or causal intervention effect can be reported from this protocol. The previously collected four-family mechanistic dataset remains separate.

The prepared answer-only Campsite comparison is the next independent benchmark stage. It uses the completed 27B study’s selected puzzle cohort and procedure settings; it does not replace ledger validation or the broader ARC/math comparison.
