"""Model adapters connecting recurrent transformers to BPD."""

from mechinterp.bpd.adapters.huginn import (
    HugCal02AuditResult,
    HuginnBPDAdapter,
    SettlingCoreLooper,
    discover_huginn_core_linears,
    evaluate_hug_cal02_eight_factors,
)
from mechinterp.bpd.adapters.looped_chase import (
    LoopedChaseBPDAdapter,
    LoopedChaseBatch,
    default_looped_chase_c_per_module,
    load_looped_chase_checkpoint,
    sample_pointer_chase_batch,
)
from mechinterp.bpd.adapters.trm import (
    SudokuHorizonMetrics,
    TRMBPDAdapter,
    TRMBatch,
    build_mini_trm,
    compute_head_projector_f64,
    default_trm_c_per_module,
    encode_sudoku_pairs,
    evaluate_sudoku_horizon,
    load_pretrained_trm,
)

__all__ = [
    "HugCal02AuditResult",
    "HuginnBPDAdapter",
    "LoopedChaseBPDAdapter",
    "LoopedChaseBatch",
    "SettlingCoreLooper",
    "SudokuHorizonMetrics",
    "TRMBPDAdapter",
    "TRMBatch",
    "build_mini_trm",
    "compute_head_projector_f64",
    "default_looped_chase_c_per_module",
    "default_trm_c_per_module",
    "discover_huginn_core_linears",
    "encode_sudoku_pairs",
    "evaluate_hug_cal02_eight_factors",
    "evaluate_sudoku_horizon",
    "load_looped_chase_checkpoint",
    "load_pretrained_trm",
    "sample_pointer_chase_batch",
]
