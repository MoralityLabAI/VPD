from __future__ import annotations

import math
from collections.abc import Sequence


def decision_logit_margin(pro_logit: float, request_logit: float) -> float:
    """Return the first-token PRO minus REQUEST logit margin."""
    return float(pro_logit) - float(request_logit)


def candidate_argmax_valid(
    logits: Sequence[float], pro_token_id: int, request_token_id: int
) -> bool:
    """Whether the global next-token argmax is one of the decision candidates."""
    if not logits:
        raise ValueError("logits must be non-empty")
    winner = max(range(len(logits)), key=lambda index: float(logits[index]))
    return winner in {int(pro_token_id), int(request_token_id)}


def token_logprob(logits: Sequence[float], token_id: int) -> float:
    """Stable log-softmax value for one token."""
    if not logits:
        raise ValueError("logits must be non-empty")
    if token_id < 0 or token_id >= len(logits):
        raise IndexError("token_id is outside the logit vector")
    maximum = max(float(value) for value in logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(float(value) - maximum) for value in logits)
    )
    return float(logits[token_id]) - log_normalizer
