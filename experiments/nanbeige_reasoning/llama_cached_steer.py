"""Incremental, bounded-context residual interventions; pure ctypes/NumPy."""
import numpy as np
from .llama_steer import SteeringModel


class CachedSteeringModel(SteeringModel):
    def __init__(self, *args, context=4096, **kwargs):
        self.context_capacity = context
        self.next_position = 0
        self.cache_valid = False
        self.total_mutations = 0
        super().__init__(*args, context=context, **kwargs)

    def reset_cache(self):
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.context), True)
        self.next_position = 0
        self.cache_valid = True
        self.total_mutations = 0
        self.clear_steering()

    def evaluate(self, tokens, *, intervention=None, chunk_size=64):
        """Append tokens, applying the same direction at every processed position.

        intervention is a dictionary accepted by set_steering except positions.
        Each call returns only its final next-token logits. No silent truncation,
        cache shifting or sampling is permitted.
        """
        if not self.cache_valid:
            raise RuntimeError('Call reset_cache before using a new or failed cache')
        if not tokens or not 1 <= chunk_size <= 64:
            raise ValueError('Nonempty tokens and chunk size 1..64 required')
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
            raise ValueError('Invalid vocabulary token')
        if self.next_position + len(tokens) > self.context_capacity:
            raise ValueError('Context capacity exceeded; no automatic shifting')
        self.enabled = False
        self.selected = set()
        self.arrays = {}
        self.errors = []
        batch = self.lib.llama_batch_init(chunk_size, 0, 1)
        try:
            for offset in range(0, len(tokens), chunk_size):
                chunk = tokens[offset:offset+chunk_size]
                start = self.next_position
                self.chunk_positions = list(range(start, start+len(chunk)))
                self.mutations = []
                if intervention is None:
                    self.clear_steering()
                else:
                    self.set_steering(**intervention, positions=self.chunk_positions)
                batch.n_tokens = len(chunk)
                for i, token in enumerate(chunk):
                    batch.token[i] = token
                    batch.pos[i] = start+i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = 1
                result = self.lib.llama_decode(self.context, batch)
                if result or self.errors:
                    raise RuntimeError(f'Decode {result}; callbacks: {self.errors}')
                expected = len(chunk)*len(intervention['passes']) if intervention else 0
                if len(self.mutations) != expected:
                    raise RuntimeError(f'Missing invocation writes: {len(self.mutations)}/{expected}')
                self.total_mutations += expected
                self.next_position += len(chunk)
            pointer = self.lib.llama_get_logits_ith(self.context, len(chunk)-1)
            logits = np.ctypeslib.as_array(pointer, shape=(self.vocab_size,)).copy()
            if not np.isfinite(logits).all():
                raise RuntimeError('Nonfinite logits')
            return logits
        except BaseException:
            self.cache_valid = False
            raise
        finally:
            self.lib.llama_batch_free(batch)

    def replay(self, *args, **kwargs):
        # The inherited API clears/replaces KV memory, invalidating append state.
        self.cache_valid = False
        return super().replay(*args, **kwargs)
