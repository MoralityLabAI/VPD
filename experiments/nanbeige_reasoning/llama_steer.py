"""Ephemeral pass-specific residual steering through the pinned ggml C API.

Pure NumPy/ctypes: do not import Torch into the llama.cpp callback process.
Writes affect computed activations, never model weights or checkpoint files.
"""
import ctypes as C
import math
import re
import numpy as np

from .llama_capture import CaptureModel, P, I


class SteeringModel(CaptureModel):
    def __init__(self, *args, **kwargs):
        self.steering = None
        self.mutations = []
        super().__init__(*args, **kwargs)
        setter = self.runtime.base.ggml_backend_tensor_set
        setter.argtypes = [P, P, C.c_size_t, C.c_size_t]
        setter.restype = None

    def set_steering(self, direction, *, alpha, passes, positions, physical_layer=22, identity_write=False):
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (self.width,) or not np.isfinite(direction).all():
            raise ValueError('Finite width-matched direction required')
        if not math.isfinite(alpha) or not 1 <= physical_layer <= 22:
            raise ValueError('Invalid dose or physical layer')
        passes, positions = frozenset(passes), frozenset(positions)
        if not passes or any(type(i) is not int or not 0 <= i < self.loops for i in passes):
            raise ValueError('Invalid invocation scope')
        if not positions or any(type(i) is not int or i < 0 for i in positions):
            raise ValueError('Invalid token positions')
        self.steering = dict(direction=direction.copy(), alpha=float(alpha), passes=passes,
            positions=positions, physical_layer=physical_layer, identity_write=bool(identity_write))

    def clear_steering(self):
        self.steering = None

    def _callback(self, tensor, ask, user):
        try:
            config = self.steering
            name = self.runtime.base.ggml_get_name(tensor).decode()
            match = re.fullmatch(r'l_out-(\d+)', name)
            targeted = bool(config and match and int(match[1]) % 22 == config['physical_layer']-1
                            and int(match[1]) // 22 in config['passes'])
            indices = [i for i, pos in enumerate(self.chunk_positions) if targeted and pos in config['positions']]
            if ask:
                return bool(indices) or super()._callback(tensor, ask, user)
            if indices:
                if C.cast(tensor, C.POINTER(I))[0] != 0 or not self.runtime.base.ggml_is_contiguous(tensor):
                    raise ValueError('Steering requires contiguous float32 output')
                if self.runtime.base.ggml_nelements(tensor) != self.width * len(self.chunk_positions):
                    raise ValueError('Unexpected steering tensor shape')
                invocation = int(match[1]) // 22
                for i in indices:
                    position = self.chunk_positions[i]
                    if any(m['pass_index'] == invocation and m['position'] == position for m in self.mutations):
                        raise ValueError('Duplicate intervention callback')
                    before = np.empty(self.width, dtype=np.float32)
                    offset, size = i * self.width * 4, self.width * 4
                    self.runtime.base.ggml_backend_tensor_get(tensor, before.ctypes.data, offset, size)
                    after = before.copy() if config['alpha'] == 0 else before + np.float32(config['alpha']) * config['direction']
                    if not np.isfinite(after).all():
                        raise ValueError('Nonfinite steered activation')
                    wrote = config['alpha'] != 0 or config['identity_write']
                    if wrote:
                        self.runtime.base.ggml_backend_tensor_set(tensor, after.ctypes.data, offset, size)
                    observed = np.empty_like(after)
                    self.runtime.base.ggml_backend_tensor_get(tensor, observed.ctypes.data, offset, size)
                    np.testing.assert_array_equal(observed, after)
                    self.mutations.append(dict(pass_index=invocation, position=position, tensor=name,
                        before=before, after=observed, wrote=wrote))
                    # The callback captures the post-intervention tensor below.
            return super()._callback(tensor, ask, user)
        except BaseException as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')
            return False

    def replay(self, tokens, positions, capture=True, logit_positions=None):
        self.mutations = []
        if self.steering and max(self.steering['positions']) >= len(tokens):
            raise ValueError('Steering position outside input')
        result = super().replay(tokens, positions, capture, logit_positions)
        expected = len(self.steering['passes']) * len(self.steering['positions']) if self.steering else 0
        if len(self.mutations) != expected:
            raise RuntimeError(f'Incomplete steering callbacks: {len(self.mutations)}/{expected}')
        return result
