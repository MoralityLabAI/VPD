"""Equal-executed-depth carry/reset control using the existing native runtime.

Reset replaces the entire loop-boundary residual with the original embedding
for every token, including prefix tokens. This removes state carry, not training
on recurrence. The 44-layer graph and separate invocation KV slots remain.
"""
import ctypes as C
import numpy as np
from .llama_capture import P,I,B
from .llama_steer import SteeringModel


class StateHistoryModel(SteeringModel):
    def __init__(self,*args,reset_history=False,**kwargs):
        self.reset_history=reset_history
        self.embedding_chunk=None;self.boundary_writes=0
        super().__init__(*args,**kwargs)
        if reset_history and self.loops!=2:raise ValueError('Reset requires two executed passes')
        self.lib.llama_memory_seq_rm.argtypes=[P,I,I,I]
        self.lib.llama_memory_seq_rm.restype=B

    def _read_chunk(self,tensor):
        if C.cast(tensor,C.POINTER(I))[0]!=0 or not self.runtime.base.ggml_is_contiguous(tensor):
            raise ValueError('Expected contiguous float32 boundary state')
        n=self.runtime.base.ggml_nelements(tensor)
        if n!=len(self.chunk_positions)*self.width:raise ValueError('Boundary dimensions differ')
        x=np.empty((len(self.chunk_positions),self.width),np.float32)
        self.runtime.base.ggml_backend_tensor_get(tensor,x.ctypes.data,0,x.nbytes)
        if not np.isfinite(x).all():raise ValueError('Nonfinite boundary state')
        return x

    def _callback(self,tensor,ask,user):
        try:
            name=self.runtime.base.ggml_get_name(tensor).decode()
            if self.reset_history and name in ('embd','loop_norm-21'):
                if ask:return True
                x=self._read_chunk(tensor)
                if name=='embd':self.embedding_chunk=x.copy()
                else:
                    if self.embedding_chunk is None or self.embedding_chunk.shape!=x.shape:
                        raise ValueError('Missing same-chunk original embeddings')
                    self.runtime.base.ggml_backend_tensor_set(tensor,self.embedding_chunk.ctypes.data,0,x.nbytes)
                    np.testing.assert_array_equal(self._read_chunk(tensor),self.embedding_chunk)
                    self.boundary_writes+=len(self.chunk_positions)
            return super()._callback(tensor,ask,user)
        except BaseException as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')
            return False

    def clear_cache(self):
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.context),True)
        self.clear_steering()

    def append(self,tokens,start,*,capture_last=False):
        if not tokens or start<0:raise ValueError('Nonempty tokens and nonnegative position required')
        self.enabled=capture_last;self.selected={start+len(tokens)-1} if capture_last else set()
        self.arrays={};self.errors=[];self.mutations=[];self.boundary_writes=0
        batch=self.lib.llama_batch_init(64,0,1)
        try:
            for offset in range(0,len(tokens),64):
                chunk=tokens[offset:offset+64]
                self.chunk_positions=list(range(start+offset,start+offset+len(chunk)))
                self.embedding_chunk=None;batch.n_tokens=len(chunk)
                for i,t in enumerate(chunk):
                    batch.token[i]=t;batch.pos[i]=self.chunk_positions[i]
                    batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1
                result=self.lib.llama_decode(self.context,batch)
                if result or self.errors:raise RuntimeError(f'Decode {result}: {self.errors}')
            if self.reset_history and self.boundary_writes!=len(tokens):raise ValueError('Missing boundary reset')
            logits=np.ctypeslib.as_array(self.lib.llama_get_logits_ith(self.context,len(chunk)-1),shape=(self.vocab_size,)).copy()
            if not np.isfinite(logits).all():raise ValueError('Nonfinite logits')
            expected=len(self.steering['passes'])*len(self.steering['positions']) if self.steering else 0
            if len(self.mutations)!=expected:raise ValueError('Missing or repeated direct writes')
            arrays={key:row[None,:] for (key,pos),row in self.arrays.items()}
            if capture_last and len(arrays)!=23*self.loops:raise ValueError('Incomplete all-layer capture')
            return arrays,logits
        finally:self.lib.llama_batch_free(batch)

    def last_token(self,tokens,*,direction=None,dose=0.,layer=22):
        position=len(tokens)-1
        mem=self.lib.llama_get_memory(self.context)
        if not self.lib.llama_memory_seq_rm(mem,0,position,-1):raise ValueError('Could not remove last-token KV entries')
        self.clear_steering()
        if direction is not None:
            self.set_steering(direction,alpha=dose,passes=[self.loops-1],positions=[position],physical_layer=layer,identity_write=dose==0)
        return self.append(tokens[-1:],position,capture_last=True)
