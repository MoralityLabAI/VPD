"""Pinned llama.cpp b10731 C ABI bindings for observing recurrent activations.

Layouts are transcribed from the revision-pinned headers in sources/llama_cpp.
Use only with the runtime hashes recorded by the Q4 experiment.
"""
import ctypes as C
import os
import re
from pathlib import Path
import numpy as np

P=C.c_void_p
I=C.c_int32
U=C.c_uint32
F=C.c_float
B=C.c_bool
CALLBACK=C.CFUNCTYPE(B,P,B,P)


class OverrideValue(C.Union):
    _fields_=[('val_i64',C.c_int64),('val_f64',C.c_double),('val_bool',B),('val_str',C.c_char*128)]


class Override(C.Structure):
    _anonymous_=('value',)
    _fields_=[('tag',I),('key',C.c_char*128),('value',OverrideValue)]


class ModelParams(C.Structure):
    _fields_=[('devices',P),('tensor_buft_overrides',P),('n_gpu_layers',I),('split_mode',I),
        ('load_mode',I),('lazy_mode',I),('main_gpu',I),('tensor_split',P),
        ('progress_callback',P),('progress_callback_user_data',P),('kv_overrides',P),
        *[(n,B) for n in ('vocab_only','check_tensors','use_extra_bufts','no_host','no_alloc','load_mtp')]]


class ContextParams(C.Structure):
    _fields_=[*[(n,U) for n in ('n_ctx','n_batch','n_ubatch','n_seq_max','n_rs_seq','n_outputs_max','n_outputs_max_per_seq')],
        *[(n,I) for n in ('n_threads','n_threads_batch','ctx_type','rope_scaling_type','pooling_type','attention_type','flash_attn_type')],
        *[(n,F) for n in ('rope_freq_base','rope_freq_scale','yarn_ext_factor','yarn_attn_factor','yarn_beta_fast','yarn_beta_slow')],
        ('yarn_orig_ctx',U),('defrag_thold',F),('cb_eval',P),('cb_eval_user_data',P),
        ('type_k',I),('type_v',I),('abort_callback',P),('abort_callback_data',P),
        *[(n,B) for n in ('embeddings','offload_kqv','no_perf','op_offload','swa_full','kv_unified')],
        ('samplers',P),('n_samplers',C.c_size_t),('ctx_other',P)]


class Batch(C.Structure):
    _fields_=[('n_tokens',I),('token',C.POINTER(I)),('embd',C.POINTER(F)),
        ('pos',C.POINTER(I)),('n_seq_id',C.POINTER(I)),('seq_id',C.POINTER(C.POINTER(I))),
        ('logits',C.POINTER(C.c_int8))]


class Runtime:
    def __init__(self,path,require_cuda=False):
        self.path=Path(path).resolve()
        self.dll_directory=os.add_dll_directory(str(self.path))
        self.base=C.CDLL(str(self.path/'ggml-base.dll'))
        self.backend=C.CDLL(str(self.path/'ggml.dll'))
        self.llama=C.CDLL(str(self.path/'llama.dll'))
        self.cuda_dependencies=[]
        if require_cuda:
            # The backend's native LoadLibrary call does not inherit ctypes'
            # DLL-directory flags. Preload pinned dependencies with those flags.
            for name in ('cudart64_12.dll','cublasLt64_12.dll','cublas64_12.dll','ggml-cuda.dll'):
                self.cuda_dependencies.append(C.CDLL(str(self.path/name)))
        specs={
            'llama_model_default_params':(ModelParams,[]),
            'llama_context_default_params':(ContextParams,[]),
            'llama_backend_init':(None,[]),'llama_backend_free':(None,[]),
            'llama_model_load_from_file':(P,[C.c_char_p,ModelParams]),
            'llama_model_free':(None,[P]),'llama_init_from_model':(P,[P,ContextParams]),
            'llama_free':(None,[P]),'llama_model_n_layer':(I,[P]),'llama_model_n_embd':(I,[P]),
            'llama_model_get_vocab':(P,[P]),'llama_vocab_n_tokens':(I,[P]),
            'llama_batch_init':(Batch,[I,I,I]),'llama_batch_free':(None,[Batch]),
            'llama_decode':(I,[P,Batch]),'llama_get_logits_ith':(C.POINTER(F),[P,I]),
            'llama_get_memory':(P,[P]),'llama_memory_clear':(None,[P,B])}
        for name,(result,arguments) in specs.items():
            function=getattr(self.llama,name);function.restype=result;function.argtypes=arguments
        for name,result,arguments in [('ggml_get_name',C.c_char_p,[P]),('ggml_nelements',C.c_int64,[P]),
                ('ggml_nrows',C.c_int64,[P]),('ggml_nbytes',C.c_size_t,[P]),('ggml_is_contiguous',B,[P])]:
            function=getattr(self.base,name);function.restype=result;function.argtypes=arguments
        self.backend.ggml_backend_load_all_from_path.argtypes=[C.c_char_p]
        self.backend.ggml_backend_load_all_from_path.restype=None
        self.base.ggml_backend_tensor_get.argtypes=[P,P,C.c_size_t,C.c_size_t]
        self.base.ggml_backend_tensor_get.restype=None
        self.backend.ggml_backend_load_all_from_path(str(self.path).encode())
        self.llama.llama_backend_init()
        for name,result,arguments in [('ggml_backend_dev_count',C.c_size_t,[]),
                ('ggml_backend_dev_get',P,[C.c_size_t]),('ggml_backend_dev_name',C.c_char_p,[P]),
                ('ggml_backend_dev_type',I,[P]),('ggml_backend_dev_backend_reg',P,[P]),
                ('ggml_backend_reg_name',C.c_char_p,[P])]:
            function=getattr(self.backend,name,None) or getattr(self.base,name)
            function.restype=result;function.argtypes=arguments
            setattr(self.backend,name,function)
        self.devices=[]
        for i in range(self.backend.ggml_backend_dev_count()):
            pointer=self.backend.ggml_backend_dev_get(i)
            self.devices.append(dict(pointer=pointer,name=self.backend.ggml_backend_dev_name(pointer).decode(),
                type=self.backend.ggml_backend_dev_type(pointer),
                backend=self.backend.ggml_backend_reg_name(self.backend.ggml_backend_dev_backend_reg(pointer)).decode()))


class CaptureModel:
    def __init__(self,runtime,model_path,loops=2,context=4096,gpu_layers=0,threads=2,capture_mlp_inputs=False):
        self.runtime=runtime;self.lib=runtime.llama;self.context=None;self.model=None
        self.loops=loops;self.width=3072;self.chunk_positions=[];self.selected=set()
        self.capture_mlp_inputs=capture_mlp_inputs
        self.enabled=False;self.arrays={};self.errors=[];self.observed_names=set()
        self.callback=CALLBACK(self._callback)
        overrides=(Override*2)();overrides[0].tag=0
        overrides[0].key=b'nanbeige.num_loops';overrides[0].val_i64=loops
        params=self.lib.llama_model_default_params()
        params.n_gpu_layers=gpu_layers;params.kv_overrides=C.cast(overrides,P)
        if gpu_layers!=0:
            cuda=[d for d in runtime.devices if d['backend']=='CUDA' and d['type']==1]
            if len(cuda)!=1:raise RuntimeError(f'Expected one CUDA device, observed {[d["name"] for d in runtime.devices]}')
            self.device_list=(P*2)(cuda[0]['pointer'],None)
            params.devices=C.cast(self.device_list,P)
        self.model=self.lib.llama_model_load_from_file(str(Path(model_path).resolve()).encode(),params)
        if not self.model:raise RuntimeError('Could not load Q4 capture model')
        if self.lib.llama_model_n_layer(self.model)!=22*loops or self.lib.llama_model_n_embd(self.model)!=3072:
            self.close();raise RuntimeError('Model layout or loop override mismatch')
        self.vocab_size=self.lib.llama_vocab_n_tokens(self.lib.llama_model_get_vocab(self.model))
        cp=self.lib.llama_context_default_params()
        cp.n_ctx=context;cp.n_batch=64;cp.n_ubatch=64;cp.n_seq_max=1
        cp.n_threads=threads;cp.n_threads_batch=threads
        cp.type_k=8;cp.type_v=8;cp.flash_attn_type=1
        cp.offload_kqv=gpu_layers!=0;cp.op_offload=gpu_layers!=0
        cp.cb_eval=C.cast(self.callback,P)
        self.context=self.lib.llama_init_from_model(self.model,cp)
        if not self.context:self.close();raise RuntimeError('Could not allocate capture context')

    def _callback(self,tensor,ask,user):
        try:
            name=self.runtime.base.ggml_get_name(tensor).decode()
            width=self.width
            match=re.fullmatch(r'l_out-(\d+)',name)
            mlp_match=re.fullmatch(r'ffn_swiglu-(\d+)',name)
            if match:
                logical=int(match[1]);key=f'model.layers.{logical%22}__pass_{logical//22}'
            elif name=='loop_norm-21':key='model.norm__pass_0'
            elif name=='result_norm':key=f'model.norm__pass_{self.loops-1}'
            elif self.capture_mlp_inputs and mlp_match and int(mlp_match[1])%22 in (11,21):
                logical=int(mlp_match[1]);width=10752
                key=f'model.layers.{logical%22}.mlp.down_proj.input__pass_{logical//22}'
            else:return False if ask else True
            keep=[i for i,pos in enumerate(self.chunk_positions) if pos in self.selected]
            if ask:return bool(self.enabled and keep)
            self.observed_names.add(name)
            if C.cast(tensor,C.POINTER(I))[0]!=0 or not self.runtime.base.ggml_is_contiguous(tensor):
                raise ValueError('Expected contiguous float32 activation')
            count=self.runtime.base.ggml_nelements(tensor)
            if count!=width*len(self.chunk_positions):
                raise ValueError(f'Unexpected activation shape {name}: {count}')
            for index in keep:
                row=np.empty(width,dtype=np.float32)
                self.runtime.base.ggml_backend_tensor_get(tensor,row.ctypes.data,index*width*4,width*4)
                pair=(key,self.chunk_positions[index])
                if pair in self.arrays:raise ValueError('Duplicate activation callback')
                self.arrays[pair]=row
            return True
        except BaseException as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')
            return False

    def replay(self,tokens,positions,capture=True,logit_positions=None,chunk_size=64):
        if not 1<=chunk_size<=64:raise ValueError('Replay chunk size must be 1..64')
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.context),True)
        self.selected=set(positions);self.enabled=capture;self.arrays={};self.errors=[]
        wanted=set(logit_positions if logit_positions is not None else [len(tokens)-1])
        logits={}
        batch=self.lib.llama_batch_init(chunk_size,0,1)
        try:
            for start in range(0,len(tokens),chunk_size):
                chunk=tokens[start:start+chunk_size];batch.n_tokens=len(chunk)
                self.chunk_positions=list(range(start,start+len(chunk)))
                for i,token in enumerate(chunk):
                    batch.token[i]=token;batch.pos[i]=start+i
                    batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1
                result=self.lib.llama_decode(self.context,batch)
                if result or self.errors:raise RuntimeError(f'Decode {result}; callback errors: {self.errors}')
                for i,pos in enumerate(self.chunk_positions):
                    if pos in wanted:
                        pointer=self.lib.llama_get_logits_ith(self.context,i)
                        logits[pos]=np.ctypeslib.as_array(pointer,shape=(self.vocab_size,)).copy()
        finally:self.lib.llama_batch_free(batch)
        arrays={}
        if capture:
            keys=[f'model.layers.{i}__pass_{j}' for j in range(self.loops) for i in range(22)]
            keys += [f'model.norm__pass_{j}' for j in range(self.loops)]
            if self.capture_mlp_inputs:
                keys += [f'model.layers.{i}.mlp.down_proj.input__pass_{j}' for j in range(self.loops) for i in (11,21)]
            for key in keys:
                arrays[key]=np.stack([self.arrays[(key,pos)] for pos in positions])
        return arrays,logits

    def close(self):
        if self.context:self.lib.llama_free(self.context);self.context=None
        if self.model:self.lib.llama_model_free(self.model);self.model=None
