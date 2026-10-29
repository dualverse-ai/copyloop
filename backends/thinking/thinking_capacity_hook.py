"""Observation-only capacity hook, installed separately in the smoke backend."""
import json
import torch

def install():
    from sglang.srt.model_executor.model_runner import ModelRunner
    if getattr(ModelRunner, '_thinking_capacity_installed', False): return
    original = ModelRunner.alloc_memory_pool
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        pool = self.token_to_kv_pool
        tensors=[]
        for name in ('k_buffer','v_buffer','kv_buffer'):
            value=getattr(pool,name,None)
            if isinstance(value,torch.Tensor): tensors.append((name,value))
            elif isinstance(value,(list,tuple)):
                tensors.extend((f'{name}/{i}',t) for i,t in enumerate(value) if isinstance(t,torch.Tensor))
        unique={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for _,t in tensors}
        free,total=torch.cuda.mem_get_info()
        print('thinking_capacity '+json.dumps(dict(max_total_num_tokens=self.max_total_num_tokens,
            eos_ids=sorted(self.model_config.hf_eos_token_id),pool_class=type(pool).__name__,
            pool_bytes=sum(unique.values()),pool_tensors=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype)) for n,t in tensors],
            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            free_bytes=free,total_bytes=total)),flush=True)
    ModelRunner.alloc_memory_pool=init
    ModelRunner._thinking_capacity_installed=True
