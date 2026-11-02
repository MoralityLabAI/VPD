"""Check loss, gradient and update equivalence of autograd CPU offload.

Uses a tiny random recurrent model on the requested device, never a pretrained
reasoning result. The CUDA control exercises actual host/device tensor copies.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.nanbeige_looped import vpd_math as vpd
from experiments.nanbeige_looped.recurrent import RecurrentTarget, train_calibration
from experiments.nanbeige_looped.run_multipass import TinyRecurrentLM


def control(device):
    outputs = []
    for offload in (False, True):
        torch.manual_seed(123)
        target = RecurrentTarget(TinyRecurrentLM().to(device), {'shared': 3})
        batches = [torch.tensor([[1,3,5,7,9,2,4,6,8,10,12,14][:n]], device=device) for n in (7,12,9)]
        cfg = vpd.Config(C_per_module={k:3 for k in target.views}, batch_size=1, seq_len=12,
                         n_steps=3, faithfulness_warmup_steps=2, ci_d_model=16,
                         ci_n_blocks=1, ci_n_heads=4, ci_mlp_hidden=32)
        progress = []
        ci, history = train_calibration(target, batches, cfg, variable_lengths=True,
            saved_tensors_on_cpu=offload, progress_callback=progress.append)
        parameters = {'V':target.physical['shared'].core.V, 'U':target.physical['shared'].core.U}
        parameters.update({'ci.'+name:value for name,value in ci.named_parameters()})
        snapshots = {name:value.detach().cpu().clone() for name,value in parameters.items()}
        snapshots.update({name+'.grad':value.grad.detach().cpu().clone()
                          for name,value in parameters.items() if value.grad is not None})
        if history != progress:
            raise ValueError('Progress callback changed or omitted training records')
        outputs.append((history,snapshots))
        del ci, target, parameters
    maximum_error = 0.
    if outputs[0][1].keys() != outputs[1][1].keys():
        raise ValueError('Offload changed optimized parameter/gradient keys')
    for left,right in zip(outputs[0][0],outputs[1][0],strict=True):
        for name in left:
            torch.testing.assert_close(torch.tensor(left[name]),torch.tensor(right[name]),rtol=1e-5,atol=1e-6)
    for name, left in outputs[0][1].items():
        right = outputs[1][1][name]
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
        maximum_error = max(maximum_error,float((left-right).abs().max()))
    return dict(status='saved_tensor_offload_control_passed', device=str(device),
        torch_version=torch.__version__, random_fixture_only=True, model_recomputed_for_backward=False,
        steps=3, sequence_lengths=[7,12,9], compared_parameter_and_gradient_arrays=len(outputs[0][1]),
        maximum_parameter_or_gradient_error=maximum_error, baseline_history=outputs[0][0],
        offloaded_history=outputs[1][0], rtol=1e-5,atol=1e-6)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    result=control(args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if 'history' not in k}))
