"""Measure how much of each original matrix the small fitted factor basis reconstructs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run_dir", type=Path)
p.add_argument("model_dir", type=Path)
args = p.parse_args()
torch.set_num_threads(4)
index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
training = json.loads((args.run_dir / "vpd_training.json").read_text())
metrics = {}
with np.load(args.run_dir / "vpd_factors.npz", allow_pickle=False) as factors:
    for name in training["physical_matrices"]:
        tensor_name = name + ".weight"
        with safe_open(args.model_dir / index[tensor_name], framework="pt", device="cpu") as checkpoint:
            target = checkpoint.get_tensor(tensor_name).float()
        v = torch.from_numpy(factors[name + "__V"])
        u = torch.from_numpy(factors[name + "__U"])
        fitted = (v @ u).T
        squared_error = float((target - fitted).square().sum())
        target_energy = float(target.square().sum())
        metrics[name] = {"components": int(v.shape[1]), "weight_shape": list(target.shape),
                         "relative_squared_reconstruction_error": squared_error / target_energy,
                         "fraction_squared_weight_energy_reconstructed": 1 - squared_error / target_energy,
                         "reconstruction_mse": squared_error / target.numel(),
                         "note": "Residual weights are retained by the VPD spillover component."}
        del target, fitted
(args.run_dir / "factor_coverage.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
