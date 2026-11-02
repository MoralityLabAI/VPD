"""A common top-k dictionary for aligned recurrent-pass comparisons."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SharedSAE(nn.Module):
    def __init__(self, width, features, top_k):
        super().__init__()
        if not 1 <= top_k <= features or width < 1:
            raise ValueError("Invalid SAE dimensions or sparsity")
        self.top_k = top_k
        self.pre_bias = nn.Parameter(torch.zeros(width))
        self.encoder = nn.Linear(width, features)
        self.decoder = nn.Linear(features, width, bias=False)
        with torch.no_grad():
            self.encoder.bias.zero_()
            self.decoder.weight.copy_(self.encoder.weight.T)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x):
        positive = self.encoder(x - self.pre_bias).relu()
        values, indices = positive.topk(self.top_k, dim=-1)
        return torch.zeros_like(positive).scatter(-1, indices, values)

    def forward(self, x):
        features = self.encode(x)
        return self.decoder(features) + self.pre_bias, features

    def export(self, path, *, site):
        np.savez_compressed(path, activation_key=np.asarray(site),
                            encoder_weight=self.encoder.weight.detach().cpu().numpy(),
                            encoder_bias=self.encoder.bias.detach().cpu().numpy(),
                            decoder=self.decoder.weight.detach().cpu().numpy().T,
                            pre_bias=self.pre_bias.detach().cpu().numpy(),
                            top_k=np.asarray(self.top_k))


def fit_shared_sae(pass_rows, *, steps=100, expansion=2, top_k=32, batch_size=128, seed=42, device="cpu"):
    """Balanced passes from training examples only; no per-pass normalization."""
    if steps < 1 or batch_size < 1 or not pass_rows:
        raise ValueError("Positive training budget and nonempty pass data required")
    if len({array.shape for array in pass_rows}) != 1:
        raise ValueError("Passes must contain identical aligned token rows")
    if any(array.ndim != 2 or not len(array) or not np.isfinite(array).all()
           for array in pass_rows):
        raise ValueError("Invalid activation rows")
    torch.manual_seed(seed)
    width = pass_rows[0].shape[1]
    model = SharedSAE(width, width * expansion, top_k).to(device)
    with torch.no_grad():
        model.pre_bias.copy_(torch.from_numpy(np.mean([x.mean(0) for x in pass_rows], axis=0)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = np.random.default_rng(seed)
    for _ in range(steps):
        rows = generator.integers(len(pass_rows[0]), size=max(1, batch_size // len(pass_rows)))
        batch = torch.from_numpy(np.concatenate([x[rows] for x in pass_rows])).to(device=device, dtype=torch.float32)
        reconstruction, _ = model(batch)
        loss = (reconstruction - batch).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite SAE objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def encode_rows(sae, rows):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("Finite nonempty activation rows required")
    x = torch.from_numpy(rows).to(next(sae.parameters()).device)
    reconstruction, features = sae(x)
    squared_error = float((reconstruction - x).square().sum())
    variance = float((x - x.mean(0)).square().sum())
    return features.cpu().numpy(), {
        "mse": squared_error / x.numel(),
        "explained_variance": 1 - squared_error / variance if variance > 0 else None,
        "mean_l0": float((features > 0).float().sum(-1).mean()),
        "rows": len(rows),
    }
