"""Pytest configuration for fast deterministic CPU tests."""

import torch

# Prevent OpenMP thread-pool thrashing on tiny unit-test tensors on multi-core Windows hosts
if torch.get_num_threads() > 2:
    torch.set_num_threads(2)
