import unittest
import numpy as np
import torch

from experiments.nanbeige_looped.sae import fit_shared_sae
from experiments.nanbeige_reasoning.control_sae_muon import initialize, optimizers
from scripts.nanbeige.verify_sae_muon import check_delta


class MuonControlTests(unittest.TestCase):
    def test_baseline_preserves_existing_sae_training_recipe(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(3)
        train = [rng.normal(size=(8, 6)).astype(np.float32) for _ in range(2)]
        expected = fit_shared_sae(train, steps=3, expansion=2, top_k=2, seed=42)
        actual, _ = initialize(train, 42, features=12, top_k=2)
        opts, _ = optimizers(actual, 'adamw')
        batches = np.random.default_rng(42).integers(8, size=(3, 64))
        for idx in batches:
            x = torch.from_numpy(np.concatenate([a[idx] for a in train]))
            y, _ = actual(x)
            loss = (y-x).square().reshape(2, 64, 6).mean((1, 2)).mean()
            opts[0].zero_grad(set_to_none=True); loss.backward(); opts[0].step(); actual.normalize_decoder()
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_hybrid_partitions_parameters_and_updates_biases(self):
        train = [np.ones((8, 6), np.float32) * v for v in (1, 3)]
        model, _ = initialize(train, 42, features=12, top_k=2)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opts, groups = optimizers(model, 'muon_hybrid')
        self.assertEqual(set(groups['muon']), {'encoder.weight', 'decoder.weight'})
        self.assertEqual(set(groups['adamw']), {'pre_bias', 'encoder.bias'})
        x = torch.from_numpy(np.concatenate(train)); y, _ = model(x)
        (y-x).square().mean().backward()
        for opt in opts: opt.step()
        for n, p in model.named_parameters():
            self.assertFalse(torch.equal(before[n], p), n)

    def test_independent_delta_check_rejects_corruption(self):
        a = np.zeros((2, 4), np.float32); b = np.ones((2, 4), np.float32)
        check_delta(a, b, b-a)
        wrong = b-a; wrong[0, 0] += .1
        with self.assertRaises(AssertionError): check_delta(a, b, wrong)


if __name__ == '__main__': unittest.main()
