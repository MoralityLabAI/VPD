"""Numerical checks against actual shared-weight edits and known recurrences."""
import unittest

import numpy as np
import torch
from torch import nn

from experiments.nanbeige_looped import vpd_math as vpd
from experiments.nanbeige_looped.capture import torch_tail_rows
from experiments.nanbeige_looped.measure import capture_forward, measure_component
from experiments.nanbeige_looped.recurrent import RecurrentTarget, train_calibration
from experiments.nanbeige_looped.run_multipass import TinyRecurrentLM, validate_split, training_prefixes, architecture_canary
from pathlib import Path
from experiments.nanbeige_looped.sae import SharedSAE, encode_rows


class ScalarRecurrence(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.shared.weight.fill_(2)

    def forward(self, tokens):
        return self.shared(self.shared(tokens))


class MultipassTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_all_pass_intervention_matches_physical_weight_edit_and_restores(self):
        model = TinyRecurrentLM().eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        original_weight = model.shared.weight.detach().clone()
        original = model(tokens).detach()
        target = RecurrentTarget(model, {"shared": 3})
        torch.testing.assert_close(target(tokens), original)
        core = target.physical["shared"].core
        delta = torch.outer(core.U[1], core.V[:, 1]).detach()
        with target.intervene("shared", 1, 0.2, [0, 1]):
            diagnostic = target(tokens).detach()
        with torch.no_grad():
            core.W_target.copy_(original_weight + 0.2 * delta)
            physical_edit = target(tokens).detach()
            core.W_target.copy_(original_weight)
        torch.testing.assert_close(diagnostic, physical_edit, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(target(tokens), original)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with target.intervene("shared", 1, 0.2, [1]):
                raise RuntimeError("injected")
        self.assertIsNone(target.physical["shared"].intervention)
        torch.testing.assert_close(core.W_target, original_weight)

    def test_known_direct_inherited_and_interaction_effects(self):
        target = RecurrentTarget(ScalarRecurrence(), {"shared": 1})
        core = target.physical["shared"].core
        with torch.no_grad():
            core.V.fill_(1)
            core.U.fill_(1)
        tokens = torch.ones(1, 1, 1)
        # The runtime accepts a generic recurrent tensor model as well as LM IDs.
        self.assertEqual(float(target(tokens).detach()), 4)
        values = {}
        for scope, passes in {"first": [0], "second": [1], "both": [0, 1]}.items():
            with target.intervene("shared", 0, 0.5, passes):
                values[scope] = float(target(tokens).detach())
        self.assertEqual(values, {"first": 5, "second": 5, "both": 6.25})
        self.assertEqual(values["both"] - values["first"] - values["second"] + 4, 0.25)

    def test_invocation_masks_and_shared_gradients(self):
        target = RecurrentTarget(ScalarRecurrence(), {"shared": 1})
        core = target.physical["shared"].core
        with torch.no_grad():
            core.V.fill_(1)
            core.U.fill_(2)
        tokens = torch.ones(1, 1, 1)
        for view in target.views.values():
            view.mode = "component"
            view.mask = torch.ones(1, 1, 1)
            view.delta_mask = torch.zeros(1, 1)
        output = target(tokens)
        output.sum().backward()
        self.assertAlmostEqual(float(core.U.grad), 4.0)
        self.assertAlmostEqual(float(core.V.grad), 8.0)
        self.assertEqual(len(list(target.parameters())), 2)
        views = list(target.views.values())
        torch.testing.assert_close(views[0].last_input, torch.ones(1, 1, 1))
        torch.testing.assert_close(views[1].last_input, torch.full((1, 1, 1), 2.0))
        views[1].mask.zero_()
        self.assertEqual(float(target(tokens).detach()), 0.0)
        vpd.clear_wrapper_masks(target.views)
        self.assertEqual(float(target(tokens).detach()), 4.0)

    def test_masked_components_plus_spillover_reconstruct_target(self):
        model = TinyRecurrentLM().eval()
        tokens = torch.tensor([[1, 2, 3]])
        target = RecurrentTarget(model, {"shared": 3})
        expected = target(tokens).detach()
        for view in target.views.values():
            view.mode = "component"
            view.mask = torch.ones(1, 3, 3)
            view.delta_mask = torch.ones(1, 3)
        torch.testing.assert_close(target(tokens), expected, rtol=1e-5, atol=1e-6)

    def test_actual_sae_diffs_and_pass_causality(self):
        model = TinyRecurrentLM().eval()
        target = RecurrentTarget(model, {"shared": 2})
        sae = SharedSAE(8, 16, 4).eval().requires_grad_(False)
        tokens = torch.tensor([[1, 2, 3, 4]])
        arrays, summary = measure_component(target, tokens, ["norm"], {"norm": sae},
                                             path="shared", component=0, rows=3, alpha=0.1)
        expected, _ = encode_rows(sae, arrays["both_plus__norm__pass_1__h"])
        np.testing.assert_array_equal(expected, arrays["both_plus__norm__pass_1__z"])
        np.testing.assert_array_equal(arrays["second__norm__pass_0__z__positive_effect"], 0)
        self.assertGreater(np.linalg.norm(arrays["first__norm__pass_1__h__positive_effect"]), 0)
        np.testing.assert_array_equal(arrays["token_positions"], [1, 2, 3])
        np.testing.assert_array_equal(arrays["token_ids"], [2, 3, 4])
        self.assertIn("both_plus", summary["output_proxies"])
        self.assertEqual(len(model.norm._forward_hooks), 0)

    def test_short_optimizer_uses_all_four_objectives(self):
        target = RecurrentTarget(TinyRecurrentLM(), {"shared": 3})
        before = target.physical["shared"].core.V.detach().clone()
        cfg = vpd.Config(C_per_module={n: 3 for n in target.views}, batch_size=1,
                         seq_len=4, n_steps=2, faithfulness_warmup_steps=1,
                         ci_d_model=16, ci_n_blocks=1, ci_n_heads=2, ci_mlp_hidden=32)
        ci, history = train_calibration(target, [torch.tensor([[1, 2, 3, 4]])], cfg)
        self.assertEqual(set(history[0]), {"step", "faith", "importance", "stochastic", "adversarial"})
        self.assertEqual(len(ci.module_order), 2)
        self.assertFalse(torch.equal(before, target.physical["shared"].core.V))
        self.assertTrue(all(view.mode == "target" for view in target.views.values()))

    def test_generated_prefixes_retain_context_and_reject_eval_data(self):
        row = dict(id='reasoning',split='train',prompt_tokens=4,token_ids=list(range(12)))
        prefixes, receipt = training_prefixes([row], 'generated_prefixes', 2)
        self.assertEqual(prefixes,[list(range(6)),list(range(8)),list(range(12))])
        self.assertEqual([r['generated_tokens_in_prefix'] for r in receipt],[2,4,8])
        with self.assertRaisesRegex(ValueError,'training split'):
            training_prefixes([{**row,'split':'eval'}], 'generated_prefixes', 2)
        with self.assertRaisesRegex(ValueError,'continuation'):
            training_prefixes([{**row,'prompt_tokens':12}], 'generated_prefixes', 2)

    def test_publisher_recurrence_optimizes_alternating_prefix_lengths(self):
        model,_ = architecture_canary(Path('models/Nanbeige4.2-3B-tokenizer'))
        target = RecurrentTarget(model, {'model.layers.11.mlp.down_proj':2,'model.layers.21.mlp.down_proj':2})
        before = {name:wrapper.core.V.detach().clone() for name,wrapper in target.physical.items()}
        cfg = vpd.Config(C_per_module={n:2 for n in target.views},batch_size=1,seq_len=6,
            n_steps=3,faithfulness_warmup_steps=0,ci_d_model=16,ci_n_blocks=1,ci_n_heads=2,ci_mlp_hidden=32)
        batches = [torch.tensor([[1,2,3,4]]),torch.tensor([[1,2,3,4,5,6]])]
        _,history = train_calibration(target,batches,cfg,variable_lengths=True)
        self.assertEqual([r['sequence_length'] for r in history],[4,6,4])
        self.assertTrue(all(np.isfinite(r[key]) for r in history for key in ('faith','importance','stochastic','adversarial')))
        self.assertTrue(all(not torch.equal(before[name],wrapper.core.V) for name,wrapper in target.physical.items()))
        self.assertTrue(all(view.mode=='target' for view in target.views.values()))

    def test_alignment_and_reuse_errors_fail_explicitly(self):
        target = RecurrentTarget(TinyRecurrentLM(), {"shared": 2}, passes=3)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            target(torch.tensor([[1, 2]]))
        with self.assertRaises(ValueError):
            torch_tail_rows(torch.ones(1, 3, 2), rows=0)
        with self.assertRaises(ValueError):
            capture_forward(target, torch.ones(2, 3, dtype=torch.long), ["norm"])
        examples = [{"id": "a", "pair_id": "same", "split": "train", "token_ids": [1]},
                    {"id": "b", "pair_id": "same", "split": "eval", "token_ids": [2]}]
        with self.assertRaisesRegex(ValueError, "pair_id"):
            validate_split(examples)
        examples[1]["pair_id"] = "different"
        examples[1]["token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "token_ids"):
            validate_split(examples)


if __name__ == "__main__":
    unittest.main()
