"""Synthetic checks of boundary selection, probe math, and reused intervention hooks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .capture_decisions import decision_prefix
from .probes import fit_linear,predict,metrics,permute_within_world,clustered_interval
from .sae_analysis import describe_effect
from .io import digest


class CharacterTokenizer:
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)


class AnalysisTests(unittest.TestCase):
    def test_action_token_excluded(self):
        raw='Consider. </think>{"action":"M"}'
        response=dict(input_ids=[1,2],output_ids=list(map(ord,raw)))
        tokens,anchors,status=decision_prefix(response,CharacterTokenizer())
        self.assertEqual(status,'decision_prefix_found')
        self.assertEqual(CharacterTokenizer().decode(tokens[2:]),'Consider. </think>{"action":"')
        self.assertEqual(anchors[1]['excluded_next_token_id'],ord('M'))

    def test_truncated_reasoning_has_prompt_anchor_only(self):
        tokens,anchors,status=decision_prefix(dict(input_ids=[1],output_ids=list(map(ord,'thinking'))),CharacterTokenizer())
        self.assertEqual(tokens,[1]);self.assertEqual(len(anchors),1)

    def test_action_merged_with_punctuation_is_entirely_excluded(self):
        class MergedTokenizer:
            def decode(self,ids,**kwargs):return ''.join({1:'reason </think>',2:'{"action":',3:'"M"}',4:'eos'}[i] for i in ids)
        tokens,anchors,status=decision_prefix(dict(input_ids=[99],output_ids=[1,2,3,4]),MergedTokenizer())
        self.assertEqual(tokens,[99,1,2])
        self.assertEqual(anchors[1]['excluded_next_token_id'],3)

    def test_linear_probe_generalizes_known_linear_signal(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(200,8));y=(x[:,0]>0).astype(int)
        model=fit_linear(x[:100],y[:100]);m=metrics(y[100:],predict(model,x[100:]))
        self.assertGreater(m['auroc'],.97);self.assertGreater(m['balanced_accuracy'],.9)
        np.testing.assert_allclose(model['mean'],x[:100].mean(0))

    def test_missing_classes_remain_undefined(self):
        self.assertIsNone(metrics(np.zeros(4),np.arange(4))['auroc'])
        with self.assertRaises(ValueError):fit_linear(np.ones((4,2)),np.zeros(4))

    def test_permutation_preserves_world_class_counts(self):
        y=np.array([0,0,1,1,0,1,1,1]);w=['a']*4+['b']*4
        shuffled=permute_within_world(y,w,np.random.default_rng(8))
        self.assertEqual(y[:4].sum(),shuffled[:4].sum());self.assertEqual(y[4:].sum(),shuffled[4:].sum())

    def test_few_worlds_do_not_produce_confident_interval(self):
        r=clustered_interval(np.array([0,1]),np.array([-1,1]),['a','b'],np.random.default_rng(1))
        self.assertEqual(r['status'],'not_reported_fewer_than_four_test_worlds')

    def test_support_change_reconstruction_is_accounted_for(self):
        d=dict(encoder_weight=np.eye(2),encoder_bias=np.zeros(2),pre_bias=np.zeros(2),decoder=np.eye(2),top_k=np.array(1))
        arrays,m=describe_effect(d,np.array([[2.,1.]]),np.array([[2.,3.]]))
        np.testing.assert_array_equal(arrays['delta_h'],arrays['delta_reconstruction']+arrays['delta_error'])
        self.assertAlmostEqual(m['fidelity'],-.25)

    def test_existing_vpd_wrappers_preserve_baseline_and_restore_masks(self):
        import torch
        from experiments.nanbeige_looped.run_multipass import TinyRecurrentLM
        from .vpd_bridge import attach_frozen_factors,continue_action
        torch.manual_seed(42);model=TinyRecurrentLM();tokens=torch.tensor([[1,2,3]])
        before=model(tokens).detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'factors.npz'
            np.savez(path,shared__V=np.ones((8,2),np.float32)*.05,shared__U=np.ones((2,8),np.float32)*.07)
            target=attach_frozen_factors(model,path,digest(path))
            torch.testing.assert_close(target(tokens),before,rtol=0,atol=0)
            result=continue_action(target,tokens,path='shared',component=0,alpha=.05,passes=[1],eog_ids=set(),max_new_tokens=2)
            self.assertEqual(result['finish_reason'],'length')
            self.assertIsNone(target.physical['shared'].intervention)
            torch.testing.assert_close(target(tokens),before,rtol=0,atol=0)

    def test_existing_measurement_keeps_hidden_and_sae_effects_separate(self):
        import torch
        from experiments.nanbeige_looped.run_multipass import TinyRecurrentLM
        from experiments.nanbeige_looped.recurrent import RecurrentTarget
        from experiments.nanbeige_looped.sae import SharedSAE
        from .vpd_bridge import measured_probe_effects
        torch.manual_seed(3)
        target=RecurrentTarget(TinyRecurrentLM().eval(),{'shared':2})
        sae=SharedSAE(8,16,2).eval().requires_grad_(False)
        probe=dict(mean=np.zeros(8),scale=np.ones(8),coef=np.arange(8)/8,intercept=0.)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sae.npz';sae.export(path,site='norm')
            with np.load(path,allow_pickle=False) as z:dictionary={k:z[k] for k in z.files}
            arrays,features,receipt=measured_probe_effects(target,torch.tensor([[1,2,3]]),path='shared',component=0,alpha=.05,
                probe_site='norm',probe_invocation=1,probe=probe,dictionaries={'norm':sae},numpy_dictionaries={'norm':dictionary})
        self.assertEqual(len(receipt['probe_effects']),6)
        np.testing.assert_array_equal(features['second_plus__delta_h'],0.)
        np.testing.assert_array_equal(features['second_plus__delta_z'],0.)
        for scope in ['first','second','both']:
            for sign in ['plus','minus']:
                prefix=f'{scope}_{sign}__'
                np.testing.assert_allclose(features[prefix+'delta_h'],features[prefix+'delta_reconstruction']+features[prefix+'delta_error'])


if __name__=='__main__':unittest.main()
