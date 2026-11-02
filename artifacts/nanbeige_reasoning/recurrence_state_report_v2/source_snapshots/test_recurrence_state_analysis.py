import unittest
import numpy as np
from .analyze_recurrence_state_control import encode,pool
from .intervention_change_fidelity import totals


class StateAnalysisTests(unittest.TestCase):
    def test_support_switch_is_separate_from_actual_change(self):
        w=np.eye(2);b=np.zeros(2);pre=np.zeros(2)
        baseline=np.array([[1.,.99]]);changed=np.array([[.99,1.]])
        z,mask=encode(baseline,w,b,pre,1)
        full,_=encode(changed,w,b,pre,1)
        frozen,_=encode(changed,w,b,pre,1,mask)
        np.testing.assert_array_equal(z,[[1,0]])
        np.testing.assert_array_equal(full,[[0,1]])
        np.testing.assert_array_equal(frozen,[[.99,0]])
        self.assertGreater(np.linalg.norm(full-z),90*np.linalg.norm(changed-baseline))

    def test_zero_and_relu_are_preserved(self):
        x=np.array([[-2.,1.]])
        z,mask=encode(x,np.eye(2),np.zeros(2),np.zeros(2),1)
        again,_=encode(x,np.eye(2),np.zeros(2),np.zeros(2),1,mask)
        np.testing.assert_array_equal(z,again)
        self.assertTrue(np.all(z>=0))

    def test_pool_uses_total_energy_not_mean_fidelity(self):
        rows=[dict(group='a',**totals(np.array([[1.]]),np.array([[1.]]))),
              dict(group='a',**totals(np.array([[10.]]),np.array([[0.]])))]
        r=pool(rows,['group'])[0]
        self.assertAlmostEqual(r['fidelity'],1/101)
        self.assertEqual(r['inputs'],2)

    def test_zero_signal_is_undefined(self):
        row=dict(group='a',**totals(np.zeros((1,2)),np.zeros((1,2))))
        self.assertIsNone(pool([row],['group'])[0]['fidelity'])

    def test_frozen_support_can_share_exact_encoder_arithmetic(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(9,7));x[5]=x[2]
        z,support,positive=encode(x,rng.normal(size=(13,7)),np.zeros(13),np.zeros(7),3,return_positive=True)
        frozen=np.where(support[2:3],positive[5:6],0.)
        np.testing.assert_array_equal(frozen,z[2:3])


if __name__=='__main__':unittest.main()
