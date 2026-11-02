import unittest
import numpy as np
from .summarize_pass_dynamics import paired_metrics,grouped_states


class PassDynamicsTests(unittest.TestCase):
    def test_identity_and_sign_reversal(self):
        x=np.array([[1.,2.],[3.,4.]])
        same=paired_metrics(x,x);flipped=paired_metrics(x,-x)
        self.assertAlmostEqual(same['mean_cosine'],1.)
        self.assertEqual(same['mean_relative_change'],0.)
        self.assertAlmostEqual(flipped['mean_cosine'],-1.)
        self.assertAlmostEqual(flipped['mean_relative_change'],2.)

    def test_tasks_with_more_test_grids_do_not_get_more_weight(self):
        rows=[]
        for group,value in [('a',0.),('a',0.),('b',1.)]:
            rows.append(dict(lane='arc_agi_1',calibration_split='eval',region='late',physical_layer=12,group_id=group,
                mean_cosine=value,mean_relative_change=value,mean_pass1_norm=1.,mean_pass2_norm=1.))
        tasks,pooled=grouped_states(rows)
        self.assertEqual(pooled[0]['task_groups'],2)
        self.assertEqual(pooled[0]['inputs'],3)
        self.assertEqual(pooled[0]['mean_cosine'],.5)


if __name__=='__main__':unittest.main()
