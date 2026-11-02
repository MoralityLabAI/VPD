import unittest
import numpy as np
from experiments.nanbeige_reasoning.transfer_muon_to_vpd import CONDITIONS, check_feature_arithmetic


class TransferArithmeticTests(unittest.TestCase):
    def test_distinguishes_pass_and_intervention_differences_and_rejects_corruption(self):
        a = {}
        for index, name in enumerate(CONDITIONS):
            a[f'{name}__pass_0'] = np.zeros((2, 4), np.float32) if name.startswith('second') else np.full((2, 4), index, np.float32)
            a[f'{name}__pass_1'] = np.full((2, 4), 10+index, np.float32)
            a[f'{name}__pass_difference'] = a[f'{name}__pass_1']-a[f'{name}__pass_0']
            if name!='baseline':
                for i in (0, 1): a[f'{name}__delta_{i}'] = a[f'{name}__pass_{i}']-a[f'baseline__pass_{i}']
        self.assertEqual(check_feature_arithmetic(a), 21)
        a['first_plus__delta_1'][0, 0] += 1
        with self.assertRaises(AssertionError): check_feature_arithmetic(a)


if __name__ == '__main__': unittest.main()
