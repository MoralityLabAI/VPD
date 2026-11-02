"""Scientific accounting checks for matched model/skill outcomes."""
import unittest
from .compare_skill_models import matched_tables


class MatchedSkillTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(id=str(i)) for i in range(4)]
        def rows(scores):return [dict(id=str(i),arm='vanilla',official_score=s,max_token_censored=i==3) for i,s in enumerate(scores)]
        return cases,rows([1,1,0,0]),rows([1,0,1,1])

    def test_pair_direction_and_censoring_are_retained(self):
        cases,left,right=self.fixture();_,rows=matched_tables(cases,left,right,['vanilla']);r=rows[0]
        self.assertEqual((r['both_correct'],r['qwen_only_correct'],r['nanbeige_only_correct'],r['both_incorrect']),(1,1,2,0))
        self.assertEqual(r['nanbeige_minus_qwen'],.25)
        self.assertEqual((r['qwen_capped'],r['nanbeige_capped']),(1,1))

    def test_missing_case_is_not_an_incorrect_answer(self):
        cases,left,right=self.fixture()
        with self.assertRaisesRegex(ValueError,'Missing'):matched_tables(cases,left,right[:-1],['vanilla'])

    def test_duplicate_case_is_not_an_extra_observation(self):
        cases,left,right=self.fixture()
        with self.assertRaisesRegex(ValueError,'duplicate'):matched_tables(cases,left,right+[right[0]],['vanilla'])


if __name__=='__main__':unittest.main()
