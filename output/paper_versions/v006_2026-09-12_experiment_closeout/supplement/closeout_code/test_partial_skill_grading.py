"""Missing outputs must not become negative outcomes or unmatched comparisons."""
import unittest
from .grade_logic_skill_comparison import paired, source_key, validate_coverage
from .compare_skill_models import partial_matched_tables


class PartialGradingTests(unittest.TestCase):
    def test_only_scheduled_prefix_allowed(self):
        calls=dict.fromkeys(['x','y','z'])
        validate_coverage(dict.fromkeys(['x','y']),calls,True)
        for keys in ([],['y'],['x','z'],['y','x'],['x','y','z','extra']):
            with self.assertRaises(ValueError):validate_coverage(dict.fromkeys(keys),calls,True)
        with self.assertRaises(ValueError):validate_coverage({'x':None},calls,False)

    def test_pair_denominator_is_intersection(self):
        rows=[dict(id='one',arm='a',official_score=1),dict(id='one',arm='b',official_score=0),
              dict(id='two',arm='a',official_score=0)]
        result=paired(rows,'a','b',True)
        self.assertEqual((result['pairs'],result['a_only'],result['both_incorrect']),(1,1,0))
        with self.assertRaises(AssertionError):paired(rows,'a','b')
        with self.assertRaises(AssertionError):paired(rows+[rows[0]],'a','b',True)
        self.assertEqual(paired(rows,'a','missing',True)['pairs'],0)

    def test_repairs_reuse_only_their_actual_source(self):
        case=dict(id='one',route='logic_skill_trm')
        self.assertEqual(source_key(case,'routed_final'),'one::logic_skill_trm')
        self.assertEqual(source_key(case,'plain_candidate_routed_repair'),'one::logic_skill')
        self.assertEqual(source_key(case,'vanilla'),'one::vanilla')

    def test_cross_model_denominators_follow_each_procedure(self):
        def row(i,a,s):return dict(id=i,arm=a,official_score=s,max_token_censored=True)
        left=[row(i,a,0) for i in ('one','two') for a in ('a','b')]
        right=[row('one','a',1),row('two','a',0),row('one','b',1)]
        _,result=partial_matched_tables([dict(id=i) for i in ('one','two')],left,right,['a','b'])
        self.assertEqual([r['puzzles'] for r in result],[2,1])
        self.assertEqual([r['nanbeige_only_correct'] for r in result],[1,1])
        with self.assertRaises(ValueError):partial_matched_tables([dict(id='one')],left,right,['a','b'])


if __name__=='__main__':unittest.main()
