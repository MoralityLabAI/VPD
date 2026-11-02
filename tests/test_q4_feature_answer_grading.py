import copy
import unittest
from experiments.nanbeige_reasoning.grade_q4_feature_answers import validate_record
from experiments.nanbeige_reasoning.compare_models import outcome_tables


class Tokenizer:
    def decode(self,tokens,**kwargs): return '<think>' if tokens==[1,2] else '</think> answer <eos>'


class FeatureAnswerGradingTests(unittest.TestCase):
    def test_audited_cleanup_preserves_raw_scoring_and_rejects_other_edits(self):
        class PunctuationTokenizer:
            def decode(self,tokens,**kwargs):
                return '<think>' if tokens==[1,2] else '</think> 6 . 6 <eos>'
        row=dict(complete=True,id='q',condition='baseline',generated=[3,9],stopped_on_eog=True,
            max_token_censored=False,prompt_tokens=2,model_processed_tokens=3,mutation_count=0,
            generated_text='</think> 6. 6 <eos>')
        q=dict(id='q',token_ids=[1,2]);arm=dict(name='baseline',passes=[])
        c=dict(max_new_tokens=8);runtime=dict(vocab_size=10,eog_token_ids=[9])
        with self.assertRaises(ValueError):validate_record(row,q,arm,c,runtime,PunctuationTokenizer())
        self.assertEqual(validate_record(row,q,arm,c,runtime,PunctuationTokenizer(),native_cleanup_verified=True),
            '<think></think> 6 . 6 <eos>')
        for altered in ['</think> 7. 6 <eos>', '</think> 6.6 <eos>', '</think>  6. 6 <eos>']:
            bad=copy.deepcopy(row);bad['generated_text']=altered
            with self.assertRaises(ValueError):
                validate_record(bad,q,arm,c,runtime,PunctuationTokenizer(),native_cleanup_verified=True)

    def test_stopping_and_invocation_corruptions_rejected(self):
        row=dict(complete=True,id='q',condition='both_plus',generated=[3,9],stopped_on_eog=True,
            max_token_censored=False,prompt_tokens=2,model_processed_tokens=3,mutation_count=6,
            generated_text='</think> answer <eos>')
        q=dict(id='q',token_ids=[1,2]);arm=dict(name='both_plus',passes=[0,1])
        c=dict(max_new_tokens=8);runtime=dict(vocab_size=10,eog_token_ids=[9])
        self.assertEqual(validate_record(row,q,arm,c,runtime,Tokenizer()),'<think></think> answer <eos>')
        for key,value in [('complete',False),('mutation_count',4),('model_processed_tokens',4),('max_token_censored',True),('generated',[9,3,9]),('generated_text','different')]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_record(bad,q,arm,c,runtime,Tokenizer())

    def test_partial_arc_task_is_missing_then_incorrect(self):
        questions=[dict(id=f'q{i}',group_id='one_task',lane='arc_agi_1',split='evaluation',calibration_split='eval') for i in range(2)]
        graded=[dict(id='q0',arm='baseline',grade=dict(status='graded',score=1))]
        tables=outcome_tables(questions,graded,['baseline'])
        self.assertEqual(tables['arms'][0]['missing'],1)
        self.assertIsNone(tables['arms'][0]['accuracy'])
        graded.append(dict(id='q1',arm='baseline',grade=dict(status='graded',score=0)))
        tables=outcome_tables(questions,graded,['baseline'])
        self.assertEqual(tables['arms'][0]['incorrect'],1)
        self.assertEqual(tables['arms'][0]['accuracy'],0)


if __name__=='__main__': unittest.main()
