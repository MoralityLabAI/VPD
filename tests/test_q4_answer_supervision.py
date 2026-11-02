import json
import tempfile
import unittest
from pathlib import Path

from experiments.nanbeige_reasoning.run_q4_feature_answers import save_record
from scripts.nanbeige.supervise_q4_feature_answers import assess_stop


class SupervisionTests(unittest.TestCase):
    def test_only_progressing_clean_wall_stop_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            def write(name,obj):(p/name).write_text(json.dumps(obj))
            write('cleanup.json',dict(model_closed=True,context_closed=True))
            write('manifest.json',dict(status='q4_feature_answers_partial',wall_budget_reached=False))
            write('interrupted.json',dict(error='RuntimeError: Guard wall budget reached'))
            save_record(p/'00_baseline.json',dict(generated=[1,2]))
            self.assertEqual(assess_stop(p,1,1),'resume')
            with self.assertRaisesRegex(RuntimeError,'no saved tokens'):assess_stop(p,1,2)
            for error in ['Temperature hard-stop threshold reached','Temperature unavailable; stopping decoding','Cooling or wall budget exhausted; stopping decoding']:
                write('interrupted.json',dict(error='RuntimeError: '+error))
                with self.assertRaisesRegex(RuntimeError,'Automatic continuation stopped'):assess_stop(p,1,1)
            write('cleanup.json',dict(model_closed=True,context_closed=False))
            with self.assertRaisesRegex(RuntimeError,'cleanup'):assess_stop(p,1,1)


if __name__=='__main__':unittest.main()
