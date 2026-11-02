import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from experiments.nanbeige_reasoning.temperature_guard_burst import TemperatureGuard


class BurstGuardTests(unittest.TestCase):
    def test_warm_idle_and_four_to_one_duty(self):
        now=[0.];sleeps=[]
        def sleep(value):now[0]+=value;sleeps.append(value)
        def native(context,batch):now[0]+=.1;return 0
        def synchronize(context):pass
        with tempfile.TemporaryDirectory() as folder:
            g=TemperatureGuard(Path(folder)/'log',sensor=lambda:81,clock=lambda:now[0],sleep=sleep)
            model=SimpleNamespace(lib=SimpleNamespace(llama_decode=native,llama_synchronize=synchronize))
            g.attach(model);model.lib.llama_decode(None,None);model.lib.llama_decode(None,None)
            self.assertAlmostEqual(sleeps[0],.3)
            self.assertAlmostEqual(sleeps[1],.4)
            self.assertEqual(g.calls,2)
            self.assertEqual(g.policy['poll_seconds'],.25)

    def test_cooling_and_emergency_still_apply(self):
        with tempfile.TemporaryDirectory() as folder:
            now=[0.];values=iter([85,82,78,87])
            def sleep(value):now[0]+=value
            g=TemperatureGuard(Path(folder)/'log',sensor=lambda:next(values),clock=lambda:now[0],sleep=sleep)
            g.check(force=True);self.assertEqual(g.last_temperature,78)
            with self.assertRaises(RuntimeError):g.check(force=True)


if __name__=='__main__':unittest.main()
