import tempfile
import unittest
from pathlib import Path
from experiments.nanbeige_reasoning.temperature_guard_85 import TemperatureGuard


class Guard85Tests(unittest.TestCase):
    def guard(self,values):
        values=iter(values);now=[0.]
        def sleep(seconds):now[0]+=seconds
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        return TemperatureGuard(Path(folder.name)/'events.jsonl',sensor=lambda:next(values),clock=lambda:now[0],sleep=sleep)

    def test_84_runs_and_85_waits_for_78(self):
        g=self.guard([84]);g.check(force=True)
        self.assertEqual(g.cooling_pauses,0)
        g=self.guard([85,83,80,78]);g.check(force=True)
        self.assertEqual(g.last_temperature,78)
        self.assertEqual(g.idle_seconds,6)

    def test_emergency_sensor_and_timeout_stops(self):
        for values in [[87],[],[85]*100]:
            with self.subTest(values=values[:1]),self.assertRaises(RuntimeError):self.guard(values).check(force=True)


if __name__=='__main__':unittest.main()
