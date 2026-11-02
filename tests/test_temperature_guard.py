import tempfile
import unittest
from pathlib import Path
from experiments.nanbeige_reasoning.temperature_guard import TemperatureGuard


class GuardTests(unittest.TestCase):
    def guard(self,values):
        values=iter(values);now=[0.]
        def sleep(seconds):now[0]+=seconds
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        return TemperatureGuard(Path(folder.name)/'events.jsonl',sensor=lambda:next(values),clock=lambda:now[0],sleep=sleep)

    def test_hysteresis_waits_until_lower_threshold(self):
        g=self.guard([79,76,70,68]);g.check(force=True)
        self.assertEqual(g.last_temperature,68)
        self.assertEqual(g.cooling_pauses,1)
        self.assertEqual(g.idle_seconds,6)

    def test_hot_reading_and_missing_sensor_stop(self):
        for values in [[84],[]]:
            with self.subTest(values=values),self.assertRaises(RuntimeError): self.guard(values).check(force=True)

    def test_failed_cooling_stops_instead_of_restarting_hot(self):
        g=self.guard([79]*100)
        with self.assertRaisesRegex(RuntimeError,'Cooling or wall budget'):g.check(force=True)
        self.assertEqual(g.idle_seconds,180)


if __name__=='__main__':unittest.main()
