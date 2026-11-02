"""Guard policy checks without hardware inference."""
from pathlib import Path
import tempfile
import unittest
from experiments.nanbeige_reasoning.temperature_guard_burst import TemperatureGuard


class ThermalTests(unittest.TestCase):
    def test_existing_default_cooling_threshold_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            now=[0.];samples=iter([85.,84.,78.])
            guard=TemperatureGuard(Path(d)/'t.jsonl',sensor=lambda:next(samples),clock=lambda:now[0],sleep=lambda x:now.__setitem__(0,now[0]+x))
            guard.check()
            self.assertEqual(guard.last_temperature,78.)
            self.assertEqual(guard.idle_seconds,4.)

    def test_ledger_cooling_threshold_and_poll_interval_are_applied(self):
        with tempfile.TemporaryDirectory() as d:
            now=[0.];samples=iter([85.,84.])
            guard=TemperatureGuard(Path(d)/'t.jsonl',sensor=lambda:next(samples),clock=lambda:now[0],sleep=lambda x:now.__setitem__(0,now[0]+x))
            guard.policy.update(resume_c=84,cooling_poll_seconds=.05)
            guard.check()
            self.assertEqual(guard.last_temperature,84.)
            self.assertEqual(guard.idle_seconds,.05)


if __name__=='__main__':unittest.main()
