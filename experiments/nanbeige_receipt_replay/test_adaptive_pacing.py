"""Pacing bands must leave the existing thermal stops in control."""
import tempfile
from pathlib import Path
import unittest
from .adaptive_pacing import AdaptivePacingGuard


class PacingTests(unittest.TestCase):
    def guard(self,temperatures):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        values=iter(temperatures);self.now=0.
        def sleep(seconds):self.now+=seconds
        return AdaptivePacingGuard(Path(folder.name)/'thermal.jsonl',sensor=lambda:next(values),
            clock=lambda:self.now,sleep=sleep,wall_seconds=100)

    def test_warming_reduces_duty_cycle(self):
        guard=self.guard([60,73.9,74,79.9,80,82.9,83,84.9])
        ratios=[]
        for _ in range(8):guard.check(force=True);ratios.append(guard.policy['idle_to_decode_ratio'])
        self.assertEqual(ratios,[.5,.5,1.,1.,2.,2.,4.,4.])

    def test_85_degree_pause_still_waits_for_cooling(self):
        guard=self.guard([85,84.5,84]+[84]*8);guard.policy['resume_c']=84
        guard.check(force=True)
        self.assertEqual(guard.cooling_pauses,1)
        self.assertEqual(guard.last_temperature,84)
        self.assertGreater(self.now,0)
        self.assertEqual(guard.policy['idle_to_decode_ratio'],4.)

    def test_reheating_restarts_stability_period(self):
        guard=self.guard([85,80,80,81,80]+[80]*8);guard.policy['resume_c']=80
        guard.check(force=True)
        self.assertGreaterEqual(self.now,4.75)
        self.assertEqual(guard.last_temperature,80)

    def test_hard_stop_during_stabilization(self):
        guard=self.guard([85,80,87]);guard.policy['resume_c']=80
        with self.assertRaisesRegex(RuntimeError,'hard-stop'):guard.check(force=True)

    def test_a_second_cooling_pause_requires_a_new_stability_period(self):
        guard=self.guard([85,80,85,80]+[80]*9);guard.policy['resume_c']=80
        guard.check(force=True)
        self.assertEqual(guard.cooling_pauses,2)
        self.assertGreaterEqual(self.now,6.5)

    def test_87_degree_hard_stop_unchanged(self):
        guard=self.guard([87])
        with self.assertRaisesRegex(RuntimeError,'hard-stop'):guard.check(force=True)

    def test_sensor_failure_stops_inference(self):
        guard=self.guard([])
        with self.assertRaisesRegex(RuntimeError,'Temperature unavailable'):guard.check(force=True)


if __name__=='__main__':unittest.main()
