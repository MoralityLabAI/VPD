import unittest
import numpy as np

from experiments.nanbeige_looped.capture import RepeatedForwardCapture


class SharedAffine:
    """Small recurrent computation with the forward-hook interface."""
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        owner = self
        class Handle:
            def remove(self):
                owner.hooks.remove(callback)
        return Handle()

    def __call__(self, x):
        output = x * 2 + 1
        for hook in tuple(self.hooks):
            hook(self, (x,), output)
        return output


class RepeatedCaptureTests(unittest.TestCase):
    def test_recurrence_retains_distinct_passes_without_changing_result(self):
        layer = SharedAffine()
        x = np.array([1.0, 3.0])
        reference = layer(layer(x))
        with RepeatedForwardCapture({"shared": layer}, expected_calls=2, extract=np.copy) as capture:
            actual = layer(layer(x))
        np.testing.assert_array_equal(reference, actual)
        np.testing.assert_array_equal(capture.values[("shared", 0)], [3, 7])
        np.testing.assert_array_equal(capture.values[("shared", 1)], [7, 15])
        self.assertEqual(layer.hooks, [])

    def test_incomplete_execution_and_accidental_generation_are_rejected(self):
        for count in (1, 3):
            layer = SharedAffine()
            with self.assertRaises(RuntimeError):
                with RepeatedForwardCapture({"shared": layer}, expected_calls=2, extract=np.copy):
                    for _ in range(count):
                        layer(np.array([1.0]))
            self.assertEqual(layer.hooks, [])

    def test_multiple_modules_keep_separate_call_counts(self):
        a, b = SharedAffine(), SharedAffine()
        with RepeatedForwardCapture({"a": a, "b": b}, expected_calls=2, extract=np.copy) as capture:
            x = np.array([0.0])
            for _ in range(2):
                x = b(a(x))
        self.assertEqual(capture.counts, {"a": 2, "b": 2})
        self.assertEqual(len(capture.values), 4)
        self.assertEqual(capture.values[("a", 1)].item(), 7)
        self.assertEqual(capture.values[("b", 1)].item(), 15)


if __name__ == "__main__":
    unittest.main()
