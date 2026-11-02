"""Automatic resumption is limited to a verified wall-budget stop."""
from pathlib import Path
import tempfile
import unittest
from experiments.nanbeige_receipt_replay.io import save
from .finish_nanbeige_skill_comparison import wall_resume_allowed


class FinisherTests(unittest.TestCase):
    def test_only_wall_stop_is_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save(root/'manifest.json',dict(status='interrupted',records=1))
            save(root/'runtime/worker_stopped.json',dict(stopped=True))
            for error,expected in [('RuntimeError: RuntimeError: Guard wall budget reached',True),
                                   ('RuntimeError: Temperature hard-stop threshold reached',False),
                                   ('RuntimeError: Temperature unavailable; stopping decoding',False),
                                   ('PermissionError: Locked',False),(None,False)]:
                save(root/'runtime/server_stopped.json',dict(stopped=True,guard_error=error))
                self.assertEqual(wall_resume_allowed(root),expected)

    def test_no_progress_or_unverified_cleanup_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);save(root/'manifest.json',dict(status='interrupted',records=0))
            self.assertFalse(wall_resume_allowed(root))
            save(root/'manifest.json',dict(status='interrupted',records=1))
            save(root/'runtime/worker_stopped.json',dict(stopped=False))
            save(root/'runtime/server_stopped.json',dict(stopped=True,guard_error='RuntimeError: Guard wall budget reached'))
            self.assertFalse(wall_resume_allowed(root))


if __name__=='__main__':unittest.main()
