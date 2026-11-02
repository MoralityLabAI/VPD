from __future__ import annotations

import json
import unittest
from pathlib import Path

from small_control_harness.crypto import sign_grant, verify_grant_signature


class CrossLanguageCryptoVectorTests(unittest.TestCase):
    def test_typescript_python_hmac_vector(self) -> None:
        grant = json.loads(
            (Path(__file__).resolve().parents[1] / "fixtures" / "cross_language_grant.json").read_text(
                encoding="utf-8"
            )
        )
        expected = grant["signature"]["value"]
        self.assertEqual(sign_grant(grant, b"local-development-key"), expected)
        self.assertTrue(verify_grant_signature(grant, b"local-development-key"))


if __name__ == "__main__":
    unittest.main()
