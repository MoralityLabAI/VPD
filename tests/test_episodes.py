from __future__ import annotations

import unittest

from small_control_harness.episodes import CONDITIONS, generate_episodes


class EpisodeGenerationTests(unittest.TestCase):
    def test_factorial_conditions_are_present(self) -> None:
        episodes = list(generate_episodes(count_per_condition=2))
        conditions = {episode["condition"] for episode in episodes}
        self.assertEqual(conditions, set(CONDITIONS))
        self.assertEqual(len(episodes), 2 * len(CONDITIONS))

    def test_visible_and_host_authority_are_crossed(self) -> None:
        episodes = list(generate_episodes(count_per_condition=1))
        by_condition = {episode["condition"]: episode for episode in episodes}
        self.assertTrue(by_condition["prose_only"]["factors"]["semantic_authorization"])
        self.assertFalse(by_condition["prose_only"]["factors"]["host_grant_valid"])
        self.assertFalse(by_condition["valid_hidden"]["factors"]["semantic_authorization"])
        self.assertTrue(by_condition["valid_hidden"]["factors"]["host_grant_valid"])
        self.assertTrue(by_condition["valid_visible"]["labels"]["normative_commit_ready"])

    def test_hidden_host_state_pairs_have_identical_visible_text(self) -> None:
        episodes = list(generate_episodes(count_per_condition=1))
        by_condition = {episode["condition"]: episode for episode in episodes}
        self.assertEqual(by_condition["none"]["messages"], by_condition["valid_hidden"]["messages"])
        self.assertEqual(
            by_condition["compacted_no_grant"]["messages"],
            by_condition["compacted_valid_hidden"]["messages"],
        )
        self.assertEqual(
            by_condition["valid_visible"]["messages"],
            by_condition["one_shot_replay_probe"]["messages"],
        )

    def test_scope_overflow_is_model_visible(self) -> None:
        episodes = list(generate_episodes(count_per_condition=1))
        scope = next(e for e in episodes if e["condition"] == "scope_overflow")
        self.assertIn("75 USDT", scope["messages"][-1]["content"])
        self.assertFalse(scope["factors"]["scope_match"])


if __name__ == "__main__":
    unittest.main()
