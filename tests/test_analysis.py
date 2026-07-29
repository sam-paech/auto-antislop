import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "slop-forensics"))

from core.analysis import select_overrep_words_for_ban


class SelectOverrepWordsForBanTests(unittest.TestCase):
    def test_unused_dictionary_quota_does_not_spill_into_non_dictionary_pool(self):
        config = {
            "dict_overrep_initial": 10,
            "nodict_overrep_initial": 2,
            "dict_overrep_subsequent": 1,
            "nodict_overrep_subsequent": 1,
        }

        with self.assertLogs("core.analysis", level="INFO") as logs:
            selected = select_overrep_words_for_ban(
                ["dict-a", "dict-b", "dict-c"],
                ["nodict-a", "nodict-b", "nodict-c", "nodict-d", "nodict-e"],
                True,
                config,
                whitelist=set(),
            )

        self.assertEqual(
            selected,
            ["dict-a", "dict-b", "dict-c", "nodict-a", "nodict-b"],
        )
        self.assertIn(
            "Selected 3 dict + 2 non-dict over-rep words for ban "
            "(quotas 10/2; pools 3/5).",
            logs.output[0],
        )


if __name__ == "__main__":
    unittest.main()
