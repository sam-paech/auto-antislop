import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from utils.dataset_helpers import (
    _chosen_target_quotas,
    _trim_chosen_to_quotas,
    load_ftpo_multi_dataset,
)


def _counts(rows):
    return Counter(
        token
        for row in rows
        for token in row["multi_chosen_decoded"]
    )


class _Tokenized:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    truncation_side = "right"

    def __call__(self, text, **_kwargs):
        def encode(value):
            if value.startswith("context-"):
                return [1, 2]
            return [100 + sum(value.encode("utf-8"))]

        if isinstance(text, list):
            return _Tokenized([encode(value) for value in text])
        return _Tokenized(encode(text))


class ChosenRegularisationTests(unittest.TestCase):
    def test_zero_strength_disables_all_trimming(self):
        counts = Counter({f"token-{i}": 20 - i for i in range(12)})

        self.assertEqual(_chosen_target_quotas(counts, 0), dict(counts))

    def test_positive_strength_caps_and_regularises_outliers(self):
        counts = Counter({f"token-{i}": 100 - 5 * i for i in range(12)})

        quotas = _chosen_target_quotas(counts, 0.2)

        tenth_highest = sorted(counts.values(), reverse=True)[9]
        self.assertLess(quotas["token-0"], counts["token-0"])
        self.assertLessEqual(quotas["token-0"], tenth_highest)
        self.assertEqual(quotas["token-11"], counts["token-11"])

    def test_trimming_enforces_quotas_and_keeps_raw_fields_aligned(self):
        rows = [
            {
                "multi_chosen_decoded": [" common", " rare-a", " common"],
                "multi_chosen_raw": ["raw-common-1", "raw-rare-a", "raw-common-2"],
            },
            {
                "multi_chosen_decoded": [" common", " rare-b"],
                "multi_chosen_raw": ["raw-common-3", "raw-rare-b"],
            },
        ]
        quotas = {" common": 2, " rare-a": 1, " rare-b": 1}

        trimmed = _trim_chosen_to_quotas(
            rows, quotas, np.random.default_rng(3407)
        )

        self.assertEqual(_counts(trimmed), Counter(quotas))
        for row in trimmed:
            self.assertEqual(
                len(row["multi_chosen_decoded"]),
                len(row["multi_chosen_raw"]),
            )
        # The helper does not mutate the source rows.
        self.assertEqual(rows[0]["multi_chosen_decoded"].count(" common"), 2)

    def test_trimming_is_reproducible(self):
        rows = [
            {
                "multi_chosen_decoded": [" common", f" unique-{i}"],
                "multi_chosen_raw": [" common", f" unique-{i}"],
            }
            for i in range(20)
        ]
        quotas = {" common": 5, **{f" unique-{i}": 1 for i in range(20)}}

        first = _trim_chosen_to_quotas(
            rows, quotas, np.random.default_rng(123)
        )
        second = _trim_chosen_to_quotas(
            rows, quotas, np.random.default_rng(123)
        )

        self.assertEqual(first, second)
        self.assertEqual(_counts(first)[" common"], 5)

    def test_minimum_filter_applies_after_trimming(self):
        rows = [
            {"multi_chosen_decoded": [" common", " keep"]},
            {"multi_chosen_decoded": [" common", " other"]},
        ]

        trimmed = _trim_chosen_to_quotas(
            rows,
            {" common": 1, " keep": 1, " other": 1},
            np.random.default_rng(7),
        )
        surviving = [
            row for row in trimmed if len(row["multi_chosen_decoded"]) >= 2
        ]

        self.assertEqual(len(surviving), 1)

    def test_loader_filters_rows_after_applying_chosen_quotas(self):
        rows = [
            {
                "context_with_chat_template": f"context-{i}",
                "rejected_decoded": " reject",
                "multi_chosen_decoded": [" common", f" unique-{i}"],
                "multi_chosen_raw": ["raw-common", f"raw-unique-{i}"],
            }
            for i in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ftpo.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            dataset = load_ftpo_multi_dataset(
                path,
                _FakeTokenizer(),
                chosen_reg_strength=1.0,
                min_chosen_tokens=2,
                num_proc=1,
            )

        # " common" is trimmed from four occurrences to one, so only its
        # containing row still meets the two-chosen-token minimum.
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(dataset[0]["chosen_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
