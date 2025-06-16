# utils/dataset_helpers.py
from __future__ import annotations
import logging, os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Collection, Optional

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

# tokens we want to watch closely
_WATCH = [" nodded", " leaned"]


def load_ftpo_multi_dataset(
    path: Path,
    tokenizer,
    *,
    max_seq_len: int = 4096,
    rejected_reg_strength: float = 0.0,
    chosen_reg_strength: float = 0.0,
    min_chosen_tokens: int = 1,
    max_train_examples: int | None = None,
    stop_words: Optional[Collection[str]] = None,
    num_proc: int | None = None,
    batch_size: int = 512,
):
    """
    Parallel loader for “multi-chosen” FTPO JSONL with dual regularisation.
    Logs the counts of `_WATCH` tokens at every major stage.
    """

    if min_chosen_tokens < 1:
        min_chosen_tokens = 1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    rng = np.random.default_rng(3407)

    def _median_threshold(cts: Counter[str], strength: float) -> dict[str, float]:
        if not cts or strength <= 0:
            return {}
        med = float(np.median(list(cts.values())))
        return {t: 1.0 if c <= med else (med / c) ** strength for t, c in cts.items()}

    def _log_top(cts: Counter[str], what: str) -> None:
        head = ", ".join(f"{tok!r}:{cnt}" for tok, cnt in cts.most_common(20))
        logger.info(f"[ftpo-loader] {what} top-20 → {head}")
        logger.info(
            "          ↳ watch «%s»: %s   «%s»: %s",
            _WATCH[0], cts[_WATCH[0]],
            _WATCH[1], cts[_WATCH[1]],
        )

    # ------------------------------------------------------------------
    # stop-word list (unchanged)
    # ------------------------------------------------------------------
    if stop_words is None:
        stop_words = {
            "the","a","an","in","on","at","by","for","to","of","and","or","but",
            "if","then","else","when","where","how","why","what","who","whom",
            "this","that","these","those","is","are","was","were","be","being",
            "been","have","has","had","do","does","did","will","would","shall",
            "should","can","could","may","might","must"
        }
    stop_words = {w.lower() for w in stop_words}

    # ------------------------------------------------------------------
    # 0️⃣  raw load + shuffle
    # ------------------------------------------------------------------
    raw = load_dataset("json", data_files=str(path), split="train").shuffle(seed=3407)
    rows = list(raw)
    if not rows:
        raise ValueError(f"{path} contained no rows")

    rej_counts = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_counts, "BEFORE")

    # ------------------------------------------------------------------
    # 1️⃣  per-token quotas for rejected (no data mutation)
    # ------------------------------------------------------------------
    quotas = _median_threshold(rej_counts, rejected_reg_strength)
    S      = max_train_examples or len(rows)
    total  = sum(quotas.get(t, 1.0) * c for t, c in rej_counts.items())
    target = {t: int(round(S * quotas.get(t, 1.0) * c / total))
              for t, c in rej_counts.items()}

    # ------------------------------------------------------------------
    # 2️⃣  chosen-token down-sampling
    # ------------------------------------------------------------------
    rej_pre_chosen = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_pre_chosen, "PRE-CHOSEN")


    rows = [r for r in rows
            if len(r["multi_chosen_decoded"]) >= min_chosen_tokens]
    rej_debug = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_debug, "DEBUG")
    

    if chosen_reg_strength > 0:
        # 1) tally current chosen-token counts
        chosen_cts = Counter(tok
                            for r in rows
                            for tok in (r["multi_chosen_decoded"] or []))

        # 2) convert median-threshold weights → integer *target counts*
        w_chosen   = _median_threshold(chosen_cts, chosen_reg_strength)
        tgt_chosen = {tok: int(round(cnt * w_chosen.get(tok, 1.0)))
                    for tok, cnt in chosen_cts.items()}

        # 3) build an index of every occurrence (row-idx, pos-in-list)
        from collections import defaultdict
        occ_idx = defaultdict(list)          # tok → [(row_i, pos)]
        for i, r in enumerate(rows):
            for p, tok in enumerate(r["multi_chosen_decoded"] or []):
                occ_idx[tok].append((i, p))

        # 4) for tokens that exceed their quota, randomly blank out surplus
        for tok, occ in occ_idx.items():
            surplus = len(occ) - tgt_chosen.get(tok, len(occ))
            if surplus <= 0:
                continue
            rng.shuffle(occ)
            for row_i, pos in occ[:surplus]:
                rows[row_i]["multi_chosen_decoded"][pos] = None     # mark

        # 5) remove all marked (=None) tokens from every row
        for r in rows:
            r["multi_chosen_decoded"] = [
                tok for tok in r["multi_chosen_decoded"] if tok is not None
            ]

    # ------------------------------------------------------------------
    #  Debug: rejected-token distribution **after** chosen regularisation
    # ------------------------------------------------------------------
    rej_post_chosen = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_post_chosen, "POST-CHOSEN")


    # ------------------------------------------------------------------
    # 3️⃣  min-chosen filter
    # ------------------------------------------------------------------
    rows = [r for r in rows
            if len(r["multi_chosen_decoded"]) >= min_chosen_tokens]
    rej_after_chosen = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_after_chosen, "POST-CHOSEN")

    # ------------------------------------------------------------------
    # 4️⃣  quota sampling
    # ------------------------------------------------------------------
    rng.shuffle(rows)
    accepted, seen = [], defaultdict(int)
    for r in rows:
        tok = r["rejected_decoded"]
        if seen[tok] < target.get(tok, 0):
            accepted.append(r)
            seen[tok] += 1
        if len(accepted) >= S:
            break
    rows = accepted
    _log_top(Counter(r["rejected_decoded"] for r in rows), "AFTER")
    logger.info("[ftpo-loader] kept %s / %s rows", len(rows), len(raw))

    # ------------------------------------------------------------------
    # 5️⃣  tokenisation (unchanged section)
    # ------------------------------------------------------------------
    from datasets import Dataset
    ds = Dataset.from_list(rows)

    tokenizer.truncation_side = "left"
    num_proc = num_proc or max(1, int(os.cpu_count() / 4))

    def _tok(batch):
        out_prompt, out_chosen, out_rej, out_valid = [], [], [], []

        prompt_tok = tokenizer(
            batch["context_with_chat_template"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        ).input_ids

        for p_ids, chosen_surf, rej_surf in zip(
            prompt_tok, batch["multi_chosen_decoded"], batch["rejected_decoded"]
        ):
            chosen_surf = chosen_surf or []
            chosen_tok_ids = [tokenizer(t, add_special_tokens=False).input_ids
                              for t in chosen_surf]
            rej_tok_ids = tokenizer(rej_surf, add_special_tokens=False).input_ids

            valid = (
                chosen_tok_ids
                and all(len(t) == 1 for t in chosen_tok_ids)
                and len(rej_tok_ids) == 1
                and rej_surf.strip().lower() not in stop_words
                and len(p_ids) + 1 <= max_seq_len
            )
            if valid and rej_tok_ids[0] in [t[0] for t in chosen_tok_ids]:
                valid = False

            out_valid.append(valid)
            if valid:
                out_prompt.append(p_ids)
                out_chosen.append([t[0] for t in chosen_tok_ids])
                out_rej.append(rej_tok_ids[0])
            else:
                out_prompt.append([0]); out_chosen.append([0]); out_rej.append(0)

        return {
            "prompt_ids":        out_prompt,
            "chosen_ids":        out_chosen,
            "rejected_token_id": out_rej,
            "__valid":           out_valid,
        }

    ds = ds.map(
        _tok, batched=True, batch_size=batch_size,
        remove_columns=ds.column_names,
        num_proc=num_proc, desc="tokenising",
    )
    ds = ds.filter(lambda ex: ex["__valid"], num_proc=num_proc, desc="filter")
    ds = ds.remove_columns("__valid")
    if len(ds) == 0:
        raise ValueError("no ftpo samples survived length / sanity checks")

    return ds.shuffle(seed=3407)
