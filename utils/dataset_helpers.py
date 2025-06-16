# utils/dataset_helpers.py
# ------------------------------------------------------------------
# Parallel FTPO loader (multi-chosen schema, with dual regularisation
# and distribution diagnostics)
# ------------------------------------------------------------------
from __future__ import annotations
import logging, os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Collection, Optional

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_ftpo_multi_dataset(
    path: Path,
    tokenizer,
    *,
    max_seq_len: int = 4096,
    rejected_reg_strength: float = 0.0,      # balance *rejected* tokens
    chosen_reg_strength: float = 0.0,        # balance *chosen* tokens
    min_chosen_tokens: int = 1,              # floor on |chosen|
    max_train_examples: int | None = None,   # overall size cap
    stop_words: Optional[Collection[str]] = None,
    num_proc: int | None = None,
    batch_size: int = 512,                   # tokenizer batch size
):
    """
    Load, regularise and tokenise an FTPO dataset with extensive diagnostics.

    Diagnostics
    -----------
    Immediately after reading the raw JSONL the function logs the 20 most
    frequent *surface* forms of `rejected_decoded`, and logs the same list
    again after all regularisation / filtering but before tokenisation.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    RNG = np.random.default_rng(3407)

    def _median_threshold(counts: Counter[int], strength: float) -> dict[str, float]:
        if not counts or strength <= 0:
            return {}
        thresh = float(np.median(list(counts.values())))
        return {
            t: 1.0 if c <= thresh else (thresh / c) ** strength
            for t, c in counts.items()
        }

    def _log_top(counter: Counter[str], title: str) -> None:
        head = ", ".join(f"{tok!r}:{cnt}" for tok, cnt in counter.most_common(20))
        logger.info(f"[ftpo-loader] {title} top-20 → {head}")

    # ------------------------------------------------------------------
    # stop-word set (validation only — unchanged)
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
    # 0️⃣  raw load + stable shuffle
    # ------------------------------------------------------------------
    raw = load_dataset("json", data_files=str(path), split="train")
    raw = raw.shuffle(seed=3407)
    rows = list(raw)
    N0   = len(rows)
    if N0 == 0:
        raise ValueError(f"{path} contained no rows")

    # pre-reg counts
    rej_counts = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_counts, "BEFORE")

    # ------------------------------------------------------------------
    # 1️⃣  quotas for rejected tokens (NO data change yet)
    # ------------------------------------------------------------------
    rej_weights = _median_threshold(rej_counts, rejected_reg_strength)
    S = max_train_examples or N0
    denom = sum(rej_weights.get(t, 1.0) * c for t, c in rej_counts.items())
    target_quota = {
        t: int(round(S * rej_weights.get(t, 1.0) * c / denom))
        for t, c in rej_counts.items()
    }

    # ------------------------------------------------------------------
    # 2️⃣  chosen-token balancing  (in-place)
    # ------------------------------------------------------------------
    if chosen_reg_strength > 0:
        chosen_counts = Counter(tok
            for r in rows
            for tok in (r["multi_chosen_decoded"] or [])
        )
        chosen_w = _median_threshold(chosen_counts, chosen_reg_strength)

        for r in rows:
            kept = [
                tok for tok in (r["multi_chosen_decoded"] or [])
                if RNG.random() < chosen_w.get(tok, 1.0)
            ]
            r["multi_chosen_decoded"] = kept

    # ------------------------------------------------------------------
    # 3️⃣  drop rows with too few chosen alts
    # ------------------------------------------------------------------
    if min_chosen_tokens > 1:
        rows = [r for r in rows
                if len(r["multi_chosen_decoded"]) >= min_chosen_tokens]
    if not rows:
        raise ValueError("all rows were removed by min_chosen_tokens filter")

    # ------------------------------------------------------------------
    # 4️⃣  sample rows to honour rejected-token quotas
    # ------------------------------------------------------------------
    accepted, taken = [], defaultdict(int)
    RNG.shuffle(rows)

    for r in rows:
        tok = r["rejected_decoded"]
        if taken[tok] < target_quota.get(tok, 0):
            accepted.append(r)
            taken[tok] += 1
        if len(accepted) >= S and all(taken[t] >= target_quota[t] for t in target_quota):
            break

    rows = accepted
    _log_top(Counter(r["rejected_decoded"] for r in rows), "AFTER")
    logger.info(f"[ftpo-loader] kept {len(rows):,} / {N0:,} rows after regularisation")

    # ------------------------------------------------------------------
    # 5️⃣  convert to HF Dataset for vectorised tokenisation  (original logic)
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
            prompt_tok,
            batch["multi_chosen_decoded"],
            batch["rejected_decoded"],
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
                out_prompt.append([0])
                out_chosen.append([0])
                out_rej.append(0)

        return {
            "prompt_ids":        out_prompt,
            "chosen_ids":        out_chosen,
            "rejected_token_id": out_rej,
            "__valid":           out_valid,
        }

    ds = ds.map(
        _tok,
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="tokenising",
    )
    ds = ds.filter(lambda ex: ex["__valid"], num_proc=num_proc, desc="filter")
    ds = ds.remove_columns("__valid")

    if len(ds) == 0:
        raise ValueError("no ftpo samples survived length / sanity checks")

    return ds.shuffle(seed=3407)
