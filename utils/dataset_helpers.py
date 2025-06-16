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

    # ────────────────────────────────────────────────────────────────
    # 1️⃣  Capture ORIGINAL rejected-token distribution & ratios
    #     (no rows removed, no chosen trimming yet)
    # ────────────────────────────────────────────────────────────────
    rej_cts_orig = Counter(r["rejected_decoded"] for r in rows)
    _log_top(rej_cts_orig, "PRE-NORMALISATION")

    # convert to fractional “weights” via median-threshold regularisation
    med = float(np.median(list(rej_cts_orig.values())))
    w_rej = {tok: 1.0 if c <= med else (med / c) ** rejected_reg_strength
            for tok, c in rej_cts_orig.items()}

    # normalised ratios we *want* to keep in the final dataset
    total_weighted = sum(w_rej[t] * c for t, c in rej_cts_orig.items())
    ratio_rej = {tok: (w_rej[tok] * cnt) / total_weighted
                for tok, cnt in rej_cts_orig.items()}

    # ────────────────────────────────────────────────────────────────
    # 2️⃣  Chosen-token trimming  (build quotas *before* we cut)
    # ────────────────────────────────────────────────────────────────
    chosen_cts_orig = Counter(tok
                            for r in rows
                            for tok in (r["multi_chosen_decoded"] or []))

    _log_top(Counter(r["rejected_decoded"] for r in rows), "PRE-CHOSEN")

    w_chosen = {tok: 1.0 if c <= np.median(list(chosen_cts_orig.values()))
                    else (np.median(list(chosen_cts_orig.values())) / c) ** chosen_reg_strength
                for tok, c in chosen_cts_orig.items()}

    tgt_chosen = {tok: int(round(c * w_chosen.get(tok, 1.0)))
                for tok, c in chosen_cts_orig.items()}

    # --- delete surplus chosen tokens --------------------------------
    from collections import defaultdict
    occ = defaultdict(list)                      # tok → [(row_i, pos)]
    for i, r in enumerate(rows):
        for p, tok in enumerate(r["multi_chosen_decoded"] or []):
            occ[tok].append((i, p))

    for tok, all_pos in occ.items():
        surplus = len(all_pos) - tgt_chosen.get(tok, len(all_pos))
        if surplus > 0:
            rng.shuffle(all_pos)
            for row_i, pos in all_pos[:surplus]:
                rows[row_i]["multi_chosen_decoded"][pos] = None

    for r in rows:
        r["multi_chosen_decoded"] = [t for t in r["multi_chosen_decoded"] if t is not None]

    _log_top(Counter(r["rejected_decoded"] for r in rows), "POST-CHOSEN")

    # ────────────────────────────────────────────────────────────────
    # 3️⃣  Apply min_chosen_tokens row filter
    # ────────────────────────────────────────────────────────────────
    rows = [r for r in rows if len(r["multi_chosen_decoded"]) >= min_chosen_tokens]

    _log_top(Counter(r["rejected_decoded"] for r in rows), "POST-MIN-FILTER")

    # ────────────────────────────────────────────────────────────────
    # 4️⃣  Row-level quota sampling **now** that trimming & filtering
    #     are done.  Scale the original ratios to the remaining size.
    # ────────────────────────────────────────────────────────────────
    N_final = max_train_examples or len(rows)
    target_rows = {tok: int(round(ratio_rej[tok] * N_final))
                for tok in ratio_rej}

    rng.shuffle(rows)
    selected, seen = [], defaultdict(int)
    for r in rows:
        tok = r["rejected_decoded"]
        if seen[tok] < target_rows.get(tok, 0):
            selected.append(r)
            seen[tok] += 1
        if len(selected) >= N_final:
            break
    rows = selected

    _log_top(Counter(r["rejected_decoded"] for r in rows), "AFTER-SAMPLING")
    logger.info("[ftpo-loader] kept %d rows after quota sampling", len(rows))


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
