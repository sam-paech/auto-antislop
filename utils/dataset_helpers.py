# ------------------------------------------------------------------
# Parallel ftpo loader (single-token version)
# ------------------------------------------------------------------
from __future__ import annotations
import logging, random
from pathlib import Path
from collections import Counter
from typing import Collection, Optional
import os
import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_ftpo_multi_dataset(
    path: Path,
    tokenizer,
    *,
    max_seq_len: int = 4096,
    rule_reg_strength: float = 0.0,
    stop_words: Optional[Collection[str]] = None,
    num_proc: int | None = None,
    batch_size: int = 512,           # adjust to your memory/CPU budget
):
    """
    Parallel loader for the “multi-chosen” ftpo JSONL schema.

    Returns
    -------
    HF Dataset with columns
        prompt_ids        : List[int]
        chosen_ids        : List[int]   (≥1, variable length)
        rejected_token_id : int
    """
    # ── stop-word filter ────────────────────────────────────────────────
    if stop_words is None:
        stop_words = {
            "the","a","an","in","on","at","by","for","to","of","and","or","but",
            "if","then","else","when","where","how","why","what","who","whom",
            "this","that","these","those","is","are","was","were","be","being",
            "been","have","has","had","do","does","did","will","would","shall",
            "should","can","could","may","might","must"
        }
    stop_words = {w.lower() for w in stop_words}

    # ── raw load ────────────────────────────────────────────────────────
    ds = load_dataset("json", data_files=str(path), split="train")

    # ── optional rule-frequency re-sampling ────────────────────────────
    if rule_reg_strength and rule_reg_strength > 0:
        rules  = [ex["validator"]["rule"] if isinstance(ex.get("validator"), dict) else None
                  for ex in ds]
        counts = Counter(r for r in rules if r is not None)
        if counts:
            thresh = np.median(list(counts.values()))
            w_rule = {r: 1.0 if c <= thresh else (thresh / c) ** rule_reg_strength
                      for r, c in counts.items()}
            probs  = np.asarray([w_rule.get(r, 1.0) for r in rules], dtype=np.float64)
            probs /= probs.sum()
            rng = np.random.default_rng(3407)
            idx = rng.choice(len(ds), size=len(ds), replace=False, p=probs)
            idx.sort()
            ds = ds.select(idx.tolist())

    tokenizer.truncation_side = "left"
    num_proc = num_proc or os.cpu_count()

    # ── batched tokenisation & validation ───────────────────────────────
    def _tok(batch):
        out_prompt, out_chosen, out_rej, out_valid = [], [], [], []

        # vectorised prompt encoding
        prompt_tok = tokenizer(
            batch["context_with_chat_template"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        ).input_ids

        # per-row suffix handling
        for p_ids, chosen_surf, rej_surf in zip(
            prompt_tok,
            batch["multi_chosen_decoded"],
            batch["rejected_decoded"],
        ):
            # guarantee list-of-str for chosen
            chosen_surf = chosen_surf or []
            # tokenize suffixes (lists are short; overhead negligible)
            chosen_tok_ids = [
                tokenizer(t, add_special_tokens=False).input_ids for t in chosen_surf
            ]
            rej_tok_ids = tokenizer(rej_surf, add_special_tokens=False).input_ids

            valid = (
                chosen_tok_ids
                and all(len(t) == 1 for t in chosen_tok_ids)      # each alt one token
                and len(rej_tok_ids) == 1
                and rej_surf.strip().lower() not in stop_words
                and len(p_ids) + 1 <= max_seq_len
            )

            if valid:
                flat_chosen = [t[0] for t in chosen_tok_ids]
                # reject rows where rejection collides with any chosen token
                if rej_tok_ids[0] in flat_chosen:
                    valid = False

            out_valid.append(valid)
            if valid:
                out_prompt.append(p_ids)
                out_chosen.append(flat_chosen)
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