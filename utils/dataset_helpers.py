
# ---------------------------------------------------------------------
# 1.  Dataset loader for “final-token DPO”
# ---------------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Collection, Optional
from datasets import load_dataset


def load_tdpo_dataset(
    path: Path,
    tokenizer,
    *,
    max_seq_len: int = 4096,
    rule_reg_strength: float = 0.0,
    stop_words: Optional[Collection[str]] = None,
):
    """
    Read a TDPO jsonl file → HF Dataset of prompt-ids & final-token ids.
    If rule_reg_strength > 0, re-sample the examples so that over-frequent
    validator.rules are down-weighted.  Output size stays identical.

    Additional rule: drop any sample whose *rejected* suffix (after
    whitespace-trimming) is a stop-word.
    """

    import random
    import numpy as np
    from collections import Counter

    # ── stop-word setup ----------------------------------------------------
    if stop_words is None:
        stop_words = {
            "the", "a", "an", "in", "on", "at", "by", "for", "to", "of", "and",
            "or", "but", "if", "then", "else", "when", "where", "how", "why",
            "what", "who", "whom", "this", "that", "these", "those", "is", "are",
            "was", "were", "be", "being", "been", "have", "has", "had", "do",
            "does", "did", "will", "would", "shall", "should", "can", "could",
            "may", "might", "must"
        }
    stop_words = set(w.lower() for w in stop_words)

    # ── raw load -----------------------------------------------------------
    ds = load_dataset("json", data_files=str(path), split="train") #.select(range(20000))

    # ── optional rule-balanced re-sample ----------------------------------
    if rule_reg_strength and rule_reg_strength > 0:
        rules = [
            ex["validator"]["rule"] if isinstance(ex.get("validator"), dict) else None
            for ex in ds
        ]
        counts = Counter(r for r in rules if r is not None)
        if counts:
            thresh = np.median(list(counts.values()))
            w_rule = {
                r: 1.0 if c <= thresh else (thresh / c) ** rule_reg_strength
                for r, c in counts.items()
            }
            w_example = [w_rule.get(r, 1.0) for r in rules]
            probs = np.asarray(w_example, dtype=np.float64)
            probs /= probs.sum()

            rng = np.random.default_rng(3407)            # reproducible
            idx = rng.choice(len(ds), size=len(ds), replace=False, p=probs)
            idx.sort()
            ds = ds.select(idx.tolist())

    # ── tokenisation / sanity checks --------------------------------------
    tokenizer.truncation_side = "left"

    def _tok(ex):
        prompt_ids = tokenizer(
            ex["context_with_chat_template"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        ).input_ids

        # tokenize the candidate suffixes *as-is* (could include the ▁ boundary)
        ch_ids = tokenizer(ex["chosen_decoded"],   add_special_tokens=False).input_ids
        rj_ids = tokenizer(ex["rejected_decoded"], add_special_tokens=False).input_ids

        # ── Reject rows where suffix is not exactly ONE token --------------
        if len(ch_ids) != 1 or len(rj_ids) != 1:
            _tok.multi_tok_rows += 1
            logger.error('! failed tokenisation -- ' + str(len(ch_ids)) + ' -- ' + str(len(rj_ids)) + ' -- ' +str(ex["chosen_decoded"]) + ' -- ' +str(ex["rejected_decoded"]))
            return {
                "prompt_ids":        [],
                "chosen_token_id":    0,
                "rejected_token_id":  0,
                "__valid":           False,
            }

        # ── New rule: rejected token must not be a stop-word ---------------
        # Regex patterns sometimes ban patterns beginning with a stop word.
        # This can cause issues with model coherence if we train away from those probabilities.
        # So we'll filter any of those out.
        # If you want to include these rules with full effect, you can comment this section out.
        rj_text = ex["rejected_decoded"].strip().lower()
        if rj_text in stop_words:
            return {
                "prompt_ids":        [],
                "chosen_token_id":    0,
                "rejected_token_id":  0,
                "__valid":           False,
            }
        # ──

        ok = (
            len(prompt_ids) + 1 <= max_seq_len
            and ch_ids and rj_ids
            and ch_ids[-1] != rj_ids[-1]
        )
        if not ok:                     # ← prompt too long, empty suffix, etc.
            return {
                "prompt_ids":        [],
                "chosen_token_id":    0,
                "rejected_token_id":  0,
                "__valid":           False,
            }

        return {
            "prompt_ids":       prompt_ids,
            "chosen_token_id":   ch_ids[-1],
            "rejected_token_id": rj_ids[-1],
            "__valid":           True,
        }

    _tok.multi_tok_rows = 0            # attribute needed before first call
    ds = ds.map(_tok, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: ex["__valid"]).remove_columns("__valid")

    if len(ds) == 0:
        raise ValueError("no TDPO samples survived length / sanity checks")

    return ds.shuffle(seed=3407)



from pathlib import Path
from typing import Collection, Optional

def load_tdpo_multi_dataset(
    path: Path,
    tokenizer,
    *,
    max_seq_len: int = 4096,
    rule_reg_strength: float = 0.0,
    stop_words: Optional[Collection[str]] = None,
):
    """
    JSONL schema requirements
        context_with_chat_template : str
        multi_chosen_decoded       : list[str]   (≥1 surface forms)
        rejected_decoded           : str
        validator.rule             : optional str  (for re-weighting)

    Output HF Dataset columns
        prompt_ids      : List[int]
        chosen_ids      : List[int]  (variable length ≥1)
        rejected_id     : int
    """
    import random, numpy as np, json
    from collections import Counter
    from datasets import load_dataset

    # ── stop-word filter (same as single-token path) ─────────────────────
    if stop_words is None:
        stop_words = {
            "the","a","an","in","on","at","by","for","to","of","and","or","but",
            "if","then","else","when","where","how","why","what","who","whom",
            "this","that","these","those","is","are","was","were","be","being",
            "been","have","has","had","do","does","did","will","would","shall",
            "should","can","could","may","might","must"
        }
    stop_words = set(w.lower() for w in stop_words)

    ds = load_dataset("json", data_files=str(path), split="train")

    # ── optional rule-frequency re-sampling (identical logic) ────────────
    if rule_reg_strength and rule_reg_strength > 0:
        rules = [
            ex["validator"]["rule"] if isinstance(ex.get("validator"), dict) else None
            for ex in ds
        ]
        counts = Counter(r for r in rules if r is not None)
        if counts:
            thresh = np.median(list(counts.values()))
            w_rule = {
                r: 1.0 if c <= thresh else (thresh / c) ** rule_reg_strength
                for r, c in counts.items()
            }
            probs = np.asarray([w_rule.get(r, 1.0) for r in rules], dtype=np.float64)
            probs /= probs.sum()
            rng = np.random.default_rng(3407)
            idx = rng.choice(len(ds), size=len(ds), replace=False, p=probs)
            idx.sort()
            ds = ds.select(idx.tolist())

    tokenizer.truncation_side = "left"

    # ── tokenisation & validation ────────────────────────────────────────
    def _tok(ex):
        prompt_ids = tokenizer(
            ex["context_with_chat_template"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        ).input_ids

        # multi-chosen list
        chosen_surfaces = ex.get("multi_chosen_decoded") or []
        rejected_surface = ex["rejected_decoded"]

        # tokenise every candidate exactly once
        chosen_tok_ids = [
            tokenizer(t, add_special_tokens=False).input_ids for t in chosen_surfaces
        ]
        rejected_tok_ids = tokenizer(rejected_surface, add_special_tokens=False).input_ids

        # ── validation checks ────────────────────────────────────────────
        valid = (
            chosen_tok_ids
            and all(len(t) == 1 for t in chosen_tok_ids)   # each alt one token
            and len(rejected_tok_ids) == 1
            and rejected_surface.strip().lower() not in stop_words
            and len(prompt_ids) + 1 <= max_seq_len
        )
        if not valid:
            return {"__valid": False}

        flat_chosen = [t[0] for t in chosen_tok_ids]

        # don’t keep examples where rejected == one of the chosen
        if rejected_tok_ids[0] in flat_chosen:
            return {"__valid": False}

        return {
            "prompt_ids":  prompt_ids,
            "chosen_ids":  flat_chosen,       # variable-length list[int]
            "rejected_id": rejected_tok_ids[0],
            "__valid":     True,
        }

    ds = ds.map(_tok, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: ex["__valid"]).remove_columns("__valid")

    if len(ds) == 0:
        raise ValueError("no TDPO-MULTI samples survived length / sanity checks")

    return ds.shuffle(seed=3407)
