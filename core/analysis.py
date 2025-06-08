import json
import math
import re
import logging
from pathlib import Path
from collections import Counter
import pandas as pd
import nltk
import numpy as np
from typing import List, Tuple, Dict, Optional

# Assuming slop-forensics is in sys.path via main.py
from slop_forensics.slop_lists import extract_and_save_slop_phrases as _extract_slop_phrases
from slop_forensics import config as _sf_cfg # For SLOP_PHRASES_NGRAM_SIZE etc.
from slop_forensics.analysis import (
    get_word_counts, filter_mostly_numeric, merge_plural_possessive_s,
    filter_stopwords, filter_common_words, analyze_word_rarity,
    find_over_represented_words
)
# from slop_forensics.utils import load_jsonl_file, normalize_text, extract_words # Using local versions for now

logger = logging.getLogger(__name__)

# Local implementations of utils if slop_forensics.utils is problematic
# These should ideally come from the submodule if its structure allows easy import
def local_load_jsonl_file(file_path_str: str):
    data = []
    with open(file_path_str, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON line in {file_path_str}: {line.strip()}")
    return data

def local_normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
    text = re.sub(r"[\W_]+", " ", text)    # Replace non-alphanumeric with space
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
    return text

def local_extract_words(normalized_text: str, min_len: int):
    return [word for word in normalized_text.split() if len(word) >= min_len or "'" in word]


# --- Over-Represented Word Analysis ---
BOOST_EXPONENT = 0.75
ATTEN_EXPONENT = 0.75

def build_overrep_word_csv(
    texts: List[str],
    out_csv: Path,
    top_n_words_analysis: int,
    stop_words_set: Optional[set] = None,   # keeps the caller happy
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Notebook-faithful implementation that ALSO accepts *stop_words_set* so the
    CLI call `build_overrep_word_csv(..., stop_words_set=…)` keeps working.
    Returns (df, dict_words, nodict_words).
    """
    # ------------------------------------------------- plain-file logging ---
    log_path = out_csv.with_suffix(".log")

    def _log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")

    import datetime, traceback
    _log(f"build_overrep_word_csv ▶  {len(texts)} input texts")

    try:
        # ---------- flatten + count (identical to the notebook) -------------
        counts = get_word_counts(texts)          # ← **no extra kwargs**
        _log(f"after get_word_counts: {len(counts)} types")

        counts = filter_mostly_numeric(counts)
        counts = merge_plural_possessive_s(counts)
        counts = filter_stopwords(counts)

        _log(f"after filters: {len(counts)} types")

        # ---------- rarity + over-rep score ---------------------------------
        corpus_freqs, wf_freqs, *_ = analyze_word_rarity(counts)
        overrep = find_over_represented_words(
            corpus_freqs, wf_freqs, top_n=top_n_words_analysis
        )
        _log(f"find_over_represented_words → {len(overrep)} rows")

        # ---------- DataFrame ----------------------------------------------
        df = pd.DataFrame(
            overrep,
            columns=[
                "word", "ratio_corpus/wordfreq", "corpus_freq", "wordfreq_freq"
            ],
        )
        num_cols = ["ratio_corpus/wordfreq", "corpus_freq", "wordfreq_freq"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # ---------- modulated_score for dictionary words --------------------
        dict_mask = df["wordfreq_freq"] > 0
        if dict_mask.any():
            df_dict   = df[dict_mask].copy()
            boost     = np.power(df_dict["corpus_freq"],  BOOST_EXPONENT)
            atten     = np.power(df_dict["wordfreq_freq"], ATTEN_EXPONENT)
            atten_safe = np.where(atten == 0, 1, atten)
            df.loc[dict_mask, "modulated_score"] = (
                df_dict["ratio_corpus/wordfreq"] * boost / atten_safe
            )

        # ---------- write CSV ----------------------------------------------
        df.to_csv(out_csv, index=False)
        _log(f"CSV written → {out_csv}  ({len(df)} rows)")

        # ---------- split & sort -------------------------------------------
        dict_words_df = df[dict_mask]
        dict_words = (
            dict_words_df.sort_values(
                "modulated_score", ascending=False)["word"].tolist()
            if "modulated_score" in dict_words_df.columns
            else dict_words_df["word"].tolist()
        )
        nodict_words = df[~dict_mask]["word"].tolist()
        _log(f"returning {len(dict_words)} dict words, {len(nodict_words)} non-dict")

        return df, dict_words, nodict_words

    except Exception as exc:
        _log("ERROR:\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        raise


def select_overrep_words_for_ban(dict_words: list[str],
                                 nodict_words: list[str],
                                 is_first_iteration: bool,
                                 config: dict,
                                 *,
                                 whitelist: set[str]) -> list[str]:
    if is_first_iteration:
        dict_q, nodict_q = config['dict_overrep_initial'], config['nodict_overrep_initial']
    else:
        dict_q, nodict_q = config['dict_overrep_subsequent'], config['nodict_overrep_subsequent']

    selected = []
    for w in dict_words:
        if len(selected) >= dict_q: break
        if w.lower() not in whitelist: selected.append(w)
    for w in nodict_words:
        if len(selected) >= dict_q + nodict_q: break
        if w.lower() not in whitelist: selected.append(w)
    logger.info(f"Selected {len(selected)} over-rep words for ban ({dict_q}/{nodict_q} quotas).")
    return selected


# --- Slop Phrase Banning ---
def update_banned_slop_phrases(
    json_path: Path,
    texts: list[str],
    how_many_new: int,
    tmp_dir: Path,
    config: dict,
    *,
    whitelist: set[str],
    over_represented_words: Optional[list[str]] = None,
) -> None:
    """
    Unchanged logic EXCEPT: any candidate phrase that contains a
    *whitelisted word (case-insensitive)* is skipped, and the final
    merged list drops any legacy items that are now whitelisted.
    """
    logger.info(f"Updating slop-phrase ban list ({json_path.name}) …")

    def is_whitelisted(phrase: str) -> bool:
        return any(w == phrase.lower() for w in whitelist)

    # --------------------------------------------------------------- #
    # 1. run extractor (identical to previous body)                   #
    # --------------------------------------------------------------- #
    from slop_forensics.slop_lists import extract_and_save_slop_phrases as _extract
    from slop_forensics import config as _sf_cfg
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _extract(
        texts=texts,
        output_dir=tmp_dir,
        n=_sf_cfg.SLOP_PHRASES_NGRAM_SIZE,
        top_k_ngrams=max(1000, how_many_new * 5),
        top_phrases_to_save=max(how_many_new * 3, 100),
        chunksize=_sf_cfg.SLOP_PHRASES_CHUNKSIZE,
    )

    cand_phrases: List[str] = []
    try:
        with (tmp_dir / "slop_list_phrases.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                phrase = item[0] if isinstance(item, list) else str(item)
                if phrase and not is_whitelisted(phrase):
                    cand_phrases.append(phrase)
    except FileNotFoundError:
        pass

    # --------------------------------------------------------------- #
    # 2. merge with existing                                          #
    # --------------------------------------------------------------- #
    existing: set[str] = set()
    if json_path.exists():
        try:
            for entry in json.loads(json_path.read_text("utf-8")):
                p = entry[0] if isinstance(entry, list) else str(entry)
                if p and not is_whitelisted(p):
                    existing.add(p)
        except Exception:
            pass

    # keep requested quota only
    cand_phrases = cand_phrases[:how_many_new]
    if over_represented_words and config.get("ban_overrep_words_in_phrase_list"):
        for w in over_represented_words:
            if w not in whitelist:
                existing.add(w)

    merged = sorted((existing | set(cand_phrases)) - whitelist)
    json_path.write_text(json.dumps([[p, 1] for p in merged], indent=2, ensure_ascii=False), "utf-8")
    logger.info(f"🚫 Slop-phrase ban list now {len(merged)} entries "
                f"(+{len(merged)-len(existing)} this iter)")


# --- N-Gram Analysis ---
def _convert_and_normalize_human_ngram_list(ngram_list_of_dicts, n_value: int):
    if not isinstance(ngram_list_of_dicts, list): return {}
    converted_dict = {}
    for item in ngram_list_of_dicts:
        if not isinstance(item, dict): continue
        ngram_str, frequency = item.get("ngram"), item.get("frequency")
        if ngram_str is None or frequency is None: continue
        try: freq_int = int(frequency)
        except ValueError: continue
        
        tokens = [t.lower() for t in nltk.word_tokenize(local_normalize_text(str(ngram_str))) if t.isalpha()]
        if len(tokens) == n_value:
            processed_ngram_key = " ".join(tokens)
            if processed_ngram_key:
                converted_dict[processed_ngram_key] = converted_dict.get(processed_ngram_key, 0) + freq_int
    return converted_dict

def norm_per_freq_denom(raw_count: int, char_total: float, freq_norm_denom: int) -> float:
    if char_total == 0: return 0.0 if raw_count == 0 else math.inf
    return (raw_count / char_total) * freq_norm_denom

def build_norm_dict(counter: Counter, char_total: float, top_k: int, freq_norm_denom: int):
    char_total_float = float(char_total)
    return {
        term: {"gen_count": counter[term], "gen_freq_norm": norm_per_freq_denom(counter[term], char_total_float, freq_norm_denom)}
        for term, _ in counter.most_common(top_k) if term
    }

def compare_to_human(gen_norm: dict, human_counts: dict, human_total_chars: float, freq_norm_denom: int):
    both, gen_only = {}, {}
    human_total_chars_float = float(human_total_chars)
    for term, data in gen_norm.items():
        if not term: continue
        if term in human_counts:
            h_raw_count = human_counts[term]
            h_freq_norm = norm_per_freq_denom(h_raw_count, human_total_chars_float, freq_norm_denom)
            gen_freq = data["gen_freq_norm"]
            ratio = math.inf if h_freq_norm == 0 and gen_freq > 0 else (gen_freq / h_freq_norm if h_freq_norm > 0 else (1.0 if gen_freq == 0 else 0.0) )
            both[term] = {**data, "human_count": h_raw_count, "human_freq_norm": h_freq_norm, "freq_ratio_gen/hu": ratio}
        else:
            gen_only[term] = {**data, "human_count": 0, "human_freq_norm": 0.0, "freq_ratio_gen/hu": math.inf if data["gen_freq_norm"] > 0 else 0.0}
    return both, gen_only

def _is_refusal(rec: dict) -> bool:
    """
    Returns True if this JSONL record represents a refused / skipped prompt.
    Recognises all variants produced by main.py / auto_unslop.py.
    """
    if rec.get("refusal_detected") is True:
        return True
    status = rec.get("status", "").lower()
    if status in {"skipped"}:
        return True
    if status == "failed" and isinstance(rec.get("error"), str):
        err = rec["error"].lower()
        if err.startswith("refusal detected") or err.startswith("skipped -- prior refusal"):
            return True
    return False

def analyze_iteration_outputs_core(
    generated_jsonl_path: Path, human_profile_full: dict, 
    iter_analysis_output_dir: Path, config: dict, stop_words_set: set
):
    logger.info(f"--- Analyzing Outputs for {generated_jsonl_path.name} ---")
    iter_analysis_output_dir.mkdir(parents=True, exist_ok=True)

    gen_rows = local_load_jsonl_file(str(generated_jsonl_path))

    gen_texts: List[str] = [
        rec["generation"]
        for rec in gen_rows
        if isinstance(rec, dict)
           and isinstance(rec.get("generation"), str)
           and rec["generation"].strip()               # non-empty
           and not _is_refusal(rec)                    # ⬅️  new guard
    ]
    
    if not gen_texts:
        logger.warning(f"No usable text in {generated_jsonl_path}. Skipping analysis.")
        return None, None, None, None, [], 0

    human_profile = human_profile_full.get('human-authored', human_profile_full.get(next(iter(human_profile_full), None))) # Robust key access
    if not human_profile: raise ValueError("Human profile data not found in JSON.")

    human_bigrams = _convert_and_normalize_human_ngram_list(human_profile.get("top_bigrams", []), 2)
    human_trigrams = _convert_and_normalize_human_ngram_list(human_profile.get("top_trigrams", []), 3)
    
    h_num_texts = human_profile.get("num_texts_analyzed", 0)
    h_avg_len = human_profile.get("avg_length", 0)
    h_chars_total = float(h_num_texts * h_avg_len)
    if h_chars_total == 0: logger.warning("Human total characters is 0. Ratios might be infinite.")

    total_chars = sum(len(txt) for txt in gen_texts)
    bigram_counter, trigram_counter = Counter(), Counter()

    min_word_len = config['min_word_len_for_analysis']

    for txt in gen_texts:
        tokens = [t.lower() for t in nltk.word_tokenize(local_normalize_text(txt)) if t.isalpha()]
        tokens_filtered = [tok for tok in tokens if tok not in stop_words_set and (len(tok) >= min_word_len or tok in {"it's", "i'm"})]
        bigram_counter.update(" ".join(bg) for bg in nltk.ngrams(tokens_filtered, 2) if all(bg))
        trigram_counter.update(" ".join(tg) for tg in nltk.ngrams(tokens_filtered, 3) if all(tg))

    freq_norm_denom = config['freq_norm_denom_for_analysis']
    gen_bigrams_norm = build_norm_dict(bigram_counter, float(total_chars), config['top_k_bigrams'], freq_norm_denom)
    gen_trigrams_norm = build_norm_dict(trigram_counter, float(total_chars), config['top_k_trigrams'], freq_norm_denom)

    bigrams_dict, bigrams_nondict = compare_to_human(gen_bigrams_norm, human_bigrams, h_chars_total, freq_norm_denom)
    trigrams_dict, trigrams_nondict = compare_to_human(gen_trigrams_norm, human_trigrams, h_chars_total, freq_norm_denom)

    df_bi_dict = pd.DataFrame.from_dict(bigrams_dict, orient="index").rename_axis('ngram').reset_index()
    df_bi_nondct = pd.DataFrame.from_dict(bigrams_nondict, orient="index").rename_axis('ngram').reset_index()
    df_tri_dict = pd.DataFrame.from_dict(trigrams_dict, orient="index").rename_axis('ngram').reset_index()
    df_tri_nondct = pd.DataFrame.from_dict(trigrams_nondict, orient="index").rename_axis('ngram').reset_index()
    
    for df, sort_col in [(df_bi_dict, "freq_ratio_gen/hu"), (df_tri_dict, "freq_ratio_gen/hu")]:
        if not df.empty and sort_col in df.columns: df.sort_values(by=sort_col, ascending=False, inplace=True)
    for df, sort_col in [(df_bi_nondct, "gen_freq_norm"), (df_tri_nondct, "gen_freq_norm")]:
        if not df.empty and sort_col in df.columns: df.sort_values(by=sort_col, ascending=False, inplace=True)

    df_bi_dict.to_csv(iter_analysis_output_dir / "bigrams__dictionary_sorted.csv", index=False)
    df_bi_nondct.to_csv(iter_analysis_output_dir / "bigrams__non_dictionary_sorted.csv", index=False)
    df_tri_dict.to_csv(iter_analysis_output_dir / "trigrams__dictionary_sorted.csv", index=False)
    df_tri_nondct.to_csv(iter_analysis_output_dir / "trigrams__non_dictionary_sorted.csv", index=False)
    logger.info(f"N-gram analysis CSVs written to {iter_analysis_output_dir.resolve()}")

    return df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct, gen_texts, total_chars


def update_banned_ngrams_list(
    banned_ngrams_json_path: Path,
    dfs: list,                                  # bi/tri, dict / non-dict
    is_first_iteration: bool,
    config: dict,
    *,
    whitelist: set[str],
):
    newly: set[str] = set()
    def _take(df, n):                              # helper
        return {
            row["ngram"] for _, row in (df.head(n)).iterrows()
            if "ngram" in row and row["ngram"] and row["ngram"] not in whitelist
        } if df is not None and not df.empty and n > 0 else set()

    if is_first_iteration:
        quotas = (
            config['dict_bigrams_initial'], config['nodict_bigrams_initial'],
            config['dict_trigrams_initial'], config['nodict_trigrams_initial'],
        )
    else:
        quotas = (
            config['dict_bigrams_subsequent'], config['nodict_bigrams_subsequent'],
            config['dict_trigrams_subsequent'], config['nodict_trigrams_subsequent'],
        )
    for df, q in zip(dfs, quotas):
        newly |= _take(df, q)

    newly |= {s for s in config.get('extra_ngrams_to_ban', []) if s not in whitelist}

    current = set()
    if banned_ngrams_json_path.exists():
        try:
            current = set(json.loads(banned_ngrams_json_path.read_text("utf-8")))
        except Exception:
            pass

    final = sorted((current | newly) - whitelist)
    banned_ngrams_json_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), "utf-8")
    logger.info(f"📄 N-gram ban list updated → {banned_ngrams_json_path} "
                f"(+{len(final)-len(current)} new, total {len(final)})")


# --- Metrics Calculation ---
def calculate_lexical_diversity_stats(gen_texts: list, min_word_len: int):
    if not gen_texts: return 0.0, 0.0
    all_words = []
    for text in gen_texts:
        tokens = [t.lower() for t in nltk.word_tokenize(local_normalize_text(text)) if t.isalpha() and (len(t) >= min_word_len or t in {"a", "i"})]
        all_words.extend(tokens)
    if not all_words: return 0.0, 0.0
    num_tokens, num_types = len(all_words), len(set(all_words))
    ttr = num_types / num_tokens if num_tokens > 0 else 0.0
    rttr = num_types / math.sqrt(num_tokens) if num_tokens > 0 else 0.0
    return ttr, rttr

def calculate_repetition_score(gen_texts: list, total_chars: int, iteration_dfs: list, config: dict, stop_words_set: set):
    if not gen_texts or total_chars == 0: return 0.0
    
    target_ngrams = set()
    top_n_rep = config['top_n_repetition_stat']
    min_word_len = config['min_word_len_for_analysis']
    freq_norm_denom = config['freq_norm_denom_for_analysis']

    for df in iteration_dfs: # df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct
        if df is not None and not df.empty and 'ngram' in df.columns:
            target_ngrams.update(df.head(top_n_rep)['ngram'].tolist())
    if not target_ngrams: return 0.0

    total_repetition_instances = 0
    for text in gen_texts:
        tokens_all = [t.lower() for t in nltk.word_tokenize(local_normalize_text(text)) if t.isalpha()]
        tokens = [tok for tok in tokens_all if tok not in stop_words_set and (len(tok) >= min_word_len or tok in {"it's", "i'm"})]
        
        current_bigrams = [" ".join(bg) for bg in nltk.ngrams(tokens, 2) if all(bg)]
        current_trigrams = [" ".join(tg) for tg in nltk.ngrams(tokens, 3) if all(tg)]
        for bg in current_bigrams:
            if bg in target_ngrams: total_repetition_instances += 1
        for tg in current_trigrams:
            if tg in target_ngrams: total_repetition_instances += 1
            
    return norm_per_freq_denom(total_repetition_instances, float(total_chars), freq_norm_denom)