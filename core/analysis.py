import json
import math
import re
import logging
from pathlib import Path
from collections import Counter
import pandas as pd
import nltk
from typing import Optional

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

def build_overrep_word_csv(texts: list[str], out_csv: Path, top_n_words_analysis: int, stop_words_set: set):
    logger.info(f"Building over-represented word CSV for {len(texts)} texts, top_n={top_n_words_analysis}.")
    counts = get_word_counts(texts, normalize_func=local_normalize_text, extract_func=local_extract_words)
    counts = filter_mostly_numeric(counts)
    counts = merge_plural_possessive_s(counts)
    counts = filter_stopwords(counts, stop_words_set=stop_words_set)
    # filter_common_words might be too aggressive here if we want to catch common slop.
    # counts = filter_common_words(counts, top_n=500) # Optional

    corpus_freqs, wf_freqs, _, _, _ = analyze_word_rarity(counts) # Adjusted for tuple unpacking
    overrep = find_over_represented_words(corpus_freqs, wf_freqs, top_n=top_n_words_analysis)

    df = pd.DataFrame(overrep, columns=["word", "ratio_corpus/wordfreq", "corpus_freq", "wordfreq_freq"])
    num_cols = ["ratio_corpus/wordfreq", "corpus_freq", "wordfreq_freq"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    dict_mask = df["wordfreq_freq"] > 0
    if dict_mask.any():
        df_dict = df[dict_mask].copy()
        boost = df_dict["corpus_freq"].pow(BOOST_EXPONENT)
        atten = df_dict["wordfreq_freq"].pow(ATTEN_EXPONENT)
        modulated_score = df_dict["ratio_corpus/wordfreq"] * boost
        atten_safe = atten.replace(0, 1) # Avoid division by zero
        df.loc[dict_mask, "modulated_score"] = modulated_score / atten_safe
    
    df.to_csv(out_csv, index=False)
    logger.info(f"🔎 Over-represented word CSV -> {out_csv} ({len(df)} rows)")

    dict_words_df = df[dict_mask]
    dict_words = dict_words_df.sort_values("modulated_score", ascending=False)["word"].tolist() if "modulated_score" in dict_words_df else []
    nodict_words = df[~dict_mask]["word"].tolist()
    return df, dict_words, nodict_words

def select_overrep_words_for_ban(dict_words: list[str], nodict_words: list[str],
                                 is_first_iteration: bool, config: dict) -> list[str]:
    if is_first_iteration:
        dict_quota = config['dict_overrep_initial']
        nodict_quota = config['nodict_overrep_initial']
    else:
        dict_quota = config['dict_overrep_subsequent']
        nodict_quota = config['nodict_overrep_subsequent']
    selected = dict_words[:dict_quota] + nodict_words[:nodict_quota]
    logger.info(f"Selected {len(selected)} over-represented words for banning ({len(dict_words[:dict_quota])} dict, {len(nodict_words[:nodict_quota])} non-dict).")
    return selected

# --- Slop Phrase Banning ---
def update_banned_slop_phrases(json_path: Path, texts: list[str], how_many_new: int,
                               tmp_dir: Path, config: dict,
                               over_represented_words: Optional[list[str]] = None) -> None:
    logger.info(f"Updating slop phrase ban list: {json_path}, adding up to {how_many_new} new phrases.")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use constants from _sf_cfg if they are appropriate, or make them configurable
    _extract_slop_phrases(
        texts=texts,
        output_dir=tmp_dir,
        n=_sf_cfg.SLOP_PHRASES_NGRAM_SIZE,
        top_k_ngrams=max(1000, how_many_new * 5),
        top_phrases_to_save=max(how_many_new * 3, 100),
        chunksize=_sf_cfg.SLOP_PHRASES_CHUNKSIZE,
    )

    phrases_jsonl = tmp_dir / "slop_list_phrases.jsonl"
    new_phrases_from_file: list[str] = []
    if phrases_jsonl.exists():
        with phrases_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                    phrase, freq = (item[0], item[1]) if isinstance(item, list) and len(item) >= 2 else (str(item), 1)
                    if freq >= config['min_phrase_freq_to_keep']:
                        new_phrases_from_file.append(str(phrase))
                except (json.JSONDecodeError, IndexError, TypeError) as e:
                    logger.warning(f"Skipping malformed line in {phrases_jsonl}: {line.strip()} ({e})")
    
    existing_phrases_set: set[str] = set()
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            for entry in raw:
                existing_phrases_set.add(str(entry[0]) if isinstance(entry, list) and entry else str(entry))
        except Exception as exc:
            logger.warning(f"Could not read existing slop phrase ban list ({json_path}): {exc}")

    actually_new_phrases_to_add: list[str] = []
    for p in new_phrases_from_file:
        if len(actually_new_phrases_to_add) >= how_many_new:
            break
        if p not in existing_phrases_set:
            actually_new_phrases_to_add.append(p)

    merged_set: set[str] = existing_phrases_set.copy()
    if config.get('extra_slop_phrases_to_ban'):
        merged_set.update(config['extra_slop_phrases_to_ban'])
    
    num_added_from_file_candidates = len(merged_set)
    merged_set.update(actually_new_phrases_to_add)
    num_added_from_file = len(merged_set) - num_added_from_file_candidates

    num_added_from_overrep = 0
    if config['ban_overrep_words_in_phrase_list'] and over_represented_words:
        initial_merged_size = len(merged_set)
        merged_set.update(over_represented_words)
        num_added_from_overrep = len(merged_set) - initial_merged_size

    if not merged_set and not json_path.exists():
        logger.info(f"🚫 No slop phrases or over-represented words to ban. File not created: {json_path}")
        return

    merged_list_for_json = sorted([[phrase, 1] for phrase in merged_set], key=lambda x: x[0])
    json_path.write_text(json.dumps(merged_list_for_json, indent=2, ensure_ascii=False), encoding="utf-8")
    
    total_newly_added = num_added_from_file + num_added_from_overrep
    logger.info(
        f"🚫 Slop-phrase ban list updated -> {json_path} "
        f"(now {len(merged_list_for_json)} entries; "
        f"+{total_newly_added} new: {num_added_from_file} from phrases, {num_added_from_overrep} from overrep words)"
    )


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

def analyze_iteration_outputs_core(
    generated_jsonl_path: Path, human_profile_full: dict, 
    iter_analysis_output_dir: Path, config: dict, stop_words_set: set
):
    logger.info(f"--- Analyzing Outputs for {generated_jsonl_path.name} ---")
    iter_analysis_output_dir.mkdir(parents=True, exist_ok=True)

    gen_rows = local_load_jsonl_file(str(generated_jsonl_path))
    gen_texts = [row["generation"] for row in gen_rows if isinstance(row, dict) and isinstance(row.get("generation"), str)]
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


def update_banned_ngrams_list(banned_ngrams_json_path: Path, dfs: list,
                              is_first_iteration: bool, config: dict):
    newly = set()
    def _take(df, n): return set(df.head(n)['ngram'].tolist()) if df is not None and not df.empty and 'ngram' in df.columns and n > 0 else set()

    quotas = config
    if is_first_iteration:
        newly |= _take(dfs[0], quotas['dict_bigrams_initial'])
        newly |= _take(dfs[1], quotas['nodict_bigrams_initial'])
        newly |= _take(dfs[2], quotas['dict_trigrams_initial'])
        newly |= _take(dfs[3], quotas['nodict_trigrams_initial'])
    else:
        newly |= _take(dfs[0], quotas['dict_bigrams_subsequent'])
        newly |= _take(dfs[1], quotas['nodict_bigrams_subsequent'])
        newly |= _take(dfs[2], quotas['dict_trigrams_subsequent'])
        newly |= _take(dfs[3], quotas['nodict_trigrams_subsequent'])

    if config.get('extra_ngrams_to_ban'):
        newly |= set(config['extra_ngrams_to_ban'])

    current = []
    if banned_ngrams_json_path.exists():
        try: current = json.loads(banned_ngrams_json_path.read_text("utf-8"))
        except json.JSONDecodeError: current = []
        if not isinstance(current, list): current = []
    
    final_set = set(current) | newly
    final_list = sorted(list(final_set)) # Ensure it's a list of strings for JSON

    banned_ngrams_json_path.write_text(json.dumps(final_list, indent=2, ensure_ascii=False), "utf-8")
    added = len(final_list) - len(current)
    logger.info(f"📄 N-gram ban list updated -> {banned_ngrams_json_path} (+{added}, total {len(final_list)})")

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