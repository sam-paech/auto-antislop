# %%
# %%
##################################
# Imports                        #
##################################
import json, subprocess, sys, math, re, itertools, collections, os, pathlib, datetime
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import nltk
import numpy as np

from slop_forensics.slop_lists import extract_and_save_slop_phrases as _extract_slop_phrases
from slop_forensics import config as _sf_cfg
from slop_forensics.analysis import (
    get_word_counts, filter_mostly_numeric, merge_plural_possessive_s,
    filter_stopwords, filter_common_words, analyze_word_rarity,
    find_over_represented_words
)
from wordfreq import word_frequency   # used inside the toolkit too


# %%
##################################
# Pipeline Parameters            #
##################################
RUN_PIPELINE        = True  # ⇠ set False to skip the full pipeline (e.g., if you only want to inspect results)
NUM_ITERATIONS      = 3     # Number of iterations to run

ENABLE_NGRAM_BAN    = True
TOP_N_INITIAL_BAN   = 300   # N-grams to ban from each of 4 lists in the first iteration
TOP_N_SUBSEQUENT_BAN= 100   # N-grams to ban from each of 4 lists in subsequent iterations
TOP_N_REPETITION_STAT = 50 # N-grams from each of 4 lists to track for repetition stats

# Parameters for main.py (generation)
THREADS             = 80    # Adjust based on your system
MAX_PROMPTS         = 2000  # Max prompts for generation, can be reduced for faster iterations
HF_DATASET_NAME     = 'Nitral-AI/Reddit-SFW-Writing_Prompts_ShareGPT'
HF_DATASET_SPLIT    = 'train'
LOGGING_LEVEL       = 'INFO'

# Parameters for N-gram analysis (within each iteration)
HUMAN_PROFILE_PATH  = Path('data/human_writing_profile.json')
TOP_K_WORDS         = 200_000
TOP_K_BIGRAMS       = 1_000
TOP_K_TRIGRAMS      = 1_000
MIN_WORD_LEN        = 4
FREQ_NORM_DENOM     = 100_000

# Params for slop phrase banning
COMPUTE_OVERREP_WORDS             = True    # create CSV each iter
ENABLE_SLOP_PHRASE_BAN            = True    # maintain banned-phrase JSON
BAN_OVERREP_WORDS_IN_PHRASE_LIST  = True    # ⬅ NEW: also add words to that list
DICT_OVERREP_INITIAL      = 400   # dictionary words (wf > 0) first iter
DICT_OVERREP_SUBSEQUENT   = 200   # dictionary words later iters
NODICT_OVERREP_INITIAL    =  80   # non-dictionary words (wf == 0) first iter
NODICT_OVERREP_SUBSEQUENT =  20   # non-dictionary words later iters
MIN_PHRASE_FREQ_TO_KEEP           = 2       # ⬅ NEW: only keep phrases seen > 1×


# how many slop phrases to (newly) ban each round
TOP_N_INITIAL_SLOP_BAN   = 200
TOP_N_SUBSEQUENT_SLOP_BAN    = 100

# where we’ll keep the growing list
BANNED_SLOP_PHRASES_FILE = "banned_slop_phrases.json"

# Output directory for the experiment
# (A timestamped subdirectory will be created under this)
EXPERIMENT_BASE_DIR = Path("results") / "iterative_antislop_experiment"


# Ensure nltk resources are available
def download_nltk_resource(resource_id, resource_name):
    try:
        nltk.data.find(resource_id)
        print(f"NLTK '{resource_name}' resource found.")
    except LookupError:
        print(f"NLTK '{resource_name}' resource not found. Downloading...")
        nltk.download(resource_name, quiet=True)
        print(f"NLTK '{resource_name}' resource downloaded.")
    except Exception as e:
        print(f"Warning: Could not automatically verify/download NLTK '{resource_name}' resource: {e}.")


download_nltk_resource('tokenizers/punkt', 'punkt')
download_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')
download_nltk_resource('corpora/stopwords', 'stopwords')

from nltk import ngrams
from nltk.corpus import stopwords

# Attempt to import from slop_forensics. If not found, provide stubs.
# In a real environment, ensure slop_forensics is in PYTHONPATH or installed.
try:
    from slop_forensics.utils import load_jsonl_file, normalize_text, extract_words
    print("Successfully imported from slop_forensics.utils")
except ImportError:
    print("Warning: slop_forensics.utils not found. Using placeholder functions.")
    print("Please ensure slop_forensics is installed or in your PYTHONPATH for full functionality.")
    
    def load_jsonl_file(file_path_str: str):
        data = []
        with open(file_path_str, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
        text = re.sub(r"[\W_]+", " ", text)    # Replace non-alphanumeric with space
        text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
        return text

    def extract_words(normalized_text: str, min_len: int):
        # This is a simplified version. The original might have more sophisticated logic.
        return [word for word in normalized_text.split() if len(word) >= min_len or "'" in word]

# Initialize STOP_WORDS
try:
    STOP_WORDS = set(stopwords.words('english'))
    print(f"Loaded {len(STOP_WORDS)} NLTK stopwords for 'english'.")
except LookupError:
    print(f"NLTK 'stopwords' not found even after download attempt. Stopword filtering will be limited.")
    STOP_WORDS = set()


# %%
##################################
# Helper Functions               #
##################################

###############################################################################
# RUN GENERATION SCRIPT — RESTORED N-GRAM BANNING #############################
###############################################################################
def run_generation_script(
    iter_idx: int,
    output_jsonl_path: Path,
    banned_ngrams_file_path: Path | None = None,
    *,
    slop_phrases_file_path: Path | None = None,
    top_n_slop_phrases: int | None = None,
) -> None:
    """
    Invoke main.py for a single iteration of the pipeline.

    Parameters
    ----------
    iter_idx : int
        0-based iteration counter – only used for console logs.
    output_jsonl_path : Path
        Destination of generated text from main.py.
    banned_ngrams_file_path : Path | None
        JSON list of N-grams to ban (classic anti-slop mechanism).
    slop_phrases_file_path : Path | None, keyword-only
        JSON list of slop phrases produced by Slop-Forensics.
    top_n_slop_phrases : int | None, keyword-only
        Value for main.py’s --top-n-slop-phrases flag.

    Raises
    ------
    subprocess.CalledProcessError
        If main.py returns a non-zero exit status.
    """

    # ---------------------------------------------------------------- build CLI
    cmd: list[str] = [
        sys.executable, "main.py",
        "--output-jsonl",     str(output_jsonl_path),
        "--input-hf-dataset", HF_DATASET_NAME,
        "--hf-dataset-split", HF_DATASET_SPLIT,
        "--threads",          str(THREADS),
        "--max-prompts",      str(MAX_PROMPTS),
        "--logging-level",    LOGGING_LEVEL,
    ]

    # ------ ALWAYS pass the N-gram ban list if supplied -----------------------
    if banned_ngrams_file_path is not None:
        # Warn if the caller handed us a path that doesn’t exist **yet**,
        # but still forward it – main.py can decide what to do.
        if not banned_ngrams_file_path.exists():
            print(f"⚠️  Warning: banned-N-gram file does not exist at call time: "
                  f"{banned_ngrams_file_path}")
        cmd += ["--ngram-banned-file", str(banned_ngrams_file_path)]

    # ------ Optional Slop-Forensics phrase banning ----------------------------
    if slop_phrases_file_path is not None:
        if not slop_phrases_file_path.exists():
            print(f"⚠️  Warning: slop-phrase file not found: "
                  f"{slop_phrases_file_path}")
        cmd += ["--slop-phrases-file", str(slop_phrases_file_path)]

    if top_n_slop_phrases is not None and top_n_slop_phrases > 0:
        cmd += ["--top-n-slop-phrases", str(top_n_slop_phrases)]

    # ------------------------------------------------------------------ logging
    print(f"\n┏━━ Iteration {iter_idx}: launching main.py ━━━━━━━━━━━━━━━━━━━━━━┓")
    print(" ".join(cmd))
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    # ----------------------------------------------------------------- execute
    try:
        subprocess.run(cmd, check=True)
        print(f"✅  main.py finished — output saved to {output_jsonl_path}")
    except subprocess.CalledProcessError as exc:
        print(f"❌  main.py exited with status {exc.returncode}")
        raise





BOOST_EXPONENT     = 0.75   # boost  by corpus_frequency^BOOST_EXPONENT
ATTEN_EXPONENT     = 0.75   # attenuate by wordfreq_frequency^ATTEN_EXPONENT

def build_overrep_word_csv(texts: list[str],
                           out_csv: Path,
                           top_n: int = TOP_K_WORDS):
    """
    Returns three objects:
        1. pandas.DataFrame with columns:
           ['word','ratio_corpus/wordfreq','corpus_freq','wordfreq_freq',
            'modulated_score' (only for dictionary words)]
        2. list[str]  – dictionary words   (wordfreq_freq > 0), sorted by
           modulated_score (highest first)
        3. list[str]  – non-dictionary words (wordfreq_freq == 0), order unchanged
    The CSV written to *out_csv* contains the full DataFrame.
    """
    # ---------- flatten + count with official toolkit helpers ---------------
    counts      = get_word_counts(texts)
    counts      = filter_mostly_numeric(counts)
    counts      = merge_plural_possessive_s(counts)
    counts      = filter_stopwords(counts)

    corpus_freqs, wf_freqs, *_ = analyze_word_rarity(counts)
    overrep = find_over_represented_words(corpus_freqs, wf_freqs, top_n=top_n)

    # ---------- DataFrame ----------------------------------------------------
    df = pd.DataFrame(
        overrep,
        columns=["word",
                 "ratio_corpus/wordfreq",
                 "corpus_freq",
                 "wordfreq_freq"]
    )

    # ensure numeric dtypes
    num_cols = ["ratio_corpus/wordfreq", "corpus_freq", "wordfreq_freq"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # ---------- modulated_score for dictionary words -------------------------
    dict_mask = df["wordfreq_freq"] > 0
    if dict_mask.any():
        boost  = np.power(df.loc[dict_mask, "corpus_freq"], BOOST_EXPONENT)
        atten  = np.power(df.loc[dict_mask, "wordfreq_freq"], ATTEN_EXPONENT)
        df.loc[dict_mask, "modulated_score"] = (
            df.loc[dict_mask, "ratio_corpus/wordfreq"] * boost / atten
        )

    # ---------- write CSV ----------------------------------------------------
    df.to_csv(out_csv, index=False)
    print(f"🔎  over-rep word CSV → {out_csv}  ({len(df)} rows)")

    # ---------- split & sort -------------------------------------------------
    dict_words   = (
        df[dict_mask]
        .sort_values("modulated_score", ascending=False)
        ["word"]
        .tolist()
    )
    nodict_words = df[~dict_mask]["word"].tolist()

    return df, dict_words, nodict_words

def select_overrep_words_for_ban(dict_words: list[str],
                                 nodict_words: list[str],
                                 is_first_iteration: bool) -> list[str]:
    """Return the combined subset according to per-iteration quotas."""
    if is_first_iteration:
        dict_quota   = DICT_OVERREP_INITIAL
        nodict_quota = NODICT_OVERREP_INITIAL
    else:
        dict_quota   = DICT_OVERREP_SUBSEQUENT
        nodict_quota = NODICT_OVERREP_SUBSEQUENT

    return dict_words[:dict_quota] + nodict_words[:nodict_quota]


from slop_forensics.slop_lists import extract_and_save_slop_phrases as _extract_slop_phrases

def update_banned_slop_phrases(json_path: Path,
                               texts: list[str],
                               how_many_new: int,
                               tmp_dir: Path,
                               over_represented_words: list[str] | None = None) -> None:
    """
    1.  Runs Slop-Forensics phrase extractor (`extract_and_save_slop_phrases`)
        on `texts`, writing *slop_list_phrases.jsonl* into `tmp_dir`.
    2.  Reads that file; keeps phrases whose frequency ≥ MIN_PHRASE_FREQ_TO_KEEP.
        Takes the first `how_many_new` unseen phrases (most-frequent first).
    3.  Optionally appends `extra_tokens` (e.g. over-rep words) as one-word
        phrases, limited by TOP_N_OVERREP_WORDS_TO_BAN.
    4.  Merges with any existing ban list and writes the file back in the
        required  **[["phrase", 1], …]** format (sorted alphabetically).
    """

    # --------------------------------------------------------------------- #
    # 1.  Run the heavy Slop-Forensics extractor (this can take time!)      #
    # --------------------------------------------------------------------- #
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _extract_slop_phrases(
        texts               = texts,
        output_dir          = tmp_dir,
        n                   = _sf_cfg.SLOP_PHRASES_NGRAM_SIZE,          # trigram by default (3)
        top_k_ngrams        = max(1_000, how_many_new * 5),
        top_phrases_to_save = how_many_new * 3,
        chunksize           = _sf_cfg.SLOP_PHRASES_CHUNKSIZE,
    )

    phrases_jsonl = tmp_dir / "slop_list_phrases.jsonl"
    if not phrases_jsonl.exists():
        print("⚠️  Slop-Forensics did not produce a phrase file; nothing added.")
        return

    # --------------------------------------------------------------------- #
    # 2.  Load candidate phrases, filter by frequency, keep first N         #
    # --------------------------------------------------------------------- #
    new_phrases: list[str] = []
    with phrases_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if len(new_phrases) >= how_many_new:
                break
            item = json.loads(line)
            if isinstance(item, list):
                phrase, freq = item[0], (item[1] if len(item) > 1 else 1)
            else:                             # fallback: plain string line
                phrase, freq = str(item), 1

            if freq >= MIN_PHRASE_FREQ_TO_KEEP:
                new_phrases.append(phrase)

    # --------------------------------------------------------------------- #
    # 3.  Merge with existing ban list                                      #
    # --------------------------------------------------------------------- #
    existing_phrases: set[str] = set()
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            for entry in raw:
                if isinstance(entry, str):
                    existing_phrases.add(entry)
                elif isinstance(entry, list) and entry:
                    existing_phrases.add(str(entry[0]))
        except Exception as exc:
            print(f"⚠️  Could not read existing ban list ({json_path}): {exc}")

    merged: set[str] = existing_phrases.union(new_phrases)

    # --------------------------------------------------------------------- #
    # 4.  Add over-represented single words                      #
    # --------------------------------------------------------------------- #
    if BAN_OVERREP_WORDS_IN_PHRASE_LIST and over_represented_words:
        merged.update(over_represented_words)

    # --------------------------------------------------------------------- #
    # 5.  Save back in `[["phrase", 1], …]` format                          #
    # --------------------------------------------------------------------- #
    merged_list = sorted([[phrase, 1] for phrase in merged], key=lambda x: x[0])

    json_path.write_text(
        json.dumps(merged_list, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"🚫  Slop-phrase ban list updated → {json_path}   "
        f"(now {len(merged_list)} entries; "
        f"+{len(merged) - len(existing_phrases)} new)"
    )



def _convert_and_normalize_human_ngram_list(ngram_list_of_dicts, n_value: int):
    if not isinstance(ngram_list_of_dicts, list):
        print(f"Warning: Expected a list for human {n_value}-grams, got {type(ngram_list_of_dicts)}. Returning empty dict.")
        return {}
    
    converted_dict = {}
    skipped_count = 0
    original_count = len(ngram_list_of_dicts)

    for item in ngram_list_of_dicts:
        ngram_str = item.get("ngram")
        frequency = item.get("frequency")

        if ngram_str is None or frequency is None:
            skipped_count += 1
            continue
        
        normalized_text_for_human_ngram = normalize_text(str(ngram_str))
        tokens = [t.lower() for t in nltk.word_tokenize(normalized_text_for_human_ngram) if t.isalpha()]
        
        if len(tokens) == n_value:
            processed_ngram_key = " ".join(tokens)
            converted_dict[processed_ngram_key] = converted_dict.get(processed_ngram_key, 0) + int(frequency)
        else:
            skipped_count += 1
            # print(f"Debug: Skipping human {n_value}-gram '{ngram_str}' -> tokens {tokens} (len != {n_value})")

    if skipped_count > 0 or original_count > 0 : # Print info even if no skips but items processed
        print(f"INFO: Normalizing human {n_value}-grams: Processed {original_count} items. "
              f"Resulted in {len(converted_dict)} unique normalized {n_value}-gram keys. "
              f"{skipped_count} original items were skipped or merged.")
    return converted_dict

def norm_per_100k(raw_count: int, char_total: float) -> float:
    if char_total == 0:
        return 0.0 if raw_count == 0 else math.inf
    return (raw_count / char_total) * FREQ_NORM_DENOM

def build_norm_dict(counter: Counter, char_total: float, top_k: int):
    return {
        term: {
            "gen_count": counter[term],
            "gen_freq_per_100k": norm_per_100k(counter[term], char_total)
        }
        for term, _ in counter.most_common(top_k)
    }

def compare_to_human(gen_norm: dict, human_counts: dict, human_total_chars: float):
    both, gen_only = {}, {}
    for term, data in gen_norm.items():
        if term in human_counts:
            h_raw_count = human_counts[term]
            h_freq_norm = norm_per_100k(h_raw_count, human_total_chars)
            gen_freq = data["gen_freq_per_100k"]
            ratio = math.inf
            if h_freq_norm > 0:
                ratio = gen_freq / h_freq_norm
            elif gen_freq == 0 and h_freq_norm == 0:
                ratio = 1.0
            both[term] = {**data, "human_count": h_raw_count, "human_freq_per_100k": h_freq_norm, "freq_ratio_gen/hu": ratio}
        else:
            gen_only[term] = {**data, "human_count": 0, "human_freq_per_100k": 0.0, "freq_ratio_gen/hu": math.inf}
    return both, gen_only

def analyze_iteration_outputs(generated_jsonl_path: Path, human_profile_full: dict, iter_analysis_output_dir: Path):
    """Performs n-gram analysis for a given iteration's generated texts."""
    print(f"\n--- Analyzing Outputs for {generated_jsonl_path.name} ---")
    iter_analysis_output_dir.mkdir(parents=True, exist_ok=True)

    gen_rows = load_jsonl_file(str(generated_jsonl_path))
    gen_texts = [row["generation"] for row in gen_rows if isinstance(row.get("generation"), str)]

    if not gen_texts:
        print(f"Warning: No usable text in {generated_jsonl_path}. Skipping analysis for this iteration.")
        return None, None, None, None, [], 0 # DFs, gen_texts, total_chars

    human_profile = human_profile_full.get('human-authored')
    if not human_profile:
        raise ValueError(f"Key 'human-authored' not found in human profile data.")

    human_bigrams_list = human_profile.get("top_bigrams", [])
    human_trigrams_list = human_profile.get("top_trigrams", [])
    human_bigrams = _convert_and_normalize_human_ngram_list(human_bigrams_list, 2)
    human_trigrams = _convert_and_normalize_human_ngram_list(human_trigrams_list, 3)

    required_keys = ["num_texts_analyzed", "avg_length"]
    for key in required_keys:
        if key not in human_profile:
            raise KeyError(f"Human profile JSON missing required key: '{key}'.")
    
    h_chars_total = human_profile["num_texts_analyzed"] * human_profile["avg_length"]
    if h_chars_total == 0:
        print(f"Warning: Total characters for human data (h_chars_total) is 0.")

    # Word counts & N-gram counts (LLM output)
    word_counter = Counter()
    total_chars = sum(len(txt) for txt in gen_texts)
    
    for txt in gen_texts:
        norm_t = normalize_text(txt)
        word_counter.update(w for w in extract_words(norm_t, MIN_WORD_LEN) if w not in STOP_WORDS)

    bigram_counter = Counter()
    trigram_counter = Counter()
    for txt in gen_texts:
        normalized_llm_text = normalize_text(txt)
        tokens_all = [t.lower() for t in nltk.word_tokenize(normalized_llm_text) if t.isalpha()]
        tokens = [tok for tok in tokens_all if tok not in STOP_WORDS and (len(tok) >= MIN_WORD_LEN or tok in {"it's"})]
        bigram_counter.update(" ".join(bg) for bg in ngrams(tokens, 2))
        trigram_counter.update(" ".join(tg) for tg in ngrams(tokens, 3))

    # Normalise
    gen_bigrams_norm = build_norm_dict(bigram_counter, float(total_chars), TOP_K_BIGRAMS)
    gen_trigrams_norm = build_norm_dict(trigram_counter, float(total_chars), TOP_K_TRIGRAMS)

    # Merge with human profile
    bigrams_dict, bigrams_nondict = compare_to_human(gen_bigrams_norm, human_bigrams, h_chars_total)
    trigrams_dict, trigrams_nondict = compare_to_human(gen_trigrams_norm, human_trigrams, h_chars_total)

    # Create DataFrames
    df_bi_dict = pd.DataFrame.from_dict(bigrams_dict, orient="index")
    df_bi_nondct = pd.DataFrame.from_dict(bigrams_nondict, orient="index")
    df_tri_dict = pd.DataFrame.from_dict(trigrams_dict, orient="index")
    df_tri_nondct = pd.DataFrame.from_dict(trigrams_nondict, orient="index")

    # Sort
    if not df_bi_dict.empty and "freq_ratio_gen/hu" in df_bi_dict.columns:
        df_bi_dict.sort_values(by="freq_ratio_gen/hu", ascending=False, inplace=True)
    if not df_tri_dict.empty and "freq_ratio_gen/hu" in df_tri_dict.columns:
        df_tri_dict.sort_values(by="freq_ratio_gen/hu", ascending=False, inplace=True)
    # Non-dictionary DFs are already sorted by gen_count implicitly by most_common

    # Save CSVs
    df_bi_dict.to_csv(iter_analysis_output_dir / "bigrams__dictionary_sorted.csv")
    df_bi_nondct.to_csv(iter_analysis_output_dir / "bigrams__non_dictionary.csv")
    df_tri_dict.to_csv(iter_analysis_output_dir / "trigrams__dictionary_sorted.csv")
    df_tri_nondct.to_csv(iter_analysis_output_dir / "trigrams__non_dictionary.csv")
    print(f"N-gram analysis CSVs written to {iter_analysis_output_dir.resolve()}")

    return df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct, gen_texts, total_chars


def update_banned_ngrams_list(banned_ngrams_json_path: Path, 
                              dfs: list, # [df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct]
                              num_to_take: int, 
                              is_first_iteration: bool):
    """Updates the JSON file with banned n-grams."""
    newly_banned_ngrams = set()
    
    # df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct
    if not dfs[0].empty: # df_bi_dict
        newly_banned_ngrams.update(dfs[0].head(num_to_take).index.tolist())
    if not dfs[1].empty: # df_bi_nondct
        newly_banned_ngrams.update(dfs[1].head(num_to_take).index.tolist())
    if not dfs[2].empty: # df_tri_dict
        newly_banned_ngrams.update(dfs[2].head(num_to_take).index.tolist())
    if not dfs[3].empty: # df_tri_nondct
        newly_banned_ngrams.update(dfs[3].head(num_to_take).index.tolist())

    current_banned_list = []
    if not is_first_iteration and banned_ngrams_json_path.exists():
        with open(banned_ngrams_json_path, 'r', encoding='utf-8') as f:
            current_banned_list = json.load(f)
    
    # Add new n-grams, ensuring uniqueness and maintaining order for consistency (though set ops remove order)
    # Convert to set for efficient addition, then back to list for JSON serialization
    updated_banned_set = set(current_banned_list)
    updated_banned_set.update(newly_banned_ngrams)
    
    final_banned_list = sorted(list(updated_banned_set)) # Sort for consistent file output

    with open(banned_ngrams_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_banned_list, f, indent=4)
    
    print(f"Updated banned n-grams list at {banned_ngrams_json_path}. Total banned: {len(final_banned_list)}.")
    if is_first_iteration:
        print(f"Added {len(newly_banned_ngrams)} n-grams from initial run.")
    else:
        added_count = len(final_banned_list) - len(current_banned_list)
        print(f"Added {added_count} new unique n-grams to the list.")


def calculate_lexical_diversity_stats(gen_texts: list):
    """Calculates TTR and Root TTR for a list of texts."""
    if not gen_texts:
        return 0.0, 0.0

    all_words = []
    for text in gen_texts:
        normalized_text = normalize_text(text) # Basic normalization
        tokens = [t.lower() for t in nltk.word_tokenize(normalized_text) if t.isalpha() and len(t) > 1] # Alpha, len > 1
        all_words.extend(tokens)
    
    if not all_words:
        return 0.0, 0.0

    num_tokens = len(all_words)
    num_types = len(set(all_words))

    ttr = num_types / num_tokens if num_tokens > 0 else 0.0
    rttr = num_types / math.sqrt(num_tokens) if num_tokens > 0 else 0.0
    
    return ttr, rttr

def calculate_repetition_score(gen_texts: list, total_chars: int, iteration_dfs: list):
    """
    Counts occurrences of top N n-grams from this iteration's analysis within this iteration's texts.
    iteration_dfs: [df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct] for the current iteration.
    """
    if not gen_texts or total_chars == 0:
        return 0.0

    target_ngrams_for_repetition = set()
    # df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct
    if not iteration_dfs[0].empty:
        target_ngrams_for_repetition.update(iteration_dfs[0].head(TOP_N_REPETITION_STAT).index.tolist())
    if not iteration_dfs[1].empty:
        target_ngrams_for_repetition.update(iteration_dfs[1].head(TOP_N_REPETITION_STAT).index.tolist())
    if not iteration_dfs[2].empty:
        target_ngrams_for_repetition.update(iteration_dfs[2].head(TOP_N_REPETITION_STAT).index.tolist())
    if not iteration_dfs[3].empty:
        target_ngrams_for_repetition.update(iteration_dfs[3].head(TOP_N_REPETITION_STAT).index.tolist())

    if not target_ngrams_for_repetition:
        return 0.0

    total_repetition_instances = 0
    for text in gen_texts:
        normalized_text = normalize_text(text)
        # Tokenize consistent with n-gram generation in analyze_iteration_outputs
        tokens_all = [t.lower() for t in nltk.word_tokenize(normalized_text) if t.isalpha()]
        tokens = [tok for tok in tokens_all if tok not in STOP_WORDS and (len(tok) >= MIN_WORD_LEN or tok in {"it's"})]

        current_bigrams = [" ".join(bg) for bg in ngrams(tokens, 2)]
        current_trigrams = [" ".join(tg) for tg in ngrams(tokens, 3)]

        for bg in current_bigrams:
            if bg in target_ngrams_for_repetition:
                total_repetition_instances += 1
        for tg in current_trigrams:
            if tg in target_ngrams_for_repetition:
                total_repetition_instances += 1
    
    repetition_score_normalized = norm_per_100k(total_repetition_instances, float(total_chars))
    return repetition_score_normalized

# --------------------------------------------------------------------- #
# 3)  BUILD DPO DATASET (iteration-0 vs final iteration)                 #
# --------------------------------------------------------------------- #
def create_dpo_dataset(
    iter0_jsonl: Path,
    final_iter_jsonl: Path,
    output_jsonl: Path,
) -> None:
    """
    Writes a JSONL file where each line is:
        {"prompt": <cleaned_prompt>,
            "chosen":  <final_iter_generation>,
            "rejected":<iter0_generation>}
    Prompts present in only one of the two files are skipped.
    """
    KEY_PROMPT     = "prompt"       # ← change if your field name differs
    KEY_PROMPT_ID  = "prompt_id"    # ← optional; fallback is cleaned prompt

    def _strip_wrapping(text: str) -> str:
        # Remove the exact boiler-plate prefix / suffix requested
        prefix = "Writing prompt: "
        suffix = "\n\nWrite 1000 words to this prompt. Your response:\n"
        if text.startswith(prefix):
            text = text[len(prefix):]
        if text.endswith(suffix):
            text = text[: -len(suffix)]
        return text.strip()

    def _load_file(path: Path) -> dict:
        """Returns dict[key → {"prompt":prompt, "generation":gen}]"""
        out = {}
        with path.open(encoding="utf-8") as fh:
            for row_raw in fh:
                try:
                    row = json.loads(row_raw)
                except json.JSONDecodeError:
                    continue
                prompt_raw = row.get(KEY_PROMPT, "")
                gen        = row.get("generation", "")
                if not prompt_raw or not gen:
                    continue
                prompt_clean = _strip_wrapping(prompt_raw)
                # Prefer explicit prompt_id if present and stable
                key = row.get(KEY_PROMPT_ID, prompt_clean)
                out[key] = {"prompt": prompt_clean, "generation": gen}
        return out

    data_iter0   = _load_file(iter0_jsonl)
    data_final   = _load_file(final_iter_jsonl)
    common_keys  = data_iter0.keys() & data_final.keys()

    if not common_keys:
        print("⚠️  No overlapping prompts between iteration-0 and the final "
                "iteration; DPO dataset not written.")
        return

    with output_jsonl.open("w", encoding="utf-8") as out_fh:
        for key in common_keys:
            rec = {
                "prompt":   data_iter0[key]["prompt"],  # same cleaned prompt
                "chosen":   data_final[key]["generation"],
                "rejected": data_iter0[key]["generation"],
            }
            json.dump(rec, out_fh, ensure_ascii=False)
            out_fh.write("\n")

    print(f"📁  DPO dataset written → {output_jsonl} "
            f"({len(common_keys)} prompt pairs)")

# %%
##################################
# Main Pipeline Execution        #
##################################
def antislop_pipeline() -> None:
    """
    Run the multi-iteration anti-slop experiment.

    What this does
    --------------
    • For each iteration:
        1.  Generate LLM outputs with main.py
              – forwards `--ngram-banned-file` (classic anti-slop)
              – forwards `--slop-phrases-file` (Slop-Forensics ban list)
        2.  Analyse outputs vs. human profile:
              – bigrams / trigrams (dict & non-dict)
              – over-represented single words (via Slop-Forensics)
        3.  Update *two* growing ban lists
              a) banned_ngrams.json     (top 300 → 100 rule, freq>1 only)
              b) banned_slop_phrases.json
                 ▸ top 200 → 100 phrases (freq>1 only)
                 ▸ optionally also over-rep words
        4.  Collect lexical-diversity & repetition metrics.

    Outputs
    -------
    • per-iteration analysis CSVs
    • over-represented-word CSVs
    • `banned_ngrams.json`
    • `banned_slop_phrases.json`
    • `final_iteration_statistics.csv`
    """
    # -------------------------------------------------------------------------#
    # 0) PRE-RUN CHECKS & FOLDERS                                              #
    # -------------------------------------------------------------------------#
    if not HUMAN_PROFILE_PATH.exists():
        print(f"ERROR: human profile JSON not found → {HUMAN_PROFILE_PATH}")
        return

    with HUMAN_PROFILE_PATH.open("r", encoding="utf-8") as f_hp:
        human_profile_full: dict = json.load(f_hp)

    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir: Path = EXPERIMENT_BASE_DIR / f"run_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂  Experiment directory: {experiment_dir.resolve()}")

    banned_ngrams_json_path:      Path = experiment_dir / "banned_ngrams.json"
    banned_slop_phrases_json_path: Path = experiment_dir / "banned_slop_phrases.json"

    iteration_stats: list[dict] = []   # rows for final CSV

    # -------------------------------------------------------------------------#
    # 1) ITERATIVE LOOP                                                        #
    # -------------------------------------------------------------------------#
    for iter_idx in range(NUM_ITERATIONS):
        print(f"\n{'='*30}  ITERATION {iter_idx}  {'='*30}")

        # --- paths -----------------------------------------------------------
        iter_output_jsonl: Path = experiment_dir / (
            f"iter_{iter_idx}_creative_writing_generations.jsonl"
        )
        iter_analysis_dir: Path = experiment_dir / f"iter_{iter_idx}_analysis_results"
        iter_analysis_dir.mkdir(parents=True, exist_ok=True)

        # --- decide which ban lists to pass this round -----------------------
        ngram_file_for_cli:       Path | None = None
        slop_phrase_file_for_cli: Path | None = None
        top_n_slop_phrase_flag:   int  | None = None

        if iter_idx > 0:
            if banned_ngrams_json_path.exists():
                ngram_file_for_cli = banned_ngrams_json_path
            if banned_slop_phrases_json_path.exists():
                slop_phrase_file_for_cli = banned_slop_phrases_json_path
                top_n_slop_phrase_flag  = 999_999   # “ban every phrase in file”

        # --- GENERATE TEXTS --------------------------------------------------
        run_generation_script(
            iter_idx                = iter_idx,
            output_jsonl_path       = iter_output_jsonl,
            banned_ngrams_file_path = ngram_file_for_cli,       # NEW restored flag
            slop_phrases_file_path  = slop_phrase_file_for_cli, # phrase ban list
            top_n_slop_phrases      = top_n_slop_phrase_flag or 0,
        )

        # --- ANALYSE TEXTS ---------------------------------------------------
        (df_bi_dict,
         df_bi_nondict,
         df_tri_dict,
         df_tri_nondict,
         generated_texts,
         total_generated_chars) = analyze_iteration_outputs(
             generated_jsonl_path     = iter_output_jsonl,
             human_profile_full       = human_profile_full,
             iter_analysis_output_dir = iter_analysis_dir,
         )

        # If generation/analysis failed, record empty stats and continue
        if df_bi_dict is None:
            iteration_stats.append({
                "iteration":             iter_idx,
                "generated_text_count":  0,
                "generated_char_count":  0,
                "ttr":                   0.0,
                "rttr":                  0.0,
                "repetition_per_100k":   0.0,
                "output_file":           str(iter_output_jsonl),
            })
            continue

        # --- OVER-REPRESENTED WORDS (Slop-Forensics) -------------------------
        overrep_tokens_for_ban: list[str] = []
        if COMPUTE_OVERREP_WORDS:
            overrep_csv: Path = iter_analysis_dir / "overrepresented_words.csv"
            df_overrep, dict_words, nodict_words = build_overrep_word_csv(
                texts   = generated_texts,
                out_csv = overrep_csv,
                top_n   = TOP_K_WORDS,
            )
            overrep_tokens_for_ban = select_overrep_words_for_ban(
                dict_words       = dict_words,
                nodict_words     = nodict_words,
                is_first_iteration = (iter_idx == 0)
            )

        # --- UPDATE ① N-GRAM BAN LIST ------------------------------
        if ENABLE_NGRAM_BAN:
            n_to_ban = TOP_N_INITIAL_BAN if iter_idx == 0 else TOP_N_SUBSEQUENT_BAN
            update_banned_ngrams_list(
                banned_ngrams_json_path,
                dfs=[df_bi_dict, df_bi_nondict, df_tri_dict, df_tri_nondict],
                num_to_take=n_to_ban,
                is_first_iteration=(iter_idx == 0),
            )

        # --- UPDATE ② SLOP-PHRASE BAN LIST -----------------------------------
        if ENABLE_SLOP_PHRASE_BAN:
            phrases_to_add = TOP_N_INITIAL_SLOP_BAN if iter_idx == 0 else TOP_N_SUBSEQUENT_SLOP_BAN
            update_banned_slop_phrases(
                json_path   = banned_slop_phrases_json_path,
                texts       = generated_texts,
                how_many_new= phrases_to_add,
                tmp_dir     = iter_analysis_dir / "phrase_tmp",
                over_represented_words= overrep_tokens_for_ban,
            )

        # --- LEXICAL-DIVERSITY & REPETITION METRICS --------------------------
        ttr, rttr           = calculate_lexical_diversity_stats(generated_texts)
        repetition_norm     = calculate_repetition_score(
            gen_texts     = generated_texts,
            total_chars   = total_generated_chars,
            iteration_dfs = [df_bi_dict, df_bi_nondict, df_tri_dict, df_tri_nondict],
        )

        iteration_stats.append({
            "iteration":             iter_idx,
            "generated_text_count":  len(generated_texts),
            "generated_char_count":  total_generated_chars,
            "ttr":                   ttr,
            "rttr":                  rttr,
            "repetition_per_100k":   repetition_norm,
            "output_file":           str(iter_output_jsonl),
        })

    # -------------------------------------------------------------------------#
    # 2) FINAL SUMMARY CSV                                                     #
    # -------------------------------------------------------------------------#
    summary_df: pd.DataFrame = pd.DataFrame(iteration_stats)
    summary_csv: Path        = experiment_dir / "final_iteration_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)

    # --------------------------------------------------------------------- #
    # 4)  CALL THE DPO DATASET CREATOR                                       #
    # --------------------------------------------------------------------- #
    if NUM_ITERATIONS >= 2:
        iter0_file     = experiment_dir / "iter_0_creative_writing_generations.jsonl"
        last_iter_idx  = NUM_ITERATIONS - 1
        final_file     = experiment_dir / f"iter_{last_iter_idx}_creative_writing_generations.jsonl"
        dpo_output     = experiment_dir / "dpo_pairs_dataset.jsonl"
        create_dpo_dataset(iter0_file, final_file, dpo_output)
    else:
        print("⚠️  Need at least two iterations to build DPO pairs; skipped.")

    print(f"\n📊  Final statistics written → {summary_csv.resolve()}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__" and RUN_PIPELINE:
    # This check allows the notebook to be imported without running the pipeline,
    # or run directly if executed as a script (though it's a .ipynb file).
    # In Jupyter, you'd just run the cell containing antislop_pipeline().
    
    # For direct execution from command line (e.g. `python your_notebook.ipynb` via jupytext or similar)
    # Or if you convert this to a .py script
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("Error: main.py not found in the current directory.")
        print("Please ensure main.py (the generation script) is present.")
        # sys.exit(1) # Exit if running as script and main.py is missing
        # For notebook, we might want to proceed to define functions even if main.py is missing,
        # so commenting out sys.exit. The run_generation_script will fail later.
    
    antislop_pipeline()

elif not RUN_PIPELINE:
    print("RUN_PIPELINE is False. Pipeline execution skipped.")
    print("You can manually call functions or inspect parameters.")






