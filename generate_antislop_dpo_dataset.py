
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

# --- Add local submodule to Python path ---
# This allows importing from the slop-forensics submodule directly.
# Assumes the notebook is in the root of 'auto-antislop' and 'slop-forensics' is a subdirectory.
_slop_forensics_submodule_path = Path.cwd() / "slop-forensics"
if _slop_forensics_submodule_path.is_dir():
    # Add the 'slop-forensics' directory (which contains the 'slop_forensics' package)
    # to the Python path.
    sys.path.insert(0, str(_slop_forensics_submodule_path.resolve()))
    print(f"INFO: Added '{_slop_forensics_submodule_path.resolve()}' to sys.path for slop_forensics imports.")
else:
    print(f"WARNING: Submodule directory '{_slop_forensics_submodule_path}' not found. "
          "Imports from 'slop_forensics' might fail.")
# --- End of submodule path addition ---

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

MODEL_ID = "unsloth/gemma-3-4b-it"
VLLM_PORT=8888


# Number of unslopping iterations to run
# The minimum is 2.
# First iteration generates the baseline dataset and computes over-represnted words/phrases/n-grams
# Additional iterations generate the unslopped dataset, and add to the ban lists as new slop is surfaced.
# 2-3 iterations is a good number.
# It will get slower to compute with more iterations, as bigger banlist == more backtracking.
NUM_ITERATIONS      = 2
MAX_NEW_TOKENS      = 2000



# Parameters for main.py (generation)
THREADS             = 80    # number of parallel threads used for api queries

# Number of prompts for generation: ideal is 1000+, but you can use fewer for testing.
# This will determine the size of the final DPO dataset.
MAX_PROMPTS         = 80
HF_DATASET_NAME     = 'Nitral-AI/Reddit-SFW-Writing_Prompts_ShareGPT'
HF_DATASET_SPLIT    = 'train'
LOGGING_LEVEL       = 'INFO'

# Parameters for N-gram analysis (within each iteration)
ENABLE_NGRAM_BAN    = True
HUMAN_PROFILE_PATH  = Path('data/human_writing_profile.json')
TOP_K_WORDS         = 200_000
TOP_K_BIGRAMS       = 5_000
TOP_K_TRIGRAMS      = 5_000
MIN_WORD_LEN        = 3
FREQ_NORM_DENOM     = 100_000
TOP_N_REPETITION_STAT = 50 # N-grams from each of 4 lists to track for repetition stats
# ---------- per-category N-gram quotas ----------
DICT_BIGRAMS_INITIAL       = 400
DICT_BIGRAMS_SUBSEQUENT    = 70
NODICT_BIGRAMS_INITIAL     = 800
NODICT_BIGRAMS_SUBSEQUENT  = 100
DICT_TRIGRAMS_INITIAL      = 300
DICT_TRIGRAMS_SUBSEQUENT   = 50
NODICT_TRIGRAMS_INITIAL    = 800
NODICT_TRIGRAMS_SUBSEQUENT = 100


# Params for slop phrase banning
COMPUTE_OVERREP_WORDS             = True    # create CSV each iter
ENABLE_SLOP_PHRASE_BAN            = True    # banned-phrase
BAN_OVERREP_WORDS_IN_PHRASE_LIST  = True    # banned words
DICT_OVERREP_INITIAL      = 800   # dictionary words (wf > 0) first iter
DICT_OVERREP_SUBSEQUENT   = 200   # dictionary words later iters
NODICT_OVERREP_INITIAL    =  80   # non-dictionary words (wf == 0) first iter
NODICT_OVERREP_SUBSEQUENT =  20   # non-dictionary words later iters
MIN_PHRASE_FREQ_TO_KEEP           = 2       # ⬅ NEW: only keep phrases seen > 1×


# how many slop phrases to (newly) ban each round
TOP_N_INITIAL_SLOP_BAN   = 600
TOP_N_SUBSEQUENT_SLOP_BAN    = 100

# where we’ll keep the growing list
BANNED_SLOP_PHRASES_FILE = "banned_slop_phrases.json"

# Output directory for the experiment
# (A timestamped subdirectory will be created under this)
EXPERIMENT_BASE_DIR = Path("results") / "iterative_antislop_experiment"


# ------------------------------------------------------------------ #
# USER-SUPPLIED EXTRA BLOCKLISTS – leave empty or fill as desired    #
# ------------------------------------------------------------------ #
EXTRA_NGRAMS_TO_BAN        = [               # bigrams/trigrams, minus stop words & punctuation
    # "voice barely whisper",
]

EXTRA_SLOP_PHRASES_TO_BAN  = [               # strings to ban (lowercased)
    # "rain tasted like",
    "…", "*", " –", "–", "#",
]

EXTRA_REGEX_PATTERNS       = [               # Python regexps to ban
    
    # These ones ban "it's not x, it's y" type patterns:
    "\\bnot\\s+(?:just|only|merely)?\\s*(?:[^\\s]+\\s*){1,6}?[,;:—–-]?\\s*but\\s+(?:also\\s+)?",
    "\\bnot\\s+only\\s+(?:[^\\s]+\\s*){1,6}?[,;:—–-]?\\s*but\\s+also\\s+",
    "\\bit'?s\\s+not\\s+(?:just|only|merely)?\\s*(?:[^\\s]+\\s*){1,6}?[,;:—–-]\\s*it'?s\\s+",
    "\\b(?:[^\\s]+\\s*){1,4}?is\\s+not\\s+(?:just\\s+|only\\s+)?(?:about\\s+)?(?:[^\\s]+\\s*){1,6}?[,;:—–-]\\s*it'?s\\s+(?:about\\s+)?"
]


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
except ImportError as e:
    print(f"Warning: slop_forensics.utils not found despite sys.path modification. Error: {e}")
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
    regex_blocklist_file_path: Path | None = None,
    top_n_slop_phrases: int | None = None,
) -> None:
    """Invoke antislop-vllm/main.py with the appropriate blocklists."""

    # -------- locate main.py -----------------------------------------------
    main_py = Path.cwd() / "main.py"
    if not main_py.exists():                                      # fallback if cwd is already antislop-vllm
        alt = Path.cwd() / "main.py"
        if alt.exists():
            main_py = alt
        else:
            raise FileNotFoundError("main.py not found; expected under antislop-vllm/")

    workdir = main_py.parent                       # we’ll `cwd` there
    rel = lambda p: os.path.relpath(p.resolve(), workdir)

    # -------- build CLI ----------------------------------------------------
    cmd = [
        sys.executable, main_py.name,
        "--api-base-url", f"http://localhost:{VLLM_PORT}/v1",
        "--output-jsonl",      rel(output_jsonl_path),
        "--input-hf-dataset",  HF_DATASET_NAME,
        "--hf-dataset-split",  HF_DATASET_SPLIT,
        "--threads",           str(THREADS),
        "--max-prompts",       str(MAX_PROMPTS),
        "--logging-level",     LOGGING_LEVEL,
        "--max-new-tokens",    str(MAX_NEW_TOKENS),
        "--model-name",        MODEL_ID,
    ]

    if banned_ngrams_file_path is not None:
        cmd += ["--ngram-banned-file",  rel(banned_ngrams_file_path)]
    if slop_phrases_file_path is not None:
        cmd += ["--slop-phrases-file",  rel(slop_phrases_file_path)]
        if top_n_slop_phrases:
            cmd += ["--top-n-slop-phrases", str(top_n_slop_phrases)]
    if regex_blocklist_file_path is not None:
        cmd += ["--regex-blocklist-file", rel(regex_blocklist_file_path)]

    # -------- run ----------------------------------------------------------
    print(f"\n┏━━ Iteration {iter_idx}: launching main.py ━━━━━━━━━━━━━┓")
    print(" ".join(cmd)); print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
    subprocess.run(cmd, check=True, cwd=workdir)
    print(f"✅  main.py finished — output saved to {output_jsonl_path}")






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
        # Ensure no division by zero or power of zero if atten can be zero
        # (though dict_mask > 0 should prevent wordfreq_freq == 0)
        df_dict = df[dict_mask].copy() # Work on a copy to avoid SettingWithCopyWarning
        
        boost  = np.power(df_dict["corpus_freq"], BOOST_EXPONENT)
        atten  = np.power(df_dict["wordfreq_freq"], ATTEN_EXPONENT)
        
        # Handle potential division by zero if atten is zero for some reason
        # (should not happen with dict_mask, but good for robustness)
        modulated_score = df_dict["ratio_corpus/wordfreq"] * boost
        atten_safe = np.where(atten == 0, 1, atten) # Replace 0 with 1 to avoid division by zero
        
        df.loc[dict_mask, "modulated_score"] = modulated_score / atten_safe


    # ---------- write CSV ----------------------------------------------------
    df.to_csv(out_csv, index=False)
    print(f"🔎  over-rep word CSV → {out_csv}  ({len(df)} rows)")

    # ---------- split & sort -------------------------------------------------
    dict_words_df = df[dict_mask]
    if "modulated_score" in dict_words_df.columns:
        dict_words   = (
            dict_words_df
            .sort_values("modulated_score", ascending=False)
            ["word"]
            .tolist()
        )
    else: # Fallback if modulated_score wasn't created (e.g., no dict words)
        dict_words = dict_words_df["word"].tolist()

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


# from slop_forensics.slop_lists import extract_and_save_slop_phrases as _extract_slop_phrases # Already imported

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
        phrases. (Note: Original code mentioned TOP_N_OVERREP_WORDS_TO_BAN but didn't use it here)
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
        top_k_ngrams        = max(1_000, how_many_new * 5), # Ensure enough candidates
        top_phrases_to_save = max(how_many_new * 3, 100), # Ensure enough candidates are saved
        chunksize           = _sf_cfg.SLOP_PHRASES_CHUNKSIZE,
    )

    phrases_jsonl = tmp_dir / "slop_list_phrases.jsonl"
    if not phrases_jsonl.exists():
        print("⚠️  Slop-Forensics did not produce a phrase file; nothing added to slop phrase ban list.")
        # Still proceed to save if there are over_represented_words to ban
        if not (BAN_OVERREP_WORDS_IN_PHRASE_LIST and over_represented_words):
            return # Nothing to do if no phrases and no overrep words to add

    # --------------------------------------------------------------------- #
    # 2.  Load candidate phrases, filter by frequency, keep first N         #
    # --------------------------------------------------------------------- #
    new_phrases_from_file: list[str] = []
    if phrases_jsonl.exists():
        with phrases_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                # if len(new_phrases_from_file) >= how_many_new: # This limit should apply to *newly added* from file
                #     break
                try:
                    item = json.loads(line)
                    if isinstance(item, list) and len(item) >= 1:
                        phrase, freq = item[0], (item[1] if len(item) > 1 else 1)
                    elif isinstance(item, str): # fallback: plain string line
                        phrase, freq = item, 1
                    else:
                        print(f"⚠️  Skipping malformed line in {phrases_jsonl}: {line.strip()}")
                        continue

                    if freq >= MIN_PHRASE_FREQ_TO_KEEP:
                        new_phrases_from_file.append(str(phrase))
                except json.JSONDecodeError:
                    print(f"⚠️  Skipping non-JSON line in {phrases_jsonl}: {line.strip()}")
                    continue
    
    # new_phrases_from_file are already sorted by frequency by the extractor.
    # We will take `how_many_new` from these *after* checking against existing.

    # --------------------------------------------------------------------- #
    # 3.  Merge with existing ban list                                      #
    # --------------------------------------------------------------------- #
    existing_phrases_set: set[str] = set()
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            for entry in raw:
                if isinstance(entry, str):
                    existing_phrases_set.add(entry)
                elif isinstance(entry, list) and entry:
                    existing_phrases_set.add(str(entry[0]))
        except Exception as exc:
            print(f"⚠️  Could not read existing slop phrase ban list ({json_path}): {exc}")

    # Determine truly new phrases to add from the file, up to `how_many_new`
    actually_new_phrases_to_add: list[str] = []
    for p in new_phrases_from_file:
        if len(actually_new_phrases_to_add) >= how_many_new:
            break
        if p not in existing_phrases_set:
            actually_new_phrases_to_add.append(p)

    merged_set: set[str] = existing_phrases_set.copy()
    if EXTRA_SLOP_PHRASES_TO_BAN:           # ← new
        merged_set.update(EXTRA_SLOP_PHRASES_TO_BAN)

    merged_set.update(actually_new_phrases_to_add)
    
    num_added_from_file = len(merged_set) - len(existing_phrases_set)

    # --------------------------------------------------------------------- #
    # 4.  Add over-represented single words                                 #
    # --------------------------------------------------------------------- #
    num_added_from_overrep = 0
    if BAN_OVERREP_WORDS_IN_PHRASE_LIST and over_represented_words:
        initial_merged_size = len(merged_set)
        merged_set.update(over_represented_words)
        num_added_from_overrep = len(merged_set) - initial_merged_size

    # --------------------------------------------------------------------- #
    # 5.  Save back in `[["phrase", 1], …]` format                          #
    # --------------------------------------------------------------------- #
    if not merged_set and not json_path.exists(): # No phrases to write and no file exists
        print(f"🚫  No slop phrases or over-represented words to ban. File not created: {json_path}")
        return

    merged_list_for_json = sorted([[phrase, 1] for phrase in merged_set], key=lambda x: x[0])

    json_path.write_text(
        json.dumps(merged_list_for_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_newly_added = num_added_from_file + num_added_from_overrep
    print(
        f"🚫  Slop-phrase ban list updated → {json_path}   "
        f"(now {len(merged_list_for_json)} entries; "
        f"+{total_newly_added} new: {num_added_from_file} from phrases, {num_added_from_overrep} from overrep words)"
    )


def _convert_and_normalize_human_ngram_list(ngram_list_of_dicts, n_value: int):
    if not isinstance(ngram_list_of_dicts, list):
        print(f"Warning: Expected a list for human {n_value}-grams, got {type(ngram_list_of_dicts)}. Returning empty dict.")
        return {}
    
    converted_dict = {}
    skipped_count = 0
    malformed_count = 0
    original_count = len(ngram_list_of_dicts)

    for item in ngram_list_of_dicts:
        if not isinstance(item, dict):
            malformed_count +=1
            continue

        ngram_str = item.get("ngram")
        frequency = item.get("frequency")

        if ngram_str is None or frequency is None:
            malformed_count += 1
            continue
        
        try:
            freq_int = int(frequency)
        except ValueError:
            malformed_count += 1
            continue

        normalized_text_for_human_ngram = normalize_text(str(ngram_str))
        # Tokenize using nltk, keep only alpha, convert to lower
        tokens = [t.lower() for t in nltk.word_tokenize(normalized_text_for_human_ngram) 
                  if t.isalpha()] # Ensure only actual words form the n-gram
        
        if len(tokens) == n_value:
            processed_ngram_key = " ".join(tokens)
            if processed_ngram_key: # Ensure key is not empty string
                converted_dict[processed_ngram_key] = converted_dict.get(processed_ngram_key, 0) + freq_int
            else:
                skipped_count +=1 # Skipped due to empty n-gram after normalization
        else:
            # This case means after normalization, the number of words is not n_value
            # e.g. "word1 -" (bigram) might become "word1" (unigram)
            skipped_count += 1
            # print(f"Debug: Skipping human {n_value}-gram '{ngram_str}' -> tokens {tokens} (len != {n_value})")

    if skipped_count > 0 or malformed_count > 0 or original_count > 0 :
        print(f"INFO: Normalizing human {n_value}-grams: Processed {original_count} items. "
              f"Resulted in {len(converted_dict)} unique normalized {n_value}-gram keys. "
              f"{skipped_count} items skipped (token count != {n_value} post-norm). "
              f"{malformed_count} items skipped (malformed entry).")
    return converted_dict

def norm_per_100k(raw_count: int, char_total: float) -> float:
    if char_total == 0:
        return 0.0 if raw_count == 0 else math.inf # Or handle as NaN: float('nan')
    return (raw_count / char_total) * FREQ_NORM_DENOM

def build_norm_dict(counter: Counter, char_total: float, top_k: int):
    # Ensure char_total is float for division
    char_total_float = float(char_total)
    return {
        term: {
            "gen_count": counter[term],
            "gen_freq_per_100k": norm_per_100k(counter[term], char_total_float)
        }
        for term, _ in counter.most_common(top_k) if term # Ensure term is not empty
    }

def compare_to_human(gen_norm: dict, human_counts: dict, human_total_chars: float):
    both, gen_only = {}, {}
    # Ensure human_total_chars is float
    human_total_chars_float = float(human_total_chars)

    for term, data in gen_norm.items():
        if not term: continue # Skip empty terms

        if term in human_counts:
            h_raw_count = human_counts[term]
            h_freq_norm = norm_per_100k(h_raw_count, human_total_chars_float)
            gen_freq = data["gen_freq_per_100k"]
            
            ratio = math.inf # Default for gen_freq > 0, h_freq_norm == 0
            if h_freq_norm > 0:
                ratio = gen_freq / h_freq_norm
            elif gen_freq == 0 and h_freq_norm == 0: # Both are zero
                ratio = 1.0 # Or 0.0, depending on interpretation. 1.0 means "equally absent"
            elif gen_freq == 0 and h_freq_norm > 0: # Gen is zero, human is not
                ratio = 0.0

            both[term] = {**data, "human_count": h_raw_count, "human_freq_per_100k": h_freq_norm, "freq_ratio_gen/hu": ratio}
        else:
            # Term in generated, not in human
            gen_only[term] = {**data, "human_count": 0, "human_freq_per_100k": 0.0, "freq_ratio_gen/hu": math.inf if data["gen_freq_per_100k"] > 0 else 0.0}
    return both, gen_only

def analyze_iteration_outputs(generated_jsonl_path: Path, human_profile_full: dict, iter_analysis_output_dir: Path):
    """Performs n-gram analysis for a given iteration's generated texts."""
    print(f"\n--- Analyzing Outputs for {generated_jsonl_path.name} ---")
    iter_analysis_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gen_rows = load_jsonl_file(str(generated_jsonl_path))
    except Exception as e:
        print(f"Error loading generated JSONL file {generated_jsonl_path}: {e}")
        return None, None, None, None, [], 0

    gen_texts = [row["generation"] for row in gen_rows if isinstance(row, dict) and isinstance(row.get("generation"), str)]

    if not gen_texts:
        print(f"Warning: No usable text in {generated_jsonl_path}. Skipping analysis for this iteration.")
        return None, None, None, None, [], 0 # DFs, gen_texts, total_chars

    human_profile_data_key = 'human-authored' # As per original structure
    human_profile = human_profile_full.get(human_profile_data_key)
    if not human_profile:
        # Try to find a key that might contain the profile if the exact one is missing
        potential_keys = [k for k in human_profile_full.keys() if isinstance(human_profile_full[k], dict) and "top_bigrams" in human_profile_full[k]]
        if potential_keys:
            human_profile_data_key = potential_keys[0]
            human_profile = human_profile_full.get(human_profile_data_key)
            print(f"INFO: Using human profile data from key '{human_profile_data_key}' as '{'human-authored'}' was not found directly.")
        else:
            raise ValueError(f"Key for human profile data (e.g., 'human-authored') not found in human profile JSON.")


    human_bigrams_list = human_profile.get("top_bigrams", [])
    human_trigrams_list = human_profile.get("top_trigrams", [])
    human_bigrams = _convert_and_normalize_human_ngram_list(human_bigrams_list, 2)
    human_trigrams = _convert_and_normalize_human_ngram_list(human_trigrams_list, 3)

    required_keys = ["num_texts_analyzed", "avg_length"]
    missing_keys = [key for key in required_keys if key not in human_profile]
    if missing_keys:
        raise KeyError(f"Human profile JSON (under key '{human_profile_data_key}') missing required keys: {', '.join(missing_keys)}.")
    
    h_num_texts = human_profile["num_texts_analyzed"]
    h_avg_len = human_profile["avg_length"]
    
    if not isinstance(h_num_texts, (int, float)) or not isinstance(h_avg_len, (int, float)):
        raise ValueError("Human profile 'num_texts_analyzed' or 'avg_length' are not numeric.")

    h_chars_total = float(h_num_texts * h_avg_len)
    if h_chars_total == 0:
        print(f"Warning: Total characters for human data (h_chars_total) is 0. Frequencies will be infinite if gen counts > 0.")

    # Word counts & N-gram counts (LLM output)
    # word_counter = Counter() # Not used directly for n-gram DFs, but good for other stats if needed
    total_chars = sum(len(txt) for txt in gen_texts)
    
    # for txt in gen_texts: # Example of word counting if needed elsewhere
    #     norm_t = normalize_text(txt)
    #     word_counter.update(w for w in extract_words(norm_t, MIN_WORD_LEN) if w not in STOP_WORDS)

    bigram_counter = Counter()
    trigram_counter = Counter()
    for txt in gen_texts:
        normalized_llm_text = normalize_text(txt)
        # Tokenize, lower, keep alpha, filter stopwords and min length (consistent with human profile processing)
        tokens_all = [t.lower() for t in nltk.word_tokenize(normalized_llm_text) if t.isalpha()]
        tokens = [tok for tok in tokens_all if tok not in STOP_WORDS and (len(tok) >= MIN_WORD_LEN or tok in {"it's", "i'm"})] # allow common contractions
        
        current_bigrams = [" ".join(bg) for bg in ngrams(tokens, 2) if all(bg)]
        current_trigrams = [" ".join(tg) for tg in ngrams(tokens, 3) if all(tg)]
        
        bigram_counter.update(bg for bg in current_bigrams if bg) # Ensure not empty
        trigram_counter.update(tg for tg in current_trigrams if tg) # Ensure not empty


    # Normalise
    gen_bigrams_norm = build_norm_dict(bigram_counter, float(total_chars), TOP_K_BIGRAMS)
    gen_trigrams_norm = build_norm_dict(trigram_counter, float(total_chars), TOP_K_TRIGRAMS)

    # Merge with human profile
    bigrams_dict, bigrams_nondict = compare_to_human(gen_bigrams_norm, human_bigrams, h_chars_total)
    trigrams_dict, trigrams_nondict = compare_to_human(gen_trigrams_norm, human_trigrams, h_chars_total)

    # Create DataFrames
    df_bi_dict = pd.DataFrame.from_dict(bigrams_dict, orient="index").rename_axis('ngram').reset_index()
    df_bi_nondct = pd.DataFrame.from_dict(bigrams_nondict, orient="index").rename_axis('ngram').reset_index()
    df_tri_dict = pd.DataFrame.from_dict(trigrams_dict, orient="index").rename_axis('ngram').reset_index()
    df_tri_nondct = pd.DataFrame.from_dict(trigrams_nondict, orient="index").rename_axis('ngram').reset_index()
    
    # Set 'ngram' as index again after ensuring it's a column for reliable head() later
    if not df_bi_dict.empty: df_bi_dict.set_index('ngram', inplace=True)
    if not df_bi_nondct.empty: df_bi_nondct.set_index('ngram', inplace=True)
    if not df_tri_dict.empty: df_tri_dict.set_index('ngram', inplace=True)
    if not df_tri_nondct.empty: df_tri_nondct.set_index('ngram', inplace=True)


    # Sort
    sort_col = "freq_ratio_gen/hu"
    if not df_bi_dict.empty and sort_col in df_bi_dict.columns:
        df_bi_dict.sort_values(by=sort_col, ascending=False, inplace=True)
    if not df_tri_dict.empty and sort_col in df_tri_dict.columns:
        df_tri_dict.sort_values(by=sort_col, ascending=False, inplace=True)
    
    # For non-dictionary, sort by gen_freq_per_100k (descending) as they don't have human comparison
    sort_col_nondict = "gen_freq_per_100k"
    if not df_bi_nondct.empty and sort_col_nondict in df_bi_nondct.columns:
        df_bi_nondct.sort_values(by=sort_col_nondict, ascending=False, inplace=True)
    if not df_tri_nondct.empty and sort_col_nondict in df_tri_nondct.columns:
        df_tri_nondct.sort_values(by=sort_col_nondict, ascending=False, inplace=True)


    # Save CSVs
    df_bi_dict.to_csv(iter_analysis_output_dir / "bigrams__dictionary_sorted.csv")
    df_bi_nondct.to_csv(iter_analysis_output_dir / "bigrams__non_dictionary_sorted.csv") # Updated name
    df_tri_dict.to_csv(iter_analysis_output_dir / "trigrams__dictionary_sorted.csv")
    df_tri_nondct.to_csv(iter_analysis_output_dir / "trigrams__non_dictionary_sorted.csv") # Updated name
    print(f"N-gram analysis CSVs written to {iter_analysis_output_dir.resolve()}")

    return df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct, gen_texts, total_chars


def update_banned_ngrams_list(banned_ngrams_json_path: Path,
                              dfs: list,         # [df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct]
                              is_first_iteration: bool,
                              *,
                              extra_items: list[str] | None = None):
    """Merge newly-found n-grams plus any user-supplied extras into JSON."""
    newly = set()

    def _take(df, n):            # quick helper
        return set(df.head(n).index) if df is not None and not df.empty and n > 0 else set()

    if is_first_iteration:
        newly |= _take(dfs[0], DICT_BIGRAMS_INITIAL)
        newly |= _take(dfs[1], NODICT_BIGRAMS_INITIAL)
        newly |= _take(dfs[2], DICT_TRIGRAMS_INITIAL)
        newly |= _take(dfs[3], NODICT_TRIGRAMS_INITIAL)
    else:
        newly |= _take(dfs[0], DICT_BIGRAMS_SUBSEQUENT)
        newly |= _take(dfs[1], NODICT_BIGRAMS_SUBSEQUENT)
        newly |= _take(dfs[2], DICT_TRIGRAMS_SUBSEQUENT)
        newly |= _take(dfs[3], NODICT_TRIGRAMS_SUBSEQUENT)

    if extra_items:
        newly |= set(extra_items)


    current = []
    if banned_ngrams_json_path.exists():
        try:
            current = json.loads(banned_ngrams_json_path.read_text("utf-8"))
            if not isinstance(current, list): current = []
        except json.JSONDecodeError:
            current = []

    final = sorted(set(current) | newly)
    banned_ngrams_json_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), "utf-8")

    added = len(final) - len(current)
    print(f"📄  N-gram ban list updated → {banned_ngrams_json_path} (+{added}, total {len(final)})")


def calculate_lexical_diversity_stats(gen_texts: list):
    """Calculates TTR and Root TTR for a list of texts."""
    if not gen_texts:
        return 0.0, 0.0

    all_words = []
    for text in gen_texts:
        normalized_text = normalize_text(text) 
        # Consistent tokenization: lower, alpha, len > 1 (or common contractions)
        tokens = [t.lower() for t in nltk.word_tokenize(normalized_text) 
                  if t.isalpha() and (len(t) > 1 or t in {"a", "i"})] # Keep short words like 'a', 'i'
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
    Counts occurrences of top N n-grams from this iteration's analysis 
    (those most overrepresented or most frequent if not in human data)
    within this iteration's texts.
    iteration_dfs: [df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct] for the current iteration.
    """
    if not gen_texts or total_chars == 0:
        return 0.0

    target_ngrams_for_repetition = set()
    # df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct
    # These DFs are already sorted by overrepresentation or frequency
    if iteration_dfs[0] is not None and not iteration_dfs[0].empty:
        target_ngrams_for_repetition.update(iteration_dfs[0].head(TOP_N_REPETITION_STAT).index.tolist())
    if iteration_dfs[1] is not None and not iteration_dfs[1].empty:
        target_ngrams_for_repetition.update(iteration_dfs[1].head(TOP_N_REPETITION_STAT).index.tolist())
    if iteration_dfs[2] is not None and not iteration_dfs[2].empty:
        target_ngrams_for_repetition.update(iteration_dfs[2].head(TOP_N_REPETITION_STAT).index.tolist())
    if iteration_dfs[3] is not None and not iteration_dfs[3].empty:
        target_ngrams_for_repetition.update(iteration_dfs[3].head(TOP_N_REPETITION_STAT).index.tolist())

    if not target_ngrams_for_repetition:
        # print("No target n-grams for repetition score calculation.")
        return 0.0

    total_repetition_instances = 0
    for text in gen_texts:
        normalized_text = normalize_text(text)
        # Tokenize consistent with n-gram generation in analyze_iteration_outputs
        tokens_all = [t.lower() for t in nltk.word_tokenize(normalized_text) if t.isalpha()]
        tokens = [tok for tok in tokens_all if tok not in STOP_WORDS and (len(tok) >= MIN_WORD_LEN or tok in {"it's", "i'm"})]

        current_bigrams = [" ".join(bg) for bg in ngrams(tokens, 2) if all(bg)]
        current_trigrams = [" ".join(tg) for tg in ngrams(tokens, 3) if all(tg)]

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
    KEY_PROMPT     = "prompt"       # field name in input JSONL
    KEY_GENERATION = "generation" # field name for LLM output
    KEY_PROMPT_ID  = "prompt_id"    # optional unique ID for prompt

    def _strip_wrapping(text: str) -> str:
        # Remove the exact boiler-plate prefix / suffix requested
        # This might need adjustment if the prompt format from main.py changes
        prefix = "Writing prompt: " 
        # Example suffix, adjust if needed. The original was very specific.
        # Let's make it more general or rely on main.py to provide clean prompts if possible.
        # suffix = "\n\nWrite 1000 words to this prompt. Your response:\n" 
        
        # A more robust way might be to look for a known start and end of the actual prompt text
        # if the wrapping is complex. For now, simple prefix stripping.
        if text.startswith(prefix):
            text = text[len(prefix):]
        # if text.endswith(suffix): # Commented out as suffix might vary
        #     text = text[: -len(suffix)]
        return text.strip()

    def _load_file(path: Path) -> dict[str, dict[str, str]]:
        """Returns dict[key → {"prompt":prompt_clean, "generation":gen}]"""
        out_data: dict[str, dict[str, str]] = {}
        if not path.exists():
            print(f"Warning: DPO source file not found: {path}")
            return out_data
            
        with path.open(encoding="utf-8") as fh:
            for i, line_raw in enumerate(fh):
                try:
                    row = json.loads(line_raw)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {i+1} in {path}")
                    continue
                
                if not isinstance(row, dict):
                    print(f"Warning: Skipping non-dict row {i+1} in {path}")
                    continue

                prompt_raw = row.get(KEY_PROMPT)
                gen        = row.get(KEY_GENERATION)

                if not isinstance(prompt_raw, str) or not isinstance(gen, str) or not prompt_raw or not gen:
                    # print(f"Warning: Skipping row {i+1} in {path} due to missing/invalid prompt or generation.")
                    continue
                
                prompt_clean = _strip_wrapping(prompt_raw)
                
                # Use prompt_id if available and valid, otherwise use cleaned prompt as key
                # This assumes prompt_id is consistent across generation runs for the same base prompt
                key_val = row.get(KEY_PROMPT_ID)
                if not isinstance(key_val, (str, int)) or not key_val: # Ensure key_val is usable
                    key = prompt_clean
                else:
                    key = str(key_val)

                if not key: # Final check if key ended up empty
                    # print(f"Warning: Skipping row {i+1} in {path} due to empty key after processing prompt_id/prompt.")
                    continue

                out_data[key] = {"prompt": prompt_clean, "generation": gen}
        return out_data

    data_iter0   = _load_file(iter0_jsonl)
    data_final   = _load_file(final_iter_jsonl)
    
    if not data_iter0 or not data_final:
        print("⚠️  DPO dataset not created: one or both input files were empty or could not be loaded.")
        return

    common_keys  = data_iter0.keys() & data_final.keys()

    if not common_keys:
        print("⚠️  No overlapping prompts (based on prompt_id or cleaned prompt text) "
              "between iteration-0 and the final iteration; DPO dataset not written.")
        print(f"   Iter0 keys: {len(data_iter0)}, Final keys: {len(data_final)}")
        return

    count_written = 0
    with output_jsonl.open("w", encoding="utf-8") as out_fh:
        for key in common_keys:
            # Ensure prompts are indeed the same if keying by prompt_id
            # This is a sanity check, they should be if the key is the prompt itself.
            # If key is prompt_id, data_iter0[key]["prompt"] might differ slightly if _strip_wrapping changed.
            # For DPO, the 'prompt' field should be identical for chosen/rejected.
            # We'll use the one from iter0 as the canonical one for the DPO pair.
            
            prompt_for_dpo = data_iter0[key]["prompt"] 
            # If you want to ensure it's truly identical to final prompt (e.g. if stripping changed):
            # if data_iter0[key]["prompt"] != data_final[key]["prompt"]:
            #     print(f"Warning: Prompt text mismatch for key '{key}'. Using iter0 prompt for DPO.")

            rec = {
                "prompt":   prompt_for_dpo,
                "chosen":   data_final[key]["generation"],
                "rejected": data_iter0[key]["generation"],
            }
            json.dump(rec, out_fh, ensure_ascii=False)
            out_fh.write("\n")
            count_written +=1

    print(f"📁  DPO dataset written → {output_jsonl} "
            f"({count_written} prompt pairs from {len(common_keys)} common keys)")

# %%
##################################
# Main Pipeline Execution        #
##################################
def antislop_pipeline() -> None:
    """
    Run the multi-iteration anti-slop experiment.
    (Detailed docstring from original code omitted for brevity here)
    """
    # -------------------------------------------------------------------------#
    # 0) PRE-RUN CHECKS & FOLDERS                                              #
    # -------------------------------------------------------------------------#
    if not HUMAN_PROFILE_PATH.exists():
        print(f"ERROR: human profile JSON not found → {HUMAN_PROFILE_PATH.resolve()}")
        return
    
    print(f"INFO: Using human profile from: {HUMAN_PROFILE_PATH.resolve()}")
    try:
        with HUMAN_PROFILE_PATH.open("r", encoding="utf-8") as f_hp:
            human_profile_full: dict = json.load(f_hp)
    except Exception as e:
        print(f"ERROR: Could not load or parse human profile JSON: {e}")
        return

    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir: Path = EXPERIMENT_BASE_DIR / f"run_{timestamp}"
    try:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------- user regex blocklist
        banned_regexes_json_path: Path = experiment_dir / "banned_regexes.json"
        if EXTRA_REGEX_PATTERNS:
            banned_regexes_json_path.write_text(
                json.dumps(EXTRA_REGEX_PATTERNS, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"📝  Regex blocklist written → {banned_regexes_json_path}")
    except OSError as e:
        print(f"ERROR: Could not create experiment directory {experiment_dir}: {e}")
        return
        
    print(f"\n📂  Experiment directory: {experiment_dir.resolve()}")

    # These paths are now relative to experiment_dir
    banned_ngrams_json_path:      Path = experiment_dir / "banned_ngrams.json"
    banned_slop_phrases_json_path: Path = experiment_dir / BANNED_SLOP_PHRASES_FILE # Use defined constant

    iteration_stats: list[dict] = [] 

    # --- Store initial (iter 0) and final iteration output paths for DPO ---
    iter0_output_file_for_dpo: Path | None = None
    final_iter_output_file_for_dpo: Path | None = None


    # -------------------------------------------------------------------------#
    # 1) ITERATIVE LOOP                                                        #
    # -------------------------------------------------------------------------#
    for iter_idx in range(NUM_ITERATIONS):
        current_iter_start_time = datetime.datetime.now()
        print(f"\n{'='*30}  ITERATION {iter_idx} (started at {current_iter_start_time.strftime('%H:%M:%S')})  {'='*30}")

        iter_output_jsonl: Path = experiment_dir / (
            f"iter_{iter_idx}_creative_writing_generations.jsonl"
        )
        iter_analysis_dir: Path = experiment_dir / f"iter_{iter_idx}_analysis_results"
        iter_analysis_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

        # --- Decide which ban lists to pass this round -----------------------
        ngram_file_for_cli:       Path | None = None
        slop_phrase_file_for_cli: Path | None = None
        top_n_slop_phrase_flag:   int  | None = None # For main.py --top-n-slop-phrases

        ngram_file_for_cli       = banned_ngrams_json_path if (iter_idx > 0 and ENABLE_NGRAM_BAN and banned_ngrams_json_path.exists()) else None
        slop_phrase_file_for_cli = banned_slop_phrases_json_path if (iter_idx > 0 and ENABLE_SLOP_PHRASE_BAN and banned_slop_phrases_json_path.exists()) else None
        regex_file_for_cli       = banned_regexes_json_path if (iter_idx > 0 and EXTRA_REGEX_PATTERNS) else None
        top_n_slop_phrase_flag   = 999_999 if slop_phrase_file_for_cli else 0

        
        # --- GENERATE TEXTS --------------------------------------------------
        try:
            run_generation_script(
                iter_idx                = iter_idx,
                output_jsonl_path       = iter_output_jsonl,
                banned_ngrams_file_path = ngram_file_for_cli,
                slop_phrases_file_path  = slop_phrase_file_for_cli,
                regex_blocklist_file_path = regex_file_for_cli,
                top_n_slop_phrases      = top_n_slop_phrase_flag,
            )

        except Exception as e:
            print(f"❌ ERROR during text generation (main.py call) for iteration {iter_idx}: {e}")
            print(f"Skipping analysis and ban list updates for iteration {iter_idx}.")
            iteration_stats.append({
                "iteration": iter_idx, "status": "generation_failed", 
                "error": str(e), "output_file": str(iter_output_jsonl),
                 # Add other fields with default/NA values
                "generated_text_count": 0, "generated_char_count": 0,
                "ttr": 0.0, "rttr": 0.0, "repetition_per_100k": 0.0,
            })
            if iter_idx == 0: iter0_output_file_for_dpo = None # Mark as failed
            if iter_idx == NUM_ITERATIONS - 1: final_iter_output_file_for_dpo = None # Mark as failed
            continue # Move to next iteration or finish

        if not iter_output_jsonl.exists() or iter_output_jsonl.stat().st_size == 0:
            print(f"❌ ERROR: Generation output file {iter_output_jsonl} is missing or empty for iteration {iter_idx}.")
            iteration_stats.append({
                "iteration": iter_idx, "status": "output_file_missing_or_empty",
                "output_file": str(iter_output_jsonl),
                "generated_text_count": 0, "generated_char_count": 0,
                "ttr": 0.0, "rttr": 0.0, "repetition_per_100k": 0.0,
            })
            if iter_idx == 0: iter0_output_file_for_dpo = None
            if iter_idx == NUM_ITERATIONS - 1: final_iter_output_file_for_dpo = None
            continue

        # --- Store paths for DPO dataset ---
        if iter_idx == 0:
            iter0_output_file_for_dpo = iter_output_jsonl
        if iter_idx == NUM_ITERATIONS - 1: # This will be the last successfully generated file
            final_iter_output_file_for_dpo = iter_output_jsonl


        # --- ANALYSE TEXTS ---------------------------------------------------
        analysis_results = None
        try:
            analysis_results = analyze_iteration_outputs(
                generated_jsonl_path     = iter_output_jsonl,
                human_profile_full       = human_profile_full,
                iter_analysis_output_dir = iter_analysis_dir,
            )
        except Exception as e:
            print(f"❌ ERROR during text analysis for iteration {iter_idx}: {e}")
            # Log partial stats if generation succeeded but analysis failed
            iteration_stats.append({
                "iteration": iter_idx, "status": "analysis_failed",
                "error": str(e), "output_file": str(iter_output_jsonl),
                "generated_text_count": "N/A", "generated_char_count": "N/A", # Could try to count if file exists
                "ttr": 0.0, "rttr": 0.0, "repetition_per_100k": 0.0,
            })
            continue # Skip ban list updates for this iteration

        if analysis_results is None or analysis_results[0] is None: # Check if analysis returned valid DFs
            print(f"⚠️ Analysis for iteration {iter_idx} did not produce results. Skipping ban list updates and metrics.")
            # Attempt to get basic counts if texts were loaded by analyze_iteration_outputs before it failed
            gen_texts_count = len(analysis_results[4]) if analysis_results and len(analysis_results) > 4 else 0
            gen_chars_count = analysis_results[5] if analysis_results and len(analysis_results) > 5 else 0
            iteration_stats.append({
                "iteration": iter_idx, "status": "analysis_returned_no_data",
                "output_file": str(iter_output_jsonl),
                "generated_text_count": gen_texts_count, "generated_char_count": gen_chars_count,
                "ttr": 0.0, "rttr": 0.0, "repetition_per_100k": 0.0,
            })
            continue
        
        (df_bi_dict, df_bi_nondict, df_tri_dict, df_tri_nondict,
         generated_texts, total_generated_chars) = analysis_results
        
        if not generated_texts: # Should have been caught by analyze_iteration_outputs, but double check
            print(f"⚠️ No generated texts found after analysis for iteration {iter_idx}. Skipping further steps for this iter.")
            iteration_stats.append({
                "iteration": iter_idx, "status": "no_generated_texts_post_analysis",
                "output_file": str(iter_output_jsonl),
                "generated_text_count": 0, "generated_char_count": 0,
                "ttr": 0.0, "rttr": 0.0, "repetition_per_100k": 0.0,
            })
            continue


        # --- OVER-REPRESENTED WORDS (Slop-Forensics) -------------------------
        overrep_tokens_for_ban: list[str] = []
        if COMPUTE_OVERREP_WORDS:
            try:
                overrep_csv: Path = iter_analysis_dir / "overrepresented_words.csv"
                _, dict_words, nodict_words = build_overrep_word_csv(
                    texts   = generated_texts,
                    out_csv = overrep_csv,
                    top_n   = TOP_K_WORDS,
                )
                overrep_tokens_for_ban = select_overrep_words_for_ban(
                    dict_words       = dict_words,
                    nodict_words     = nodict_words,
                    is_first_iteration = (iter_idx == 0)
                )
            except Exception as e:
                print(f"❌ ERROR computing over-represented words for iteration {iter_idx}: {e}")
                # Continue without these words for banning, but log it.

        # --- UPDATE ① N-GRAM BAN LIST (if enabled) ------------------------------
        if ENABLE_NGRAM_BAN:
            try:
                update_banned_ngrams_list(
                    banned_ngrams_json_path,
                    dfs=[df_bi_dict, df_bi_nondict, df_tri_dict, df_tri_nondict],
                    is_first_iteration=(iter_idx == 0),
                    extra_items=EXTRA_NGRAMS_TO_BAN,
                )


            except Exception as e:
                 print(f"❌ ERROR updating N-gram ban list for iteration {iter_idx}: {e}")


        # --- UPDATE ② SLOP-PHRASE BAN LIST (if enabled) -----------------------------------
        if ENABLE_SLOP_PHRASE_BAN:
            try:
                phrases_to_add = TOP_N_INITIAL_SLOP_BAN if iter_idx == 0 else TOP_N_SUBSEQUENT_SLOP_BAN
                update_banned_slop_phrases(
                    json_path   = banned_slop_phrases_json_path,
                    texts       = generated_texts,
                    how_many_new= phrases_to_add,
                    tmp_dir     = iter_analysis_dir / "phrase_tmp", # Subdirectory for temp files
                    over_represented_words= overrep_tokens_for_ban if BAN_OVERREP_WORDS_IN_PHRASE_LIST else None,
                )
            except Exception as e:
                print(f"❌ ERROR updating slop phrase ban list for iteration {iter_idx}: {e}")

        # --- LEXICAL-DIVERSITY & REPETITION METRICS --------------------------
        ttr, rttr = 0.0, 0.0
        repetition_norm = 0.0
        try:
            ttr, rttr = calculate_lexical_diversity_stats(generated_texts)
            repetition_norm = calculate_repetition_score(
                gen_texts     = generated_texts,
                total_chars   = total_generated_chars,
                iteration_dfs = [df_bi_dict, df_bi_nondict, df_tri_dict, df_tri_nondict],
            )
        except Exception as e:
            print(f"❌ ERROR calculating metrics for iteration {iter_idx}: {e}")


        iteration_stats.append({
            "iteration":             iter_idx,
            "status":                "completed",
            "generated_text_count":  len(generated_texts),
            "generated_char_count":  total_generated_chars,
            "ttr":                   ttr,
            "rttr":                  rttr,
            "repetition_per_100k":   repetition_norm,
            "output_file":           str(iter_output_jsonl.name), # Just filename for summary
            "error":                 None
        })
        iter_duration = datetime.datetime.now() - current_iter_start_time
        print(f"--- Iteration {iter_idx} completed in {iter_duration} ---")


    # -------------------------------------------------------------------------#
    # 2) FINAL SUMMARY CSV                                                     #
    # -------------------------------------------------------------------------#
    summary_df: pd.DataFrame = pd.DataFrame(iteration_stats)
    summary_csv: Path        = experiment_dir / "final_iteration_statistics.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n📊  Final statistics written → {summary_csv.resolve()}")
        if not summary_df.empty:
            print(summary_df.to_string(index=False, na_rep="N/A"))
        else:
            print("No iteration statistics were generated.")
    except Exception as e:
        print(f"ERROR: Could not write final statistics CSV: {e}")


    # --------------------------------------------------------------------- #
    # 3)  CALL THE DPO DATASET CREATOR (adjusted from original step 4)      #
    # --------------------------------------------------------------------- #
    if NUM_ITERATIONS >= 1: # Need at least one iteration for iter0, and potentially more for final
        if iter0_output_file_for_dpo and iter0_output_file_for_dpo.exists():
            # If only 1 iteration, final is same as iter0. DPO might not be useful but can be created.
            final_dpo_src = final_iter_output_file_for_dpo if final_iter_output_file_for_dpo and final_iter_output_file_for_dpo.exists() else iter0_output_file_for_dpo
            
            if NUM_ITERATIONS == 1:
                 print("INFO: Only one iteration completed. DPO dataset will use iter_0 for both 'chosen' and 'rejected' if created.")
                 # Or, decide not to create DPO for a single iteration:
                 # print("INFO: Only one iteration. DPO dataset creation skipped as chosen and rejected would be identical.")
                 # return # if skipping DPO for single iter

            dpo_output_jsonl = experiment_dir / "dpo_pairs_dataset.jsonl"
            try:
                create_dpo_dataset(iter0_output_file_for_dpo, final_dpo_src, dpo_output_jsonl)
            except Exception as e:
                print(f"❌ ERROR creating DPO dataset: {e}")

        else:
            print("⚠️  DPO dataset creation skipped: Iteration 0 output file not found or generation failed.")
    else: # NUM_ITERATIONS == 0
        print("⚠️  No iterations run. DPO dataset creation skipped.")
    
    pipeline_duration = datetime.datetime.now() - datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    print(f"\n🏁 Pipeline finished. Total duration: {pipeline_duration}")


if __name__ == "__main__":
    # Check if main.py exists (using the path logic from run_generation_script)
    try:
        main_py_script_path = Path.cwd() / "main.py"
        if not main_py_script_path.exists():
             main_py_script_path_alt = Path.cwd() / "main.py"
             if not (main_py_script_path_alt.exists()):
                print(f"Error: main.py not found. Expected at ./antislop-vllm/main.py relative to CWD ({Path.cwd()}).")
                print("Please ensure main.py (the generation script) is present in the antislop-vllm directory.")
                sys.exit(1) # Exit if main.py is critical and not found
    except Exception as e: # Catch any error during path construction
        print(f"Error checking for main.py: {e}")
        sys.exit(1)

    # Check slop_forensics submodule path again, just before running
    if not _slop_forensics_submodule_path.is_dir():
         print(f"CRITICAL WARNING: Submodule directory '{_slop_forensics_submodule_path}' for slop_forensics not found or not added to sys.path correctly. Pipeline will likely fail.")
         # Decide if to exit or let it try and fail
         # sys.exit(1)

    antislop_pipeline()
