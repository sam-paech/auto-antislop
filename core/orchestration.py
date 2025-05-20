import json
import subprocess
import sys
import os
import datetime
import logging
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, List
import traceback

from core.analysis import (
    build_overrep_word_csv, select_overrep_words_for_ban,
    update_banned_slop_phrases, analyze_iteration_outputs_core,
    update_banned_ngrams_list, calculate_lexical_diversity_stats,
    calculate_repetition_score
)
from core.dpo import create_dpo_dataset

logger = logging.getLogger(__name__)

# --- RESUME HELPERS -------------------------------------------------
def _load_prompt_ids(path: Path) -> set[int]:
    """Return the set of prompt_id ints found in an existing generation file."""
    ids = set()
    if not path.is_file():                         # nothing yet
        return ids
    with path.open(encoding="utf-8") as fh:
        for ln in fh:
            try:
                row = json.loads(ln)
                pid = row.get("prompt_id")
                if isinstance(pid, int):
                    ids.add(pid)
            except json.JSONDecodeError:
                continue
    return ids


def _write_missing_prompt_file(missing: list[int], out_dir: Path, iter_idx: int) -> Path:
    """Write one integer per line – the format antislop-vllm already understands."""
    p = out_dir / f"iter_{iter_idx}_missing_prompt_ids.txt"
    p.write_text("\n".join(map(str, missing)), encoding="utf-8")
    return p


def _build_generation_command(
    main_script_path: Path,
    config: Dict[str, Any],
    output_jsonl_path: Path,
    iter_idx: int,
    banned_ngrams_file_for_iter: Optional[Path],
    slop_phrases_file_for_iter: Optional[Path],
    regex_blocklist_file_for_iter: Optional[Path]
) -> List[str]:
    """
    Constructs the command list for invoking the antislop-vllm generation script.

    For iter_idx == 0 (baseline generation), all file-based banning mechanisms
    in antislop-vllm are explicitly disabled by passing empty strings for file paths
    and zero for counts, overriding any defaults in antislop-vllm's local config.yaml.

    For subsequent iterations (iter_idx > 0), it uses the provided ban list file paths.
    Paths for file arguments are resolved to absolute paths.

    Args:
        main_script_path: Absolute path to antislop-vllm/main.py.
        config: The main configuration dictionary for auto-antislop.
        output_jsonl_path: Absolute path for the generation output of this iteration.
        iter_idx: The current iteration index (0-based).
        banned_ngrams_file_for_iter: Path to the n-gram ban list to use for this iteration (if iter_idx > 0).
        slop_phrases_file_for_iter: Path to the slop phrase ban list to use for this iteration (if iter_idx > 0).
        regex_blocklist_file_for_iter: Path to the regex blocklist to use for this iteration (if iter_idx > 0).

    Returns:
        A list of strings representing the command and its arguments.
    """

    def get_abs_path_str(p: Optional[Path]) -> Optional[str]:
        """Resolves a Path object to an absolute path string, or returns None."""
        return str(p.resolve()) if p else None
    
    tdpo_pairs_jsonl_path_str = get_abs_path_str(output_jsonl_path.parent / f"iter_{str(iter_idx)}_tdpo_pairs.jsonl")

    # Determine the API base URL for generation requests
    gen_api_base_url = config.get('generation_api_base_url')
    if not gen_api_base_url:
        vllm_port = config.get('vllm_port', 8000)
        gen_api_base_url = f"http://127.0.0.1:{vllm_port}/v1"
        logger.debug(
            f"generation_api_base_url not explicitly configured, defaulting to {gen_api_base_url} "
            f"based on vllm_port ({vllm_port})."
        )

    # Core command arguments that are always present
    command_base = [
        sys.executable, str(main_script_path),
        "--api-base-url", gen_api_base_url,
        "--api-key", config['generation_api_key'],
        "--model-name", config['generation_model_id'],
        "--config", str((main_script_path.parent / "config-example.yaml").resolve()), # provides pipeline defaults that we aren't passing here
        "--output-jsonl", get_abs_path_str(output_jsonl_path),
        "--input-hf-dataset", config['generation_hf_dataset_name'],
        "--hf-dataset-split", config['generation_hf_dataset_split'],
        "--threads", str(config['generation_threads']),
        "--max-prompts", str(config['generation_max_prompts']),
        "--logging-level", config['generation_logging_level'],
        "--max-new-tokens", str(config['generation_max_new_tokens']),        
        "--top-logprobs-count", str(config['generation_param_top_logprobs_count']),
        "--temperature", str(config['generation_param_temperature']),
        "--top-p", str(config['generation_param_top_p']),
        "--top-k", str(config['generation_param_top_k']),
        "--min-p", str(config['generation_param_min_p']),
        "--timeout", str(config['generation_param_timeout']),
        "--max-retries-per-position", str(config['generation_backtracking_max_retries_per_position']),
        "--force-backtrack", str(config['generation_force_backtrack']),
        "--invert-probs", str(config['generation_invert_probs']),
        "--ngram-remove-stopwords", str(config['generation_ngram_remove_stopwords']).lower(),
        "--ngram-language", config['generation_ngram_language'],
        "--enable-refusal-detection", str(config.get("generation_refusal_detection", False)),
    ]
    command = list(command_base) # Create a mutable copy

    # Use full-length chunks for the baseline run (iter_idx == 0);
    # fall back to the configured chunk size for every later iteration.
    chunk_size = (
        config['generation_max_new_tokens']
        if iter_idx == 0
        else config['generation_param_chunk_size']
    )
    command.extend(["--chunk-size", str(chunk_size)])

    # Optional command arguments based on configuration
    if config.get('generation_param_stop_sequences'):
        stop_sequences_str = ",".join(config['generation_param_stop_sequences'])
        if stop_sequences_str: # Only add if there are actual sequences
            command.extend(["--stop-sequences", stop_sequences_str])
    
    if config.get('generation_chat_template_model_id'):
        command.extend(["--chat-template-model-id", config['generation_chat_template_model_id']])

    # --- Ban list arguments: behavior depends on iteration index ---
    if iter_idx == 0:
        # For iteration 0 (baseline), explicitly disable all file-based banning in antislop-vllm
        # by passing empty strings for file paths and zero for counts.
        # This overrides any defaults in antislop-vllm's local config.yaml.
        logger.debug("Iteration 0: Configuring antislop-vllm for baseline generation (no ban lists).")
        command.extend(["--ngram-banned-file", ""])
        command.extend(["--slop-phrases-file", ""])
        command.extend(["--top-n-slop-phrases", "0"])
        command.extend(["--regex-blocklist-file", ""])
    else:
        # this seems to overlap with another param, should fix
        command.extend(["--tdpo-pairs-jsonl", tdpo_pairs_jsonl_path_str])

        # For iterations > 0, use the ban lists determined by the orchestrate_pipeline function.
        if banned_ngrams_file_for_iter:
            command.extend(["--ngram-banned-file", get_abs_path_str(banned_ngrams_file_for_iter)])
        
        if slop_phrases_file_for_iter:
            command.extend(["--slop-phrases-file", get_abs_path_str(slop_phrases_file_for_iter)])
            # When providing a slop phrases file, instruct antislop-vllm to use all phrases from it.
            command.extend(["--top-n-slop-phrases", str(999_999)]) 
        
        if regex_blocklist_file_for_iter:
            command.extend(["--regex-blocklist-file", get_abs_path_str(regex_blocklist_file_for_iter)])
            
    return command


def run_generation_script_wrapper(
    iter_idx: int,
    output_jsonl_path: Path,
    config: Dict[str, Any],
    banned_ngrams_file_path: Optional[Path] = None,
    slop_phrases_file_path: Optional[Path] = None,
    regex_blocklist_file_path: Optional[Path] = None,
    extra_generation_args: Optional[list[str]] = None,
) -> None:
    """
    Execute antislop-vllm/main.py for a single iteration, handling all paths,
    logging, errors, and now arbitrary extra CLI flags.
    """
    project_root = Path(__file__).resolve().parent.parent
    main_py_script = project_root / "antislop-vllm" / "main.py"
    if not main_py_script.is_file():
        raise FileNotFoundError(
            f"antislop-vllm/main.py not found at {main_py_script}. "
            "Ensure the submodule is present and initialised."
        )

    cmd_list = _build_generation_command(
        main_script_path=main_py_script,
        config=config,
        output_jsonl_path=output_jsonl_path,
        iter_idx=iter_idx,
        banned_ngrams_file_for_iter=banned_ngrams_file_path,
        slop_phrases_file_for_iter=slop_phrases_file_path,
        regex_blocklist_file_for_iter=regex_blocklist_file_path,
    )

    # ── append any ad-hoc flags (e.g. --prompt-id-file <path>) ─────────────
    if extra_generation_args:
        cmd_list.extend(extra_generation_args)

    # pretty-log (truncate very long paths)
    def _short(s: str) -> str:
        return f"...{s[-67:]}" if ("/" in s or "\\" in s) and len(s) > 70 else s
    log_cmd = " ".join(_short(c) for c in cmd_list)

    logger.info(f"\n┏━━ Iteration {iter_idx}: launching antislop-vllm ━━━━━━━━━━━━━┓")
    logger.info(f"cwd: {main_py_script.parent}")
    logger.info(log_cmd)
    logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

    proc = subprocess.run(
        cmd_list,
        cwd=main_py_script.parent,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"antislop-vllm exited with code {proc.returncode} "
            f"(iteration {iter_idx})"
        )

    logger.info(f"✅ antislop-vllm completed for iteration {iter_idx}. "
                f"Output: {output_jsonl_path.name}")


def orchestrate_pipeline(config: Dict[str, Any], experiment_dir: Path, resume_mode: bool):
    logger.info(f"Starting anti-slop pipeline in directory: {experiment_dir}")
    generation_enabled = config.get("generation_step_enabled", True)
    if not generation_enabled:
        logger.info("⚠️  Generation step disabled by config/CLI flag.")

    
    if generation_enabled:
        # --- NLTK Stopwords ---
        try:
            from nltk.corpus import stopwords # Import here to keep it local to this function
            stop_words_set = set(stopwords.words('english'))
            logger.info(f"Loaded {len(stop_words_set)} NLTK stopwords for 'english'.")
        except LookupError:
            logger.error("NLTK 'stopwords' for 'english' not found. Please run fs_helpers.download_nltk_resource or download manually.")
            logger.error("Pipeline cannot continue without stopwords for analysis.")
            raise # Critical for analysis
            
        # --- Human Profile ---
        human_profile_path = Path(config['human_profile_path'])
        if not human_profile_path.is_file():
            logger.error(f"Human profile JSON not found: {human_profile_path.resolve()}")
            raise FileNotFoundError(f"Human profile not found at {human_profile_path}")
        try:
            with human_profile_path.open("r", encoding="utf-8") as f_hp:
                human_profile_full: dict = json.load(f_hp)
        except Exception as e:
            logger.error(f"Could not load or parse human profile JSON '{human_profile_path}': {e}")
            raise

        # --- Ban Lists Paths (initialized here, files created/updated during iterations) ---
        banned_ngrams_json_path = experiment_dir / "banned_ngrams.json"
        if 'banned_slop_phrases_filename' not in config:
            config['banned_slop_phrases_filename'] = 'banned_slop_phrases.json'
        banned_slop_phrases_json_path = experiment_dir / config['banned_slop_phrases_filename']
        
        # --- Regex Blocklist (user-supplied, written once if provided, used from iter 1+) ---
        # This file is created before the loop, but only passed to generation from iter 1.
        user_regex_blocklist_file: Optional[Path] = None # Renamed for clarity
        if config.get('extra_regex_patterns'):
            user_regex_blocklist_file = experiment_dir / "user_defined_regex_blocklist.json"
            try:
                # Write it once if resuming and it doesn't exist, or if not resuming.
                # This ensures it's available for later iterations if resuming.
                if not resume_mode or (resume_mode and not user_regex_blocklist_file.exists()):
                    user_regex_blocklist_file.write_text(
                        json.dumps(config['extra_regex_patterns'], indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                    logger.info(f"📝 User-defined regex blocklist written to {user_regex_blocklist_file}")
                elif user_regex_blocklist_file.exists():
                    logger.info(f"📝 User-defined regex blocklist already exists at {user_regex_blocklist_file}")

            except Exception as e:
                logger.error(f"Failed to write user-defined regex blocklist: {e}. It will not be used.")
                user_regex_blocklist_file = None # Disable if write fails

        iteration_stats: list[dict] = []
        iter0_output_file_for_dpo: Optional[Path] = None
        final_iter_output_file_for_dpo: Optional[Path] = None # Tracks the latest successful output
    
    start_iter_idx = 0
    if resume_mode:
        logger.info(f"Attempting to resume from {experiment_dir}...")
        max_found_iter = -1
        # Check for successfully completed iterations by looking for their output files
        max_found_iter = -1
        need_total = config['generation_max_prompts']

        for i in range(config['num_iterations']):
            gen_file = experiment_dir / f"iter_{i}_creative_writing_generations.jsonl"

            # Does the file exist at all?
            if not (gen_file.is_file() and gen_file.stat().st_size):
                break                                   # nothing (or zero-length) → not complete

            # Does it contain the full prompt set?
            ids_seen = _load_prompt_ids(gen_file)
            if len(ids_seen) < need_total:
                logger.info(
                    f"Iteration {i} resume-check: {len(ids_seen)}/{need_total} prompts present "
                    f"({need_total-len(ids_seen)} still missing).")
                break                                   # incomplete → resume here

            max_found_iter = i                         # this one is done, keep going

        
        if max_found_iter >= 0:
            start_iter_idx = max_found_iter + 1
            logger.info(f"Resuming from iteration {start_iter_idx}.")
            # Log presence of existing ban lists if resuming past iter 0
            if start_iter_idx > 0:
                if banned_ngrams_json_path.exists(): logger.info(f"Resuming with existing n-gram ban list: {banned_ngrams_json_path}")
                else: logger.info("No existing n-gram ban list found to resume with for subsequent iterations.")
                if banned_slop_phrases_json_path.exists(): logger.info(f"Resuming with existing slop phrase ban list: {banned_slop_phrases_json_path}")
                else: logger.info("No existing slop phrase ban list found to resume with for subsequent iterations.")
        else:
            logger.info("No fully completed iterations found to resume. Starting from iteration 0.")
            # resume_mode = False # No need to change resume_mode, start_iter_idx handles it

    if start_iter_idx >= config['num_iterations']:
        logger.info(f"All {config['num_iterations']} iterations appear to be completed in {experiment_dir}.")
        # Attempt to load existing stats for DPO if needed
        summary_csv_path = experiment_dir / "final_iteration_statistics.csv"
        if summary_csv_path.exists():
            try:
                iteration_stats_df = pd.read_csv(summary_csv_path)
                iteration_stats = iteration_stats_df.to_dict('records')
                # Ensure iter0_output_file_for_dpo and final_iter_output_file_for_dpo are set if possible
                if not iter0_output_file_for_dpo and not iteration_stats_df.empty:
                    iter0_row = iteration_stats_df[iteration_stats_df['iteration'] == 0]
                    if not iter0_row.empty and 'output_file' in iter0_row.columns:
                         path_str = iter0_row.iloc[0]['output_file']
                         if path_str and isinstance(path_str, str): iter0_output_file_for_dpo = experiment_dir / path_str
                if not final_iter_output_file_for_dpo and not iteration_stats_df.empty:
                    # Find the last completed iteration in the stats
                    last_stat_iter = iteration_stats_df['iteration'].max()
                    final_iter_row = iteration_stats_df[iteration_stats_df['iteration'] == last_stat_iter]
                    if not final_iter_row.empty and 'output_file' in final_iter_row.columns:
                         path_str = final_iter_row.iloc[0]['output_file']
                         if path_str and isinstance(path_str, str): final_iter_output_file_for_dpo = experiment_dir / path_str
            except Exception as e:
                logger.warning(f"Could not load or parse existing iteration statistics from {summary_csv_path}: {e}")
        # Proceed to DPO creation if applicable (handled after the loop)
    else: # Need to run some or all iterations
        if generation_enabled:
            for iter_idx in range(start_iter_idx, config['num_iterations']):
                current_iter_start_time = datetime.datetime.now()
                logger.info(f"\n{'='*30} ITERATION {iter_idx} (started at {current_iter_start_time.strftime('%H:%M:%S')}) {'='*30}")

                iter_output_jsonl = experiment_dir / f"iter_{iter_idx}_creative_writing_generations.jsonl"
                iter_analysis_dir = experiment_dir / f"iter_{iter_idx}_analysis_results"
                iter_analysis_dir.mkdir(parents=True, exist_ok=True) # Ensure analysis dir exists

                # --- Determine ban lists for the current iteration ---
                # Iteration 0 (baseline) runs with NO BANNING.
                # Subsequent iterations use the ban lists accumulated so far.
                ngram_file_for_generation: Optional[Path] = None
                slop_file_for_generation: Optional[Path] = None
                regex_file_for_generation: Optional[Path] = None

                if iter_idx > 0: # Banning starts from iteration 1
                    if config['enable_ngram_ban'] and banned_ngrams_json_path.exists():
                        ngram_file_for_generation = banned_ngrams_json_path
                    if config['enable_slop_phrase_ban'] and banned_slop_phrases_json_path.exists():
                        slop_file_for_generation = banned_slop_phrases_json_path
                    if user_regex_blocklist_file and user_regex_blocklist_file.exists(): # User-defined regex
                        regex_file_for_generation = user_regex_blocklist_file
                
                if iter_idx == 0:
                    logger.info("Iteration 0: Running baseline generation with NO ban lists.")
                else:
                    logger.info(f"Iteration {iter_idx}: Using ban lists - N-grams: {ngram_file_for_generation}, Slop: {slop_file_for_generation}, Regex: {regex_file_for_generation}")

                # ────────────────────────────────────────────────────────────────────
                #  A. fast-path – is generation already complete?
                # ────────────────────────────────────────────────────────────────────
                existing_ids  = _load_prompt_ids(iter_output_jsonl)
                need_total    = config['generation_max_prompts']
                missing_ids   = sorted(set(range(need_total)) - existing_ids)

                if not missing_ids:
                    logger.info(f"Iteration {iter_idx}: found {need_total} / {need_total} prompts "
                                f"in {iter_output_jsonl.name} – skipping generation step.")
                else:
                    logger.info(f"Iteration {iter_idx}: {len(missing_ids)} / {need_total} prompts "
                                f"still missing – resuming generation.")
                    # antislop-vllm already supports '--prompt-id-file' (one id per line)
                    # we can skip this as antislop-vllm automatically resumes now
                    #miss_file = _write_missing_prompt_file(missing_ids, experiment_dir, iter_idx)

                    try:
                        run_generation_script_wrapper(
                            iter_idx               = iter_idx,
                            output_jsonl_path      = iter_output_jsonl,
                            config                 = config,
                            banned_ngrams_file_path= ngram_file_for_generation,
                            slop_phrases_file_path = slop_file_for_generation,
                            regex_blocklist_file_path = regex_file_for_generation,
                            #extra_generation_args  = ["--prompt-id-file", str(miss_file)]
                        )
                    except Exception as e:
                        logger.error(f"❌ Generation script failed for iteration {iter_idx}: {e}")
                        # identical failure-handling block as before …
                        iteration_stats.append({
                            "iteration": iter_idx, "status": "generation_failed",
                            "error": str(e), "output_file": str(iter_output_jsonl.name)
                        })
                        if iter_idx == 0:
                            iter0_output_file_for_dpo = None
                        continue


                if not iter_output_jsonl.exists() or iter_output_jsonl.stat().st_size == 0:
                    logger.error(f"❌ Generation output file {iter_output_jsonl} is missing or empty for iteration {iter_idx}.")
                    iteration_stats.append({
                        "iteration": iter_idx, "status": "output_file_missing_or_empty", 
                        "output_file": str(iter_output_jsonl.name)
                    })
                    if iter_idx == 0: iter0_output_file_for_dpo = None
                    continue

                # Update DPO file pointers
                if iter_idx == 0: 
                    iter0_output_file_for_dpo = iter_output_jsonl
                # final_iter_output_file_for_dpo always points to the latest successfully generated file
                final_iter_output_file_for_dpo = iter_output_jsonl 

                # --- Analysis (runs for all iterations, including iter 0 to find initial slop) ---
                analysis_results = None
                try:
                    analysis_results = analyze_iteration_outputs_core(
                        generated_jsonl_path=iter_output_jsonl, 
                        human_profile_full=human_profile_full,
                        iter_analysis_output_dir=iter_analysis_dir, 
                        config=config, 
                        stop_words_set=stop_words_set
                    )
                except Exception as e:
                    logger.error(f"❌ Text analysis failed for iteration {iter_idx}: {e}", exc_info=True)
                    iteration_stats.append({
                        "iteration": iter_idx, "status": "analysis_failed", 
                        "error": str(e), "output_file": str(iter_output_jsonl.name)
                    })
                    continue 
                
                if analysis_results is None or analysis_results[0] is None: # DFs are first part of tuple
                    logger.warning(f"Analysis for iteration {iter_idx} did not produce data. Skipping ban list updates for this iteration.")
                    iteration_stats.append({
                        "iteration": iter_idx, "status": "analysis_no_data", 
                        "output_file": str(iter_output_jsonl.name)
                    })
                    continue
                
                df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct, generated_texts, total_gen_chars = analysis_results
                if not generated_texts:
                    logger.warning(f"No generated texts found after analysis for iter {iter_idx}. Skipping ban list updates.")
                    iteration_stats.append({
                        "iteration": iter_idx, "status": "no_texts_post_analysis", 
                        "output_file": str(iter_output_jsonl.name)
                    })
                    continue

                # --- Update Ban Lists (based on current iteration's analysis) ---
                # These lists will be used by the *next* iteration's generation step.
                # --- Update ban lists (based on current iteration's analysis) --------------
                overrep_tokens_for_ban: list[str] = []
                iter_log = iter_analysis_dir / "orchestration.log"
                def _iter_log(msg: str) -> None:
                    with iter_log.open("a", encoding="utf-8") as fh:
                        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")

                # (a) over-represented words -------------------------------------------------
                if config['compute_overrep_words']:
                    try:
                        overrep_csv = iter_analysis_dir / "overrepresented_words.csv"
                        _, dict_words, nodict_words = build_overrep_word_csv(
                            texts=generated_texts,
                            out_csv=overrep_csv,
                            top_n_words_analysis=config['top_k_words_for_overrep_analysis'],
                            stop_words_set=stop_words_set,
                        )
                        overrep_tokens_for_ban = select_overrep_words_for_ban(
                            dict_words, nodict_words, (iter_idx == 0), config
                        )
                        _iter_log(f"overrep_tokens_for_ban = {len(overrep_tokens_for_ban)}")
                    except Exception as exc:
                        _iter_log("❌ build_overrep_word_csv failed:\n" +
                                "".join(traceback.format_exception_only(type(exc), exc)))

                # (b) n-gram ban list --------------------------------------------------------
                if config['enable_ngram_ban']:
                    try:
                        update_banned_ngrams_list(
                            banned_ngrams_json_path,
                            dfs=[df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct],
                            is_first_iteration=(iter_idx == 0),
                            config=config,
                        )
                        _iter_log("n-gram ban list updated")
                    except Exception as exc:
                        _iter_log("❌ update_banned_ngrams_list failed:\n" +
                                "".join(traceback.format_exception_only(type(exc), exc)))

                # (c) slop-phrase ban list ---------------------------------------------------
                if config['enable_slop_phrase_ban']:
                    try:
                        phrases_to_add_count = (
                            config['top_n_initial_slop_ban'] if iter_idx == 0
                            else config['top_n_subsequent_slop_ban']
                        )
                        update_banned_slop_phrases(
                            json_path=banned_slop_phrases_json_path,
                            texts=generated_texts,
                            how_many_new=phrases_to_add_count,
                            tmp_dir=iter_analysis_dir / "phrase_tmp",
                            config=config,
                            over_represented_words=(
                                overrep_tokens_for_ban if
                                config['ban_overrep_words_in_phrase_list'] else None
                            ),
                        )
                        _iter_log("slop-phrase ban list updated  "
                                f"(+{len(overrep_tokens_for_ban)} over-rep words)")
                    except Exception as exc:
                        _iter_log("❌ update_banned_slop_phrases failed:\n" +
                                "".join(traceback.format_exception_only(type(exc), exc)))


                # --- Calculate Metrics for this iteration ---
                ttr, rttr, repetition_norm = 0.0, 0.0, 0.0
                try:
                    ttr, rttr = calculate_lexical_diversity_stats(generated_texts, config['min_word_len_for_analysis'])
                    repetition_norm = calculate_repetition_score(
                        generated_texts, total_gen_chars,
                        [df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct], config, stop_words_set
                    )
                except Exception as e:
                    logger.error(f"❌ Error calculating metrics for iteration {iter_idx}: {e}", exc_info=True)

                iteration_stats.append({
                    "iteration": iter_idx, "status": "completed",
                    "generated_text_count": len(generated_texts), "generated_char_count": total_gen_chars,
                    "ttr": ttr, "rttr": rttr, "repetition_per_100k_chars": repetition_norm,
                    "output_file": str(iter_output_jsonl.name), "error": None
                })
                iter_duration = datetime.datetime.now() - current_iter_start_time
                logger.info(f"--- Iteration {iter_idx} completed in {iter_duration} ---")
        else:
            logger.info("Skipping generation loop.")
    
    if generation_enabled:
        # --- Final Summary & DPO Dataset Creation ---
        summary_df = pd.DataFrame(iteration_stats)
        summary_csv = experiment_dir / "final_iteration_statistics.csv"
        try:
            summary_df.to_csv(summary_csv, index=False)
            logger.info(f"\n📊 Final statistics written to {summary_csv.resolve()}")
            if not summary_df.empty: 
                # Ensure all columns are displayed if possible
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                    logger.info("\n" + summary_df.to_string(index=False, na_rep="N/A"))
            else:
                logger.info("No iteration statistics were generated to summarize.")
        except Exception as e:
            logger.error(f"Could not write final statistics CSV to {summary_csv}: {e}")

        # DPO dataset creation logic
        if config['num_iterations'] >= 1 and iter0_output_file_for_dpo and final_iter_output_file_for_dpo:
            if iter0_output_file_for_dpo.exists() and final_iter_output_file_for_dpo.exists():
                if config['num_iterations'] == 1 and iter0_output_file_for_dpo == final_iter_output_file_for_dpo:
                    logger.warning("Only one iteration completed. DPO dataset 'chosen' and 'rejected' will be from the same iter_0 data. This might not be useful for training.")
                
                dpo_output_jsonl = experiment_dir / "dpo_pairs_dataset.jsonl"
                try:
                    create_dpo_dataset(iter0_output_file_for_dpo, final_iter_output_file_for_dpo, dpo_output_jsonl)
                except Exception as e:
                    logger.error(f"❌ ERROR creating DPO dataset: {e}", exc_info=True)
            else:
                logger.warning(
                    f"DPO dataset creation skipped: Iteration 0 output file ({iter0_output_file_for_dpo}) "
                    f"or final iteration output file ({final_iter_output_file_for_dpo}) not found or generation failed."
                )
        elif config['num_iterations'] < 1:
            logger.info("No iterations were configured to run. DPO dataset creation skipped.")
        else: # Cases where DPO files might be None due to errors
            logger.warning(
                f"DPO dataset creation skipped due to missing DPO source files. "
                f"Iter0 source: {iter0_output_file_for_dpo}, Final iter source: {final_iter_output_file_for_dpo}"
            )
    
    logger.info("Anti-slop pipeline orchestration finished.")
    return experiment_dir