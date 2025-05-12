import json
import subprocess
import sys
import os
import datetime
import logging
from pathlib import Path
import pandas as pd
import nltk # For stopwords in orchestration context
from typing import Optional, Dict, Any, List

from core.analysis import (
    build_overrep_word_csv, select_overrep_words_for_ban,
    update_banned_slop_phrases, analyze_iteration_outputs_core,
    update_banned_ngrams_list, calculate_lexical_diversity_stats,
    calculate_repetition_score
)
from core.dpo import create_dpo_dataset

logger = logging.getLogger(__name__)

def _build_generation_command(
    main_script_path: Path,
    config: Dict[str, Any],
    output_jsonl_path: Path,
    banned_ngrams_file: Optional[Path],
    slop_phrases_file: Optional[Path],
    regex_blocklist_file: Optional[Path]
) -> List[str]:
    """
    Constructs the command list for invoking the antislop-vllm generation script.
    Paths for file arguments are resolved to absolute paths.
    """

    def get_abs_path_str(p: Optional[Path]) -> Optional[str]:
        """Resolves a Path object to an absolute path string, or returns None."""
        return str(p.resolve()) if p else None

    # Ensure essential API base URL is correctly determined
    gen_api_base_url = config.get('generation_api_base_url')
    if not gen_api_base_url:
        # Fallback using vllm_port if generation_api_base_url is missing (should be caught by config validation ideally)
        logger.warning(
            "generation_api_base_url not found in config. Attempting fallback using vllm_port. "
            "It's recommended to explicitly set 'generation_api_base_url' in the configuration."
        )
        vllm_port = config.get('vllm_port', 8000) # Default vLLM port if not in config
        gen_api_base_url = f"http://127.0.0.1:{vllm_port}/v1"


    # Mapping from config keys to CLI flags and value transformations (if any)
    # None as value means the config key directly maps to the flag value as string.
    # A callable as value means it will be called with config[key] to get the CLI value.
    arg_map = {
        "--api-base-url": lambda cfg: gen_api_base_url,
        "--api-key": lambda cfg: cfg['generation_api_key'],
        "--model-name": lambda cfg: cfg['generation_model_id'],
        "--output-jsonl": lambda cfg: get_abs_path_str(output_jsonl_path),
        "--input-hf-dataset": lambda cfg: cfg['generation_hf_dataset_name'],
        "--hf-dataset-split": lambda cfg: cfg['generation_hf_dataset_split'],
        "--threads": lambda cfg: str(cfg['generation_threads']),
        "--max-prompts": lambda cfg: str(cfg['generation_max_prompts']),
        "--logging-level": lambda cfg: cfg['generation_logging_level'],
        "--max-new-tokens": lambda cfg: str(cfg['generation_max_new_tokens']),
        "--chunk-size": lambda cfg: str(cfg['generation_param_chunk_size']),
        "--top-logprobs-count": lambda cfg: str(cfg['generation_param_top_logprobs_count']),
        "--temperature": lambda cfg: str(cfg['generation_param_temperature']),
        "--top-p": lambda cfg: str(cfg['generation_param_top_p']),
        "--top-k": lambda cfg: str(cfg['generation_param_top_k']),
        "--min-p": lambda cfg: str(cfg['generation_param_min_p']),
        "--timeout": lambda cfg: str(cfg['generation_param_timeout']),
        "--max-retries-per-position": lambda cfg: str(cfg['generation_backtracking_max_retries_per_position']),
        "--ngram-remove-stopwords": lambda cfg: str(cfg['generation_ngram_remove_stopwords']).lower(),
        "--ngram-language": lambda cfg: cfg['generation_ngram_language'],
    }

    command = [sys.executable, str(main_script_path)]

    for flag, value_getter in arg_map.items():
        try:
            value = value_getter(config)
            if value is not None: # Ensure value is not None before adding flag and value
                command.extend([flag, value])
        except KeyError as e:
            logger.warning(f"Configuration key {e} missing for CLI argument {flag}. Skipping.")
        except Exception as e:
            logger.error(f"Error processing argument {flag}: {e}. Skipping.")


    # Conditional arguments
    if config.get('generation_param_stop_sequences'):
        stop_sequences_str = ",".join(config['generation_param_stop_sequences'])
        if stop_sequences_str: # Add only if there are actual sequences
            command.extend(["--stop-sequences", stop_sequences_str])
    
    if config.get('generation_chat_template_model_id'):
        command.extend(["--chat-template-model-id", config['generation_chat_template_model_id']])

    if banned_ngrams_file:
        command.extend(["--ngram-banned-file", get_abs_path_str(banned_ngrams_file)])
    
    if slop_phrases_file:
        command.extend(["--slop-phrases-file", get_abs_path_str(slop_phrases_file)])
        # antislop-vllm's --top-n-slop-phrases applies to the file passed via --slop-phrases-file.
        # Using a large number effectively means "use all phrases from the provided file".
        command.extend(["--top-n-slop-phrases", str(999_999)]) 

    if regex_blocklist_file:
        command.extend(["--regex-blocklist-file", get_abs_path_str(regex_blocklist_file)])

    return command

def run_generation_script_wrapper(
    iter_idx: int,
    output_jsonl_path: Path,
    config: Dict[str, Any],
    # experiment_dir is not directly used here if paths are absolute, but good for context
    experiment_dir: Path, 
    banned_ngrams_file_path: Optional[Path] = None,
    slop_phrases_file_path: Optional[Path] = None,
    regex_blocklist_file_path: Optional[Path] = None,
) -> None:
    """
    Manages the execution of the antislop-vllm generation script for a given iteration.
    """
    project_root = Path(__file__).resolve().parent.parent 
    main_py_script = project_root / "antislop-vllm" / "main.py"

    if not main_py_script.exists():
        logger.error(f"Generation script not found: {main_py_script}")
        raise FileNotFoundError(f"antislop-vllm/main.py not found at {main_py_script}. Ensure submodule is present and correctly placed.")

    generation_script_workdir = main_py_script.parent 

    cmd_list = _build_generation_command(
        main_script_path=main_py_script,
        config=config,
        output_jsonl_path=output_jsonl_path,
        banned_ngrams_file=banned_ngrams_file_path,
        slop_phrases_file=slop_phrases_file_path,
        regex_blocklist_file=regex_blocklist_file_path
    )

    # Log the command in a readable format
    # Truncate long paths in the logged command for brevity
    log_cmd_display_list = []
    for item in cmd_list:
        if isinstance(item, str) and ("/" in item or "\\" in item) and len(item) > 70:
            log_cmd_display_list.append(f"...{item[-67:]}") # Show only the end of long paths
        else:
            log_cmd_display_list.append(str(item))
    
    logger.info(f"\n┏━━ Iteration {iter_idx}: Launching antislop-vllm/main.py ━━━━━━━━━━━━━┓")
    logger.info(f"Working Directory: {generation_script_workdir}")
    logger.info(f"Executing Command: {' '.join(log_cmd_display_list)}")
    logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
    
    try:
        # Execute the generation script
        process = subprocess.run(
            cmd_list, 
            cwd=generation_script_workdir, 
            check=False, # We will check returncode manually
        )
        
        logger.info(f"--- End of output from antislop-vllm/main.py (Iteration {iter_idx}) ---")

        if process.returncode != 0:
            error_message = (
                f"antislop-vllm/main.py failed with exit code {process.returncode} for iteration {iter_idx}. "
                f"See output above for details from the script."
            )
            logger.error(error_message)
            raise RuntimeError(error_message)
    
        logger.info(f"✅ antislop-vllm/main.py finished successfully for iteration {iter_idx}.")
        logger.info(f"   Output expected at: {output_jsonl_path.resolve()}")

    except FileNotFoundError:
        logger.error(f"Failed to execute generation script: {sys.executable} or {main_py_script} not found.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while running generation script for iteration {iter_idx}: {e}")
        raise

def orchestrate_pipeline(config: dict, experiment_dir: Path, resume_mode: bool):
    logger.info(f"Starting anti-slop pipeline in directory: {experiment_dir}")
    
    # --- NLTK Stopwords ---
    # Ensure 'english' stopwords are available for analysis functions
    try:
        from nltk.corpus import stopwords
        stop_words_set = set(stopwords.words('english'))
        logger.info(f"Loaded {len(stop_words_set)} NLTK stopwords for 'english'.")
    except LookupError:
        logger.error("NLTK 'stopwords' for 'english' not found. Please run fs_helpers.download_nltk_resource or download manually.")
        logger.error("Pipeline cannot continue without stopwords for analysis.")
        raise
        
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

    # --- Ban Lists Paths ---
    banned_ngrams_json_path = experiment_dir / "banned_ngrams.json"
    banned_slop_phrases_json_path = experiment_dir / config['banned_slop_phrases_filename']
    
    # --- Regex Blocklist (user-supplied, static per run) ---
    banned_regexes_json_path: Optional[Path] = None
    if config.get('extra_regex_patterns'):
        banned_regexes_json_path = experiment_dir / "banned_regexes.json"
        try:
            banned_regexes_json_path.write_text(
                json.dumps(config['extra_regex_patterns'], indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.info(f"📝 User-defined regex blocklist written to {banned_regexes_json_path}")
        except Exception as e:
            logger.error(f"Failed to write regex blocklist: {e}")
            banned_regexes_json_path = None # Disable if write fails

    iteration_stats: list[dict] = []
    iter0_output_file_for_dpo: Optional[Path] = None
    final_iter_output_file_for_dpo: Optional[Path] = None
    
    start_iter_idx = 0
    if resume_mode:
        logger.info(f"Attempting to resume from {experiment_dir}...")
        # Check for last completed iteration's output file to determine start_iter_idx
        # Also load existing ban lists if they exist.
        max_found_iter = -1
        for i in range(config['num_iterations']):
            iter_output_check = experiment_dir / f"iter_{i}_creative_writing_generations.jsonl"
            # A more robust check would be if the corresponding analysis dir also exists and has content
            if iter_output_check.exists() and iter_output_check.stat().st_size > 0:
                max_found_iter = i
                if i == 0: iter0_output_file_for_dpo = iter_output_check
                final_iter_output_file_for_dpo = iter_output_check # Keep updating
            else:
                break # Stop if an iteration's output is missing
        
        if max_found_iter >= 0:
            start_iter_idx = max_found_iter + 1
            logger.info(f"Resuming from iteration {start_iter_idx}.")
            if not banned_ngrams_json_path.exists() and max_found_iter >=0 : logger.info("No existing n-gram ban list found to resume with.")
            if not banned_slop_phrases_json_path.exists() and max_found_iter >=0 : logger.info("No existing slop phrase ban list found to resume with.")
        else:
            logger.info("No completed iterations found to resume. Starting from iteration 0.")
            resume_mode = False # Effectively not resuming if nothing to resume from

    if start_iter_idx >= config['num_iterations']:
        logger.info(f"All {config['num_iterations']} iterations already completed in {experiment_dir}.")
        # Load existing stats if available for DPO step
        summary_csv_path = experiment_dir / "final_iteration_statistics.csv"
        if summary_csv_path.exists():
            try:
                iteration_stats_df = pd.read_csv(summary_csv_path)
                iteration_stats = iteration_stats_df.to_dict('records')
                # Find iter0 and final iter files from stats if not already set
                if not iter0_output_file_for_dpo and not iteration_stats_df.empty:
                    iter0_row = iteration_stats_df[iteration_stats_df['iteration'] == 0]
                    if not iter0_row.empty and 'output_file' in iter0_row.columns:
                         iter0_output_file_for_dpo = experiment_dir / iter0_row.iloc[0]['output_file']
                if not final_iter_output_file_for_dpo and not iteration_stats_df.empty:
                    final_iter_row = iteration_stats_df[iteration_stats_df['iteration'] == config['num_iterations'] -1]
                    if not final_iter_row.empty and 'output_file' in final_iter_row.columns:
                         final_iter_output_file_for_dpo = experiment_dir / final_iter_row.iloc[0]['output_file']
            except Exception as e:
                logger.warning(f"Could not load existing iteration statistics: {e}")
        # Proceed to DPO creation if applicable
    else:
        for iter_idx in range(start_iter_idx, config['num_iterations']):
            current_iter_start_time = datetime.datetime.now()
            logger.info(f"\n{'='*30} ITERATION {iter_idx} (started at {current_iter_start_time.strftime('%H:%M:%S')}) {'='*30}")

            iter_output_jsonl = experiment_dir / f"iter_{iter_idx}_creative_writing_generations.jsonl"
            iter_analysis_dir = experiment_dir / f"iter_{iter_idx}_analysis_results"
            iter_analysis_dir.mkdir(parents=True, exist_ok=True)

            ngram_file_for_cli = banned_ngrams_json_path if iter_idx > 0 and config['enable_ngram_ban'] and banned_ngrams_json_path.exists() else None
            slop_file_for_cli = banned_slop_phrases_json_path if iter_idx > 0 and config['enable_slop_phrase_ban'] and banned_slop_phrases_json_path.exists() else None
            regex_file_for_cli = banned_regexes_json_path if iter_idx > 0 and banned_regexes_json_path and banned_regexes_json_path.exists() else None
            
            try:
                run_generation_script_wrapper(
                    iter_idx=iter_idx, output_jsonl_path=iter_output_jsonl, config=config,
                    experiment_dir=experiment_dir, banned_ngrams_file_path=ngram_file_for_cli,
                    slop_phrases_file_path=slop_file_for_cli, regex_blocklist_file_path=regex_file_for_cli
                )
            except Exception as e:
                logger.error(f"❌ ERROR during text generation (antislop-vllm call) for iteration {iter_idx}: {e}")
                iteration_stats.append({"iteration": iter_idx, "status": "generation_failed", "error": str(e), "output_file": str(iter_output_jsonl.name)})
                if iter_idx == 0: iter0_output_file_for_dpo = None
                final_iter_output_file_for_dpo = None # Mark current final as failed
                continue

            if not iter_output_jsonl.exists() or iter_output_jsonl.stat().st_size == 0:
                logger.error(f"❌ ERROR: Generation output file {iter_output_jsonl} is missing or empty for iteration {iter_idx}.")
                iteration_stats.append({"iteration": iter_idx, "status": "output_file_missing", "output_file": str(iter_output_jsonl.name)})
                if iter_idx == 0: iter0_output_file_for_dpo = None
                final_iter_output_file_for_dpo = None
                continue

            if iter_idx == 0: iter0_output_file_for_dpo = iter_output_jsonl
            final_iter_output_file_for_dpo = iter_output_jsonl # Always update to the latest successful one

            analysis_results = None
            try:
                analysis_results = analyze_iteration_outputs_core(
                    generated_jsonl_path=iter_output_jsonl, human_profile_full=human_profile_full,
                    iter_analysis_output_dir=iter_analysis_dir, config=config, stop_words_set=stop_words_set
                )
            except Exception as e:
                logger.error(f"❌ ERROR during text analysis for iteration {iter_idx}: {e}", exc_info=True)
                iteration_stats.append({"iteration": iter_idx, "status": "analysis_failed", "error": str(e), "output_file": str(iter_output_jsonl.name)})
                continue
            
            if analysis_results is None or analysis_results[0] is None: # DFs are first part of tuple
                logger.warning(f"Analysis for iteration {iter_idx} did not produce results. Skipping ban list updates.")
                iteration_stats.append({"iteration": iter_idx, "status": "analysis_no_data", "output_file": str(iter_output_jsonl.name)})
                continue
            
            df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct, generated_texts, total_gen_chars = analysis_results
            if not generated_texts:
                logger.warning(f"No generated texts found after analysis for iter {iter_idx}. Skipping ban list updates.")
                iteration_stats.append({"iteration": iter_idx, "status": "no_texts_post_analysis", "output_file": str(iter_output_jsonl.name)})
                continue

            overrep_tokens_for_ban: list[str] = []
            if config['compute_overrep_words']:
                try:
                    overrep_csv = iter_analysis_dir / "overrepresented_words.csv"
                    _, dict_words, nodict_words = build_overrep_word_csv(
                        texts=generated_texts, out_csv=overrep_csv,
                        top_n_words_analysis=config['top_k_words_for_overrep_analysis'],
                        stop_words_set=stop_words_set
                    )
                    overrep_tokens_for_ban = select_overrep_words_for_ban(
                        dict_words, nodict_words, (iter_idx == 0), config
                    )
                except Exception as e:
                    logger.error(f"❌ ERROR computing over-represented words for iteration {iter_idx}: {e}", exc_info=True)

            if config['enable_ngram_ban']:
                try:
                    update_banned_ngrams_list(
                        banned_ngrams_json_path,
                        dfs=[df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct],
                        is_first_iteration=(iter_idx == 0), config=config
                    )
                except Exception as e:
                    logger.error(f"❌ ERROR updating N-gram ban list for iteration {iter_idx}: {e}", exc_info=True)

            if config['enable_slop_phrase_ban']:
                try:
                    phrases_to_add = config['top_n_initial_slop_ban'] if iter_idx == 0 else config['top_n_subsequent_slop_ban']
                    update_banned_slop_phrases(
                        json_path=banned_slop_phrases_json_path, texts=generated_texts,
                        how_many_new=phrases_to_add, tmp_dir=iter_analysis_dir / "phrase_tmp",
                        config=config,
                        over_represented_words=overrep_tokens_for_ban if config['ban_overrep_words_in_phrase_list'] else None
                    )
                except Exception as e:
                    logger.error(f"❌ ERROR updating slop phrase ban list for iteration {iter_idx}: {e}", exc_info=True)
            
            ttr, rttr, repetition_norm = 0.0, 0.0, 0.0
            try:
                ttr, rttr = calculate_lexical_diversity_stats(generated_texts, config['min_word_len_for_analysis'])
                repetition_norm = calculate_repetition_score(
                    generated_texts, total_gen_chars,
                    [df_bi_dict, df_bi_nondct, df_tri_dict, df_tri_nondct], config, stop_words_set
                )
            except Exception as e:
                logger.error(f"❌ ERROR calculating metrics for iteration {iter_idx}: {e}", exc_info=True)

            iteration_stats.append({
                "iteration": iter_idx, "status": "completed",
                "generated_text_count": len(generated_texts), "generated_char_count": total_gen_chars,
                "ttr": ttr, "rttr": rttr, "repetition_per_100k_chars": repetition_norm, # Renamed for clarity
                "output_file": str(iter_output_jsonl.name), "error": None
            })
            iter_duration = datetime.datetime.now() - current_iter_start_time
            logger.info(f"--- Iteration {iter_idx} completed in {iter_duration} ---")

    # --- Final Summary & DPO ---
    summary_df = pd.DataFrame(iteration_stats)
    summary_csv = experiment_dir / "final_iteration_statistics.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"\n📊 Final statistics written to {summary_csv.resolve()}")
        if not summary_df.empty: logger.info("\n" + summary_df.to_string(index=False, na_rep="N/A"))
    except Exception as e:
        logger.error(f"Could not write final statistics CSV: {e}")

    if config['num_iterations'] >= 1 and iter0_output_file_for_dpo and final_iter_output_file_for_dpo:
        if iter0_output_file_for_dpo.exists() and final_iter_output_file_for_dpo.exists():
            if config['num_iterations'] == 1:
                logger.info("Only one iteration completed. DPO dataset 'chosen' and 'rejected' will be from the same iter_0 data if created.")
            
            dpo_output_jsonl = experiment_dir / "dpo_pairs_dataset.jsonl"
            try:
                create_dpo_dataset(iter0_output_file_for_dpo, final_iter_output_file_for_dpo, dpo_output_jsonl)
            except Exception as e:
                logger.error(f"❌ ERROR creating DPO dataset: {e}", exc_info=True)
        else:
            logger.warning("DPO dataset creation skipped: Iteration 0 or final iteration output file not found or generation failed.")
    elif config['num_iterations'] < 1:
        logger.info("No iterations run. DPO dataset creation skipped.")
    else:
         logger.warning(f"DPO dataset creation skipped. iter0_output_file_for_dpo: {iter0_output_file_for_dpo}, final_iter_output_file_for_dpo: {final_iter_output_file_for_dpo}")
    
    logger.info("Anti-slop pipeline orchestration finished.")
    return experiment_dir # Return for potential use by finetuning step