import json
import subprocess
import sys
import os
import datetime
import logging
from pathlib import Path
import pandas as pd
import nltk # For stopwords in orchestration context
from typing import Optional

from core.analysis import (
    build_overrep_word_csv, select_overrep_words_for_ban,
    update_banned_slop_phrases, analyze_iteration_outputs_core,
    update_banned_ngrams_list, calculate_lexical_diversity_stats,
    calculate_repetition_score
)
from core.dpo import create_dpo_dataset

logger = logging.getLogger(__name__)

def run_generation_script_wrapper(
    iter_idx: int,
    output_jsonl_path: Path,
    config: dict,
    experiment_dir: Path, # For resolving relative banlist paths
    banned_ngrams_file_path: Optional[Path] = None,
    slop_phrases_file_path: Optional[Path] = None,
    regex_blocklist_file_path: Optional[Path] = None,
) -> None:
    """Invokes antislop-vllm/main.py with dynamically constructed CLI arguments."""

    # Locate main.py relative to this project's root
    project_root = Path(__file__).resolve().parent.parent 
    main_py = project_root / "antislop-vllm" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"antislop-vllm/main.py not found at {main_py}. Ensure submodule is present.")

    workdir = main_py.parent # antislop-vllm directory
    
    # Helper to make paths relative to antislop-vllm workdir if they are absolute
    # or already relative to project root.
    def rel_to_workdir(p: Optional[Path]) -> Optional[str]:
        if p is None: return None
        # If p is already relative and meant to be inside workdir (e.g. "banlists/file.json")
        # and workdir is antislop-vllm, then it's fine.
        # If p is absolute (e.g. from experiment_dir), make it relative to workdir.
        if p.is_absolute():
            try:
                return os.path.relpath(p.resolve(), workdir.resolve())
            except ValueError: # Happens if paths are on different drives (Windows)
                return str(p.resolve()) # Pass absolute path if relpath fails
        # If p is relative (e.g. to project root), resolve it then make relative to workdir
        # This assumes 'p' might be like Path("results/run_xyz/banned_ngrams.json")
        # relative_to_project_root = project_root / p
        # return os.path.relpath(relative_to_project_root.resolve(), workdir.resolve())
        # Simpler: assume paths passed are already absolute from experiment_dir
        return str(p)


    cmd = [
        sys.executable, str(main_py),
        "--api-base-url", f"http://127.0.0.1:{config['vllm_port']}/v1",
        "--api-key", config['generation_api_key'],
        "--model-name", config['generation_model_id'],
        "--output-jsonl", rel_to_workdir(output_jsonl_path),
        "--input-hf-dataset", config['generation_hf_dataset_name'],
        "--hf-dataset-split", config['generation_hf_dataset_split'],
        "--threads", str(config['generation_threads']),
        "--max-prompts", str(config['generation_max_prompts']),
        "--logging-level", config['generation_logging_level'],
        "--max-new-tokens", str(config['generation_max_new_tokens']),
        # Generation params
        "--chunk-size", str(config['generation_param_chunk_size']),
        "--top-logprobs-count", str(config['generation_param_top_logprobs_count']),
        "--temperature", str(config['generation_param_temperature']),
        "--top-p", str(config['generation_param_top_p']),
        "--top-k", str(config['generation_param_top_k']),
        "--min-p", str(config['generation_param_min_p']),
        "--timeout", str(config['generation_param_timeout']),
        # Backtracking
        "--max-retries-per-position", str(config['generation_backtracking_max_retries_per_position']),
        # N-gram validator (file path is managed here)
        "--ngram-remove-stopwords", str(config['generation_ngram_remove_stopwords']).lower(),
        "--ngram-language", config['generation_ngram_language'],
    ]

    if config['generation_param_stop_sequences']:
        cmd += ["--stop-sequences", ",".join(config['generation_param_stop_sequences'])]
    
    if config.get('generation_chat_template_model_id'):
        cmd += ["--chat-template-model-id", config['generation_chat_template_model_id']]

    if banned_ngrams_file_path:
        cmd += ["--ngram-banned-file", rel_to_workdir(banned_ngrams_file_path)]
    
    if slop_phrases_file_path:
        cmd += ["--slop-phrases-file", rel_to_workdir(slop_phrases_file_path)]
        # antislop-vllm's --top-n-slop-phrases applies to its *own* config file,
        # not the one we pass. If we pass a file, it uses all phrases in it.
        # The notebook's TOP_N_INITIAL_SLOP_BAN is for *creating* the ban list.
        # For now, assume antislop-vllm uses all phrases from the provided file.
        # If antislop-vllm needs a top_n for a *passed* file, its CLI needs an update.
        # The notebook used top_n_slop_phrase_flag = 999_999 if slop_phrase_file_for_cli else 0
        # This implies antislop-vllm's main.py has a --top-n-slop-phrases that applies to the --slop-phrases-file.
        # Checking antislop-vllm/main.py: yes, it has --top-n-slop-phrases.
        cmd += ["--top-n-slop-phrases", str(999_999)] # Effectively use all from file

    if regex_blocklist_file_path:
        cmd += ["--regex-blocklist-file", rel_to_workdir(regex_blocklist_file_path)]

    logger.info(f"\n┏━━ Iteration {iter_idx}: launching antislop-vllm/main.py ━━━━━━━━━━━━━┓")
    logger.info(f"Workdir: {workdir}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
    
    process = subprocess.run(cmd, cwd=workdir, check=False) # check=False to handle errors manually
    
    if process.returncode != 0:
        logger.error(f"antislop-vllm/main.py failed with exit code {process.returncode}")
        # stderr/stdout might be useful but can be very verbose.
        # Consider capturing and logging them if errors are common.
        raise RuntimeError(f"antislop-vllm/main.py execution failed for iteration {iter_idx}.")
    
    logger.info(f"✅ antislop-vllm/main.py finished for iteration {iter_idx} — output: {output_jsonl_path}")


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