import argparse
import logging
import sys
import os
from pathlib import Path
import datetime # For pipeline duration

# ── make utils importable ────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))          # so "utils" is on sys.path

# ── guarantee NLTK data is present *before* any other project import ─
from utils.fs_helpers import ensure_core_nltk_resources
ensure_core_nltk_resources()               # downloads punkt, punkt_tab, stopwords


# --- Add project directories to sys.path ---
# This allows importing from core, utils, and submodules
sys.path.insert(0, str(ROOT_DIR / "slop-forensics"))
# antislop-vllm is called as a script, its path for direct import is not strictly needed
# unless some of its utils were to be imported by auto-antislop (not the current plan).

from utils.config_loader import load_pipeline_config, merge_config_with_cli_args
from utils.fs_helpers import (
    create_experiment_dir,
    ensure_antislop_vllm_config_exists
)
from utils.vllm_manager import start_vllm_server, stop_vllm_server, is_vllm_server_alive
from core.orchestration import orchestrate_pipeline
from core.finetuning import run_dpo_finetune

# --- Basic Logging Setup ---
# Will be refined based on CLI args
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("auto_antislop_main")


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Auto-Antislop: Iterative dataset generation and DPO finetuning.")
    
    # --- General Arguments ---
    parser.add_argument(
        "--config-file", type=Path, default=Path("auto_antislop_config.yaml"),
        help="Path to the main YAML configuration file."
    )
    parser.add_argument(
        "--resume-from-dir", type=Path, default=None,
        help="Path to an existing experiment run directory to resume."
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None, help="Set the logging level for the auto-antislop script."
    )

    # --- vLLM Management ---
    vllm_group = parser.add_argument_group('vLLM Server Management')
    vllm_group.add_argument(
        "--manage-vllm",
        type=str2bool,
        nargs="?",
        const=True,                          # `--manage-vllm` ⇒ True
        default=None,                        # fall back to config
        help="true/false to let this script start/stop a local vLLM server "
            "(default comes from config)."
    )
    vllm_group.add_argument("--vllm-port", type=int, default=None, help="Port for vLLM server. Overrides config.")
    vllm_group.add_argument("--vllm-model-id", type=str, default=None, help="Model ID for vLLM server. Overrides config.")
    vllm_group.add_argument(
        "--generation-api-base-url", type=str,
        default=None,
        help="API base URL for generation requests (passed to antislop-vllm). E.g., http://host:port/v1. Overrides config."
    )

    # --- Pipeline Control ---
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument("--num-iterations", type=int, default=None, help="Number of anti-slop iterations. Overrides config.")
    pipeline_group.add_argument("--generation-max-prompts", type=int, default=None, help="Max prompts for antislop-vllm. Overrides config.")
    pipeline_group.add_argument(
        "--generation-step-enabled",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="true/false to execute the generation step. "
            "(default from config)."
    )

    # --- Finetuning Control ---
    finetune_group = parser.add_argument_group('DPO Finetuning Control')
    finetune_group.add_argument(
        "--run-finetune",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="true/false to run DPO finetuning after the pipeline "
            "(default from config)."
    )

    finetune_group.add_argument("--finetune-base-model-id", type=str, default=None, help="Base model for DPO. Overrides config.")
    finetune_group.add_argument("--finetune-num-epochs", type=int, default=None, help="Number of epochs for DPO. Overrides config.")

    finetune_group.add_argument(
        "--finetune-mode",
        choices=["dpo", "tdpo"],
        default=None,
        help="dpo = vanilla DPO on full continuations (default); "
            "tdpo = masked Tokenwise-DPO on partial generation pairs, only computing loss for the completion token."
    )
    finetune_group.add_argument(
        "--finetune-tdpo-dataset",
        type=Path,
        default=None,
        help="(Optional) explicit path to a TDPO/last-token JSONL file. "
            "If omitted and --finetune-mode is tdpo, the script will "
            "pick the highest iter_*_tdpo_pairs.jsonl in the experiment dir."
    )

    args = parser.parse_args()

        # --- Load and Merge Configuration ---
    config = load_pipeline_config(args.config_file)
    config = merge_config_with_cli_args(config, args)

    # --- Refine Logging Setup ---
    numeric_log_level = getattr(logging, config['log_level'].upper(), logging.INFO)
    # Get all loggers that might have been created by imports
    for name in logging.root.manager.loggerDict:
        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(numeric_log_level)
        # Ensure handlers also respect this level or are more verbose
        for handler in logger_instance.handlers:
            handler.setLevel(min(numeric_log_level, handler.level)) # Don't make handler less verbose than logger
    logging.getLogger().setLevel(numeric_log_level) # Root logger
    logger.info(f"Logging level set to: {config['log_level'].upper()}")




    # --- Ensure NLTK resources ---
    # These are used by core.analysis
    # --- Ensure *all* NLTK resources are present *before* anything else ---
    logger.info("Verifying / downloading required NLTK data …")
    ensure_core_nltk_resources()
    
    # --- Ensure antislop-vllm config-example is copied (user convenience) ---
    antislop_vllm_dir = ROOT_DIR / "antislop-vllm"
    if antislop_vllm_dir.is_dir():
        ensure_antislop_vllm_config_exists(antislop_vllm_dir)
    else:
        logger.warning(f"antislop-vllm submodule directory not found at {antislop_vllm_dir}. Generation will likely fail.")


    # --- vLLM Server Management ---
    vllm_server_proc = None
    should_manage_vllm = config.get('manage_vllm', True) # Default to True if not in args/config
    
    if should_manage_vllm:
        if not is_vllm_server_alive(config['vllm_port']):
            logger.info("Attempting to start and manage vLLM server.")
            vllm_server_proc = start_vllm_server(
                model_id=config['vllm_model_id'],
                port=config['vllm_port'],
                hf_token=config.get('vllm_hf_token'),
                cuda_visible_devices=config['vllm_cuda_visible_devices'],
                gpu_memory_utilization=config['vllm_gpu_memory_utilization'],
                max_model_len=config['vllm_max_model_len'],
                dtype=config['vllm_dtype'],
                vllm_extra_args=config.get('vllm_extra_args'),
                uvicorn_log_level="error",          # <-- cut vllm chatter
                quiet_stdout=True,          # <-- discard server stream
            )
            if vllm_server_proc is None: # Failed to start
                logger.error("Failed to start managed vLLM server. Exiting.")
                sys.exit(1)
        else:
            logger.info(f"vLLM server already running on port {config['vllm_port']}. Script will not manage it.")
            should_manage_vllm = False # Don't try to stop it later
    else:
        logger.info("vLLM server management is disabled by config/CLI.")
        if not is_vllm_server_alive(config['vllm_port']):
            logger.warning(f"vLLM server management disabled, but no server found on port {config['vllm_port']}. "
                           "The generation pipeline will likely fail. Please start a vLLM server manually.")


    # --- Main Pipeline ---
    pipeline_start_time = datetime.datetime.now()
    experiment_run_dir = None
    try:
        base_dir = Path(config['experiment_base_dir'])
        resume_dir_path = Path(config['resume_from_dir']) if config.get('resume_from_dir', None) else None
        experiment_run_dir = create_experiment_dir(base_dir, resume_dir_path)
        
        # Pass the actual experiment_run_dir to orchestrate_pipeline
        config['current_experiment_run_dir'] = str(experiment_run_dir)

        orchestrate_pipeline(config, experiment_run_dir, resume_mode=(resume_dir_path is not None))

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}. Halting pipeline.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the anti-slop pipeline: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pipeline_duration = datetime.datetime.now() - pipeline_start_time
        logger.info(f"Total anti-slop pipeline duration: {pipeline_duration}")

    # --- Finetuning (Optional) ---
    should_run_finetune = config.get('finetune_enabled', False)

    if should_run_finetune:
        if experiment_run_dir:
            # NEW: shut down vLLM so the GPU is free for training
            if should_manage_vllm and vllm_server_proc:
                logger.info("Stopping managed vLLM server before finetuning.")
                stop_vllm_server(vllm_server_proc)
                vllm_server_proc = None          # prevent a second stop later

            logger.info("Proceeding to finetuning.")
            finetune_start_time = datetime.datetime.now()
            try:
                run_dpo_finetune(config, experiment_run_dir)
            except Exception as e:
                logger.error("An error occurred during finetuning: %s", e, exc_info=True)
            finally:
                finetune_duration = datetime.datetime.now() - finetune_start_time
                logger.info("Total finetuning duration: %s", finetune_duration)
        else:
            logger.warning("Skipping finetuning as the main pipeline did not complete successfully or experiment directory is not set.")
    else:
        logger.info("inetuning is disabled by config/CLI or due to pipeline issues.")


if __name__ == "__main__":
    main()