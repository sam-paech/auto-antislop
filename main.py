import argparse
import logging
import sys
import os
from pathlib import Path
import datetime # For pipeline duration

# --- Add project directories to sys.path ---
# This allows importing from core, utils, and submodules
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR)) # For core and utils
sys.path.insert(0, str(ROOT_DIR / "slop-forensics"))
# antislop-vllm is called as a script, its path for direct import is not strictly needed
# unless some of its utils were to be imported by auto-antislop (not the current plan).

from utils.config_loader import load_pipeline_config, merge_config_with_cli_args, DEFAULT_CONFIG
from utils.fs_helpers import create_experiment_dir, download_nltk_resource, ensure_antislop_vllm_config_exists
from utils.vllm_manager import start_vllm_server, stop_vllm_server, is_vllm_server_alive
from core.orchestration import orchestrate_pipeline
from core.finetuning import run_dpo_finetune

# --- Basic Logging Setup ---
# Will be refined based on CLI args
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("auto_antislop_main")

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
        default="INFO", help="Set the logging level for the auto-antislop script."
    )

    # --- vLLM Management ---
    vllm_group = parser.add_argument_group('vLLM Server Management')
    vllm_group.add_argument(
        "--manage-vllm", action=argparse.BooleanOptionalAction, default=None, # Default from config
        help="Let this script start/stop a local vLLM server. Overrides config."
    )
    vllm_group.add_argument("--vllm-port", type=int, help="Port for vLLM server. Overrides config.")
    vllm_group.add_argument("--vllm-model-id", type=str, help="Model ID for vLLM server. Overrides config.")
    vllm_group.add_argument( # <<< ADDED
        "--generation-api-base-url", type=str,
        help="API base URL for generation requests (passed to antislop-vllm). E.g., http://host:port/v1. Overrides config."
    )

    # --- Pipeline Control ---
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument("--num-iterations", type=int, help="Number of anti-slop iterations. Overrides config.")
    pipeline_group.add_argument("--generation-max-prompts", type=int, help="Max prompts for antislop-vllm. Overrides config.")

    # --- Finetuning Control ---
    finetune_group = parser.add_argument_group('DPO Finetuning Control')
    finetune_group.add_argument(
        "--run-finetune", action=argparse.BooleanOptionalAction, default=None, # Default from config
        help="Run DPO finetuning after pipeline. Overrides config."
    )
    finetune_group.add_argument("--finetune-base-model-id", type=str, help="Base model for DPO. Overrides config.")
    finetune_group.add_argument("--finetune-num-epochs", type=int, help="Number of epochs for DPO. Overrides config.")

    # Add more CLI overrides for other config keys as needed.
    # For simplicity, this example keeps CLI overrides minimal, relying more on the config file.
    # To add all, iterate DEFAULT_CONFIG keys and add_argument for each.

    args = parser.parse_args()

    # --- Refine Logging Setup ---
    numeric_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    # Get all loggers that might have been created by imports
    for name in logging.root.manager.loggerDict:
        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(numeric_log_level)
        # Ensure handlers also respect this level or are more verbose
        for handler in logger_instance.handlers:
            handler.setLevel(min(numeric_log_level, handler.level)) # Don't make handler less verbose than logger
    logging.getLogger().setLevel(numeric_log_level) # Root logger
    logger.info(f"Logging level set to: {args.log_level.upper()}")


    # --- Load and Merge Configuration ---
    config = load_pipeline_config(args.config_file)
    config = merge_config_with_cli_args(config, args)

    # --- Ensure NLTK resources ---
    # These are used by core.analysis
    logger.info("Checking NLTK resources...")
    download_nltk_resource('tokenizers/punkt', 'punkt')
    download_nltk_resource('corpora/stopwords', 'stopwords')
    
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
                vllm_extra_args=config.get('vllm_extra_args')
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
        resume_dir_path = Path(args.resume_from_dir) if args.resume_from_dir else None
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

    # --- DPO Finetuning (Optional) ---
    should_run_finetune = config.get('finetune_enabled_by_default', False)
    if args.run_finetune is not None: # CLI overrides config default
        should_run_finetune = args.run_finetune

    if should_run_finetune:
        if experiment_run_dir: # Ensure pipeline ran and dir exists
            logger.info("Proceeding to DPO finetuning.")
            finetune_start_time = datetime.datetime.now()
            try:
                run_dpo_finetune(config, experiment_run_dir)
            except Exception as e:
                logger.error(f"An error occurred during DPO finetuning: {e}", exc_info=True)
            finally:
                finetune_duration = datetime.datetime.now() - finetune_start_time
                logger.info(f"Total DPO finetuning duration: {finetune_duration}")
        else:
            logger.warning("Skipping DPO finetuning as the main pipeline did not complete successfully or experiment directory is not set.")
    else:
        logger.info("DPO finetuning is disabled by config/CLI or due to pipeline issues.")

    # --- Stop vLLM Server (if managed) ---
    if should_manage_vllm and vllm_server_proc:
        stop_vllm_server(vllm_server_proc)

    logger.info("Auto-Antislop script finished.")

if __name__ == "__main__":
    main()