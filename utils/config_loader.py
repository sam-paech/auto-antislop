import yaml
import argparse
import copy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Disable the hardcoded default config for now because it can be confusing
DEFAULT_CONFIG = {}

def _deep_update(source, overrides):
    """
    Update a nested dictionary or an argparse.Namespace with values from another dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source:
            if isinstance(source[key], dict):
                _deep_update(source[key], value)
            else: # Trying to override a non-dict with a dict
                source[key] = copy.deepcopy(value)
        else:
            source[key] = copy.deepcopy(value)
    return source


def load_pipeline_config(config_path: Path) -> dict:
    """Loads config from YAML, merges with defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                config = _deep_update(config, yaml_config)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.info(f"Configuration file {config_path} is empty. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}. Using defaults.")
    else:
        logger.info("No config ser = argparse.ArgumentParserfile specified or found. Using default configuration.")
    return config

def merge_config_with_cli_args(config: dict, cli_args: argparse.Namespace) -> dict:
    """Merges CLI arguments into the config. CLI args take precedence."""
    merged_config = copy.deepcopy(config)
    cli_args_dict = vars(cli_args)

    for key, value in cli_args_dict.items():
        if value is not None: # Only override if CLI arg was actually provided
            # Check if this key exists in DEFAULT_CONFIG or is a special CLI arg
            
            if key in DEFAULT_CONFIG or key in [
                "config_file", "resume_from_dir", "run_finetune", "log_level",
                "generation_api_base_url", "generation_step_enabled"
            ]:
                merged_config[key] = value


    # Handle specific boolean flags that might not be in DEFAULT_CONFIG but are CLI-only controls
    if hasattr(cli_args, 'run_finetune') and cli_args.run_finetune is not None:
        merged_config['finetune_enabled'] = cli_args.run_finetune
    if hasattr(cli_args, 'manage_vllm') and cli_args.manage_vllm is not None:
        merged_config['manage_vllm'] = cli_args.manage_vllm
    if getattr(cli_args, 'generation_step_enabled', None) is not None:
        merged_config['generation_step_enabled'] = cli_args.generation_step_enabled



    return merged_config