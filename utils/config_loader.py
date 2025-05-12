import yaml
import argparse
import copy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # Experiment Setup
    "experiment_base_dir": "results/auto_antislop_runs",
    "human_profile_path": "data/human_writing_profile.json",

    # vLLM Server Management
    "manage_vllm": True,
    "vllm_model_id": "unsloth/gemma-3-1b-it",
    "vllm_port": 8000,
    "vllm_hf_token": None,
    "vllm_cuda_visible_devices": "0",
    "vllm_gpu_memory_utilization": 0.90,
    "vllm_max_model_len": 8192,
    "vllm_dtype": "bfloat16",
    "vllm_extra_args": [],

    # Iterative Anti-Slop Pipeline
    "num_iterations": 2,

    # Generation Script (antislop-vllm/main.py) Parameters
    "generation_model_id": "unsloth/gemma-3-1b-it",
    "generation_api_key": "xxx",
    "generation_max_new_tokens": 2000,
    "generation_threads": 8,
    "generation_max_prompts": 80,
    "generation_hf_dataset_name": 'Nitral-AI/Reddit-SFW-Writing_Prompts_ShareGPT',
    "generation_hf_dataset_split": 'train',
    "generation_logging_level": 'INFO',
    "generation_chat_template_model_id": None,
    "generation_param_chunk_size": 50,
    "generation_param_top_logprobs_count": 20,
    "generation_param_temperature": 0.7,
    "generation_param_top_p": 1.0,
    "generation_param_top_k": 50,
    "generation_param_min_p": 0.05,
    "generation_param_timeout": 120,
    "generation_param_stop_sequences": [],
    "generation_backtracking_max_retries_per_position": 20,
    "generation_ngram_remove_stopwords": True,
    "generation_ngram_language": "english",

    # N-Gram Analysis & Banning
    "enable_ngram_ban": True,
    "top_k_bigrams": 5000,
    "top_k_trigrams": 5000,
    "dict_bigrams_initial": 400, "dict_bigrams_subsequent": 70,
    "nodict_bigrams_initial": 800, "nodict_bigrams_subsequent": 100,
    "dict_trigrams_initial": 300, "dict_trigrams_subsequent": 50,
    "nodict_trigrams_initial": 800, "nodict_trigrams_subsequent": 100,
    "extra_ngrams_to_ban": [],

    # Over-Represented Word Analysis & Banning
    "compute_overrep_words": True,
    "top_k_words_for_overrep_analysis": 200000,
    "dict_overrep_initial": 800, "dict_overrep_subsequent": 200,
    "nodict_overrep_initial": 80, "nodict_overrep_subsequent": 20,

    # Slop Phrase Banning
    "enable_slop_phrase_ban": True,
    "ban_overrep_words_in_phrase_list": True,
    "min_phrase_freq_to_keep": 2,
    "top_n_initial_slop_ban": 600,
    "top_n_subsequent_slop_ban": 100,
    "extra_slop_phrases_to_ban": [],
    "banned_slop_phrases_filename": "banned_slop_phrases.json", # Added for consistency

    # Regex Banning
    "extra_regex_patterns": [],

    # Metrics
    "min_word_len_for_analysis": 3,
    "freq_norm_denom_for_analysis": 100000,
    "top_n_repetition_stat": 50,

    # DPO Finetuning
    "finetune_enabled_by_default": False,
    "finetune_base_model_id": "unsloth/gemma-3-1b-it",
    "finetune_max_seq_length": 2048,
    "finetune_load_in_4bit": True,
    "finetune_lora_r": 16,
    "finetune_lora_alpha": 32,
    "finetune_lora_dropout": 0.05,
    "finetune_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "finetune_gradient_checkpointing": "unsloth",
    "finetune_chat_template": "gemma-3",
    "finetune_batch_size": 1,
    "finetune_gradient_accumulation_steps": 4,
    "finetune_warmup_ratio": 0.1,
    "finetune_num_epochs": 1,
    "finetune_learning_rate": 5e-5,
    "finetune_beta": 0.1,
    "finetune_output_dir_suffix": "_dpo_finetuned",
    "finetune_save_merged_16bit": False,
    "finetune_save_gguf_q8_0": False,
}

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
        logger.info("No config file specified or found. Using default configuration.")
    return config

def merge_config_with_cli_args(config: dict, cli_args: argparse.Namespace) -> dict:
    """Merges CLI arguments into the config. CLI args take precedence."""
    merged_config = copy.deepcopy(config)
    cli_args_dict = vars(cli_args)

    for key, value in cli_args_dict.items():
        if value is not None: # Only override if CLI arg was actually provided
            # Check if this key exists in DEFAULT_CONFIG to avoid adding unrelated argparse internals
            if key in DEFAULT_CONFIG:
                 merged_config[key] = value
            elif key == "config_file" or key == "resume_from_dir" or key == "run_finetune" or key == "log_level": # Special CLI args
                 merged_config[key] = value


    # Handle specific boolean flags that might not be in DEFAULT_CONFIG but are CLI-only controls
    if hasattr(cli_args, 'run_finetune') and cli_args.run_finetune is not None:
        merged_config['finetune_enabled_by_default'] = cli_args.run_finetune
    if hasattr(cli_args, 'manage_vllm') and cli_args.manage_vllm is not None:
        merged_config['manage_vllm'] = cli_args.manage_vllm


    return merged_config