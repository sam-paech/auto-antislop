# utils/config_loader.py

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Any

import yaml

logger = logging.getLogger(__name__)


_ALWAYS: Sequence[str] = (
    # minimal required keys for the pipeline to run at all
    "experiment_base_dir",
    "human_profile_path",
    "num_iterations",
    "min_word_len_for_analysis",
    "freq_norm_denom_for_analysis",
    "top_n_repetition_stat",
    "log_level"
)

_VLLM: Sequence[str] = (
    "vllm_model_id",
    "vllm_port",
    "vllm_hf_token",
    "vllm_cuda_visible_devices",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_len",
    "vllm_dtype",
    "vllm_extra_args",
)

_GENERATION: Sequence[str] = (
    "generation_api_key",
    "generation_api_base_url",       # needed if you do local or remote calls
    "generation_model_id",
    "generation_max_new_tokens",
    "generation_threads",
    "generation_max_prompts",
    "generation_hf_dataset_name",
    "generation_hf_dataset_split",
    "generation_logging_level",
    "generation_chat_template_model_id",
    "generation_param_chunk_size",
    "generation_param_top_logprobs_count",
    "generation_param_temperature",
    "generation_param_top_p",
    "generation_param_top_k",
    "generation_param_min_p",
    "generation_param_timeout",
    "generation_param_stop_sequences",
    "generation_ngram_remove_stopwords",
    "generation_ngram_language",
    "generation_force_backtrack",
    "generation_invert_probs",
    "generation_prompt_template",
    "generation_system_prompt"

)

_NGRAM: Sequence[str] = (
    "top_k_bigrams",
    "top_k_trigrams",
    "dict_bigrams_initial",
    "dict_bigrams_subsequent",
    "nodict_bigrams_initial",
    "nodict_bigrams_subsequent",
    "dict_trigrams_initial",
    "dict_trigrams_subsequent",
    "nodict_trigrams_initial",
    "nodict_trigrams_subsequent",
    "extra_ngrams_to_ban",
)

_SLOP: Sequence[str] = (
    "ban_overrep_words_in_phrase_list",
    "min_phrase_freq_to_keep",
    "top_n_initial_slop_ban",
    "top_n_subsequent_slop_ban",
    "extra_slop_phrases_to_ban",
    "banned_slop_phrases_filename",
)

_OVERREP: Sequence[str] = (
    "top_k_words_for_overrep_analysis",
    "dict_overrep_initial",
    "dict_overrep_subsequent",
    "nodict_overrep_initial",
    "nodict_overrep_subsequent",
)

_FINETUNE: Sequence[str] = (
    "finetune_mode",
    "finetune_ftpo_dataset",
    "finetune_base_model_id",
    "finetune_max_seq_length",
    "finetune_load_in_4bit",
    "finetune_lora_r",
    "finetune_lora_alpha",
    "finetune_lora_dropout",
    "finetune_weight_decay",
    "finetune_target_modules",
    "finetune_gradient_checkpointing",
    "finetune_chat_template",
    "finetune_batch_size",
    "finetune_gradient_accumulation_steps",
    "finetune_warmup_ratio",
    "finetune_num_epochs",
    "finetune_learning_rate",
    "finetune_auto_learning_rate",
    "finetune_beta",
    "finetune_output_dir_suffix",
    "finetune_save_merged_16bit",
    "finetune_save_gguf_q8_0",
    "finetune_max_train_examples",
    "finetune_ftpo_sample_regularisation_strength",
    "finetune_cuda_visible_devices",
)


def _deep_update(dst: Dict, src: Dict) -> Dict:
    """Recursively merge src into dst (src wins)."""
    for k, v in src.items():
        if k in dst and isinstance(v, dict) and isinstance(dst[k], dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst

def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    """Load config from a YAML file, or return empty dict if missing/invalid."""
    if config_path and config_path.exists():
        try:
            with config_path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", config_path)
            return data
        except Exception as e:
            logger.error("Could not load %s: %s – using empty config", config_path, e)
    else:
        logger.info("Config file %s not found – using empty config", config_path)
    return {}

def merge_config_with_cli_args(config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merges every possible CLI parameter from your old DEFAULT_CONFIG
    into 'config' if the user actually provided it (i.e. it's not None).
    Also merges housekeeping flags (config_file, resume_from_dir, log_level).
    """
    merged = copy.deepcopy(config)

    # 1. Housekeeping arguments (not originally in DEFAULT_CONFIG, but we keep them if set)
    if getattr(cli_args, 'config_file', None) is not None:
        merged['config_file'] = cli_args.config_file
    if getattr(cli_args, 'resume_from_dir', None) is not None:
        merged['resume_from_dir'] = cli_args.resume_from_dir
    if getattr(cli_args, 'log_level', None) is not None:
        merged['log_level'] = cli_args.log_level

    # 2. Booleans that map from CLI flags to known keys in config
    if getattr(cli_args, 'run_finetune', None) is not None:
        merged['finetune_enabled'] = cli_args.run_finetune
    if getattr(cli_args, 'manage_vllm', None) is not None:
        merged['manage_vllm'] = cli_args.manage_vllm
    if getattr(cli_args, 'generation_step_enabled', None) is not None:
        merged['generation_step_enabled'] = cli_args.generation_step_enabled
    if getattr(cli_args, "finetune_cuda_visible_devices", None) is not None:
        merged["finetune_cuda_visible_devices"] = cli_args.finetune_cuda_visible_devices


    # 3. All remaining keys from the old DEFAULT_CONFIG
    _all_groups: Sequence[Sequence[str]] = (
        _ALWAYS,
        _VLLM,
        _GENERATION,
        _NGRAM,
        _SLOP,
        _OVERREP,
        _FINETUNE,
    )
    all_config_keys: List[str] = [k for group in _all_groups for k in group]

    # Overwrite config if user specified a value
    for key in all_config_keys:
        cli_val = getattr(cli_args, key, None)
        if cli_val is not None:
            merged[key] = cli_val

    return merged

# ---------------------------------------------------------------------------
# Validate with partial requirements depending on which features are enabled
# ---------------------------------------------------------------------------

def _missing(cfg: Dict[str, Any], keys: Sequence[str]) -> List[str]:
    return [k for k in keys if k not in cfg]

def validate_config(cfg: Dict[str, Any]) -> None:
    """Raise ValueError if any required config is missing based on pipeline flags."""
    missing = []
    # always
    missing.extend(_missing(cfg, _ALWAYS))

    # vllm
    if cfg.get("manage_vllm", False):
        missing.extend(_missing(cfg, _VLLM))

    # generation
    if cfg.get("generation_step_enabled", True):
        missing.extend(_missing(cfg, _GENERATION))

    # n-gram ban
    if cfg.get("enable_ngram_ban", False):
        missing.extend(_missing(cfg, _NGRAM))

    # slop phrase ban
    if cfg.get("enable_slop_phrase_ban", False):
        missing.extend(_missing(cfg, _SLOP))

    # over-rep analysis
    if cfg.get("compute_overrep_words", False):
        missing.extend(_missing(cfg, _OVERREP))

    # finetuning
    if cfg.get("finetune_enabled", False):
        missing.extend(_missing(cfg, _FINETUNE))

    if missing:
        raise ValueError(
            f"Configuration is incomplete; missing these keys: {', '.join(sorted(set(missing)))}"
        )
    logger.info("Configuration validated – all required keys present (for enabled features).")

def load_merge_validate(config_path: Path, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    1) Load YAML from config_path,
    2) Merge in any CLI flags user typed,
    3) Validate that all needed keys for enabled features are present.
    """
    cfg = load_pipeline_config(config_path)
    cfg = merge_config_with_cli_args(cfg, cli_args)
    validate_config(cfg)
    return cfg
