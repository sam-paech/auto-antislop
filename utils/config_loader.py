"""
utils/config_loader.py
----------------------

Load → merge → validate configuration for the Auto-Antislop pipeline.

Order of precedence
1. explicit CLI flags
2. YAML file given with --config-file (or auto_antislop_config.yaml)
3. nothing  →  validation will fail if a required key is absent
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. YAML loader                                                              #
# --------------------------------------------------------------------------- #

def _deep_update(dst: Dict, src: Dict) -> Dict:
    """Recursively copy *src* into *dst* (src wins)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_pipeline_config(path: Path) -> Dict:
    if path and path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", path)
            return data
        except Exception as e:
            logger.error("Failed to load %s: %s – using empty config", path, e)
    else:
        logger.info("Config file %s not found – using empty config", path)
    return {}


# --------------------------------------------------------------------------- #
# 2. CLI merge – copy only flags the user supplied                            #
# --------------------------------------------------------------------------- #

_SPECIAL_CLI_ARGS = {"config_file", "resume_from_dir", "log_level"}

def merge_config_with_cli_args(base: Dict, cli_ns: argparse.Namespace) -> Dict:
    merged = copy.deepcopy(base)
    for k, v in vars(cli_ns).items():
        if k in _SPECIAL_CLI_ARGS or v is None:
            continue                         # skip bookkeeping args & unspecified flags
        merged[k] = v

    # single-flag conveniences
    if getattr(cli_ns, "run_finetune", None) is not None:
        merged["finetune_enabled"] = cli_ns.run_finetune
    if getattr(cli_ns, "manage_vllm", None) is not None:
        merged["manage_vllm"] = cli_ns.manage_vllm
    if getattr(cli_ns, "generation_step_enabled", None) is not None:
        merged["generation_step_enabled"] = cli_ns.generation_step_enabled

    return merged


# --------------------------------------------------------------------------- #
# 3. Validation                                                               #
# --------------------------------------------------------------------------- #
# Keys taken from your former DEFAULT_CONFIG.  Groups are enabled only
# when the corresponding feature-flag in *cfg* is truthy.

_ALWAYS: Sequence[str] = (
    "experiment_base_dir",
    "human_profile_path",
    "num_iterations",
)

_VLLM: Sequence[str] = (
    "vllm_model_id", "vllm_port", "vllm_cuda_visible_devices",
    "vllm_gpu_memory_utilization", "vllm_max_model_len", "vllm_dtype",
)

_GENERATION: Sequence[str] = (
    "generation_api_key", "generation_model_id",
    "generation_hf_dataset_name", "generation_hf_dataset_split",
    "generation_threads", "generation_max_prompts",
    "generation_param_chunk_size", "generation_param_temperature",
    "generation_param_top_p", "generation_param_top_k", "generation_param_min_p",
    "generation_param_timeout",
)

_NGRAM: Sequence[str] = (
    "top_k_bigrams", "top_k_trigrams",
    "dict_bigrams_initial", "dict_bigrams_subsequent",
    "nodict_bigrams_initial", "nodict_bigrams_subsequent",
    "dict_trigrams_initial", "dict_trigrams_subsequent",
    "nodict_trigrams_initial", "nodict_trigrams_subsequent",
)

_SLOP: Sequence[str] = (
    "min_phrase_freq_to_keep", "top_n_initial_slop_ban", "top_n_subsequent_slop_ban",
)

_OVERREP: Sequence[str] = (
    "top_k_words_for_overrep_analysis", "dict_overrep_initial",
    "dict_overrep_subsequent", "nodict_overrep_initial", "nodict_overrep_subsequent",
)

_FINETUNE: Sequence[str] = (
    "finetune_base_model_id", "finetune_max_seq_length",
    "finetune_load_in_4bit", "finetune_lora_r", "finetune_lora_alpha",
    "finetune_lora_dropout", "finetune_num_epochs", "finetune_learning_rate",
)

def _missing(cfg: Dict, keys: Sequence[str]) -> List[str]:
    return [k for k in keys if k not in cfg]

def validate_config(cfg: Dict) -> None:
    missing: List[str] = _missing(cfg, _ALWAYS)

    if cfg.get("manage_vllm", False):
        missing += _missing(cfg, _VLLM)

    if cfg.get("generation_step_enabled", True):
        missing += _missing(cfg, _GENERATION)

    if cfg.get("enable_ngram_ban", False):
        missing += _missing(cfg, _NGRAM)

    if cfg.get("enable_slop_phrase_ban", False):
        missing += _missing(cfg, _SLOP)

    if cfg.get("compute_overrep_words", False):
        missing += _missing(cfg, _OVERREP)

    if cfg.get("finetune_enabled", False):
        missing += _missing(cfg, _FINETUNE)

    if missing:
        raise ValueError(
            "Configuration incomplete – missing keys: "
            + ", ".join(sorted(set(missing)))
        )
    logger.info("Configuration validated – all required keys present.")


# --------------------------------------------------------------------------- #
# 4. Convenience one-shot helper                                              #
# --------------------------------------------------------------------------- #

def load_merge_validate(config_path: Path, cli_ns: argparse.Namespace) -> Dict:
    cfg = load_pipeline_config(config_path)
    cfg = merge_config_with_cli_args(cfg, cli_ns)
    validate_config(cfg)
    return cfg
