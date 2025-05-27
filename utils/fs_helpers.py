import nltk
import datetime
from pathlib import Path
import logging
import sys
import json
import shutil
from typing import List

logger = logging.getLogger(__name__)



def merge_custom_bans_into_file(path: Path, extra_items: List[str]) -> None:
    """Merge extra_items into `path`, preserving original on-disk format."""

    # 1) read whatever is already there
    try:
        current_raw = json.loads(path.read_text("utf-8")) if path.exists() else []
    except Exception:
        current_raw = []

    if not isinstance(current_raw, list):
        current_raw = []

    # 2) normalise existing items → plain strings
    existing: set[str] = set()
    slop_format = False               # do we need to write back [[phrase,1]] ?

    for entry in current_raw:
        if isinstance(entry, list):   # slop-phrase style [phrase, freq]
            slop_format = True
            if entry:                 # non-empty list
                existing.add(str(entry[0]))
        else:                         # plain string
            existing.add(str(entry))

    # 3) merge & sort
    merged = sorted(existing | set(map(str, extra_items)))

    # 4) write back in the same shape we found
    if slop_format:
        payload = [[p, 1] for p in merged]
    else:
        payload = merged

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")


def download_nltk_resource(resource_id: str, resource_name: str):
    """Downloads NLTK resource if not found."""
    try:
        nltk.data.find(resource_id)
        logger.debug(f"NLTK '{resource_name}' resource found.")
    except LookupError:
        logger.info(f"NLTK '{resource_name}' resource not found. Downloading...")
        try:
            nltk.download(resource_name, quiet=True)
            logger.info(f"NLTK '{resource_name}' resource downloaded.")
        except Exception as e:
            logger.warning(f"Could not automatically download NLTK '{resource_name}' resource: {e}. "
                           "Manual download might be required (e.g., python -m nltk.downloader punkt stopwords)")
    except Exception as e:
        logger.warning(f"Error checking NLTK '{resource_name}' resource: {e}.")

def create_experiment_dir(base_dir_path: Path, resume_dir: Path | None = None) -> Path:
    """
    Determine the directory that the pipeline should work in.

    • When --resume-from-dir is supplied we MUST use that exact path.
      If the directory is missing or not a directory, raise immediately.

    • When no resume dir is given, create a new timestamped directory under
      *base_dir_path* (parents created as needed) and return it.
    """
    if resume_dir is not None:
        if resume_dir.is_dir():
            logger.info(f"Resuming experiment in existing directory: {resume_dir.resolve()}")
            return resume_dir
        # hard-fail: the user explicitly asked to resume here
        raise FileNotFoundError(
            f"--resume-from-dir was set to '{resume_dir}', "
            "but that path does not exist or is not a directory."
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_dir_path / f"run_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Created experiment directory: {experiment_dir.resolve()}")
    return experiment_dir


def ensure_antislop_vllm_config_exists(antislop_vllm_dir: Path):
    """
    Copies antislop-vllm/config-example.yaml to config.yaml if config.yaml is absent.
    This is a helper for users, but the main pipeline will pass params via CLI.
    """
    cfg_path = antislop_vllm_dir / "config.yaml"
    example_path = antislop_vllm_dir / "config-example.yaml"

    if not cfg_path.exists():
        if example_path.exists():
            try:
                shutil.copy(example_path, cfg_path)
                logger.info(f"Copied {example_path} to {cfg_path} for antislop-vllm (user convenience).")
            except Exception as e:
                logger.warning(f"Could not copy antislop-vllm config example: {e}")
        else:
            logger.debug("antislop-vllm/config-example.yaml not found. No default config.yaml created for it.")
    else:
        logger.debug("antislop-vllm/config.yaml already exists.")


###############################################################################
# NLTK helpers
###############################################################################
CORE_NLTK_RESOURCES = [
    ("tokenizers/punkt",       "punkt"),        # sentence + word tokeniser data
    ("tokenizers/punkt_tab",   "punkt_tab"),    # new in NLTK 3.9+, used by PunktTokenizer
    ("corpora/stopwords",      "stopwords"),    # obvious
]

def ensure_core_nltk_resources() -> None:
    """
    Download the three NLTK resources our pipeline needs *once* at start-up.
    Safe to call multiple times – it‘s a no-op if they’re already present.
    """
    for resource_id, resource_name in CORE_NLTK_RESOURCES:
        download_nltk_resource(resource_id, resource_name)
