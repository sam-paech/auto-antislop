# utils/trainer_dataloaders.py

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset, Dataset
from utils.dataset_helpers import load_ftpo_multi_dataset

if TYPE_CHECKING:
    from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_and_prepare_dataset(config: dict, experiment_run_dir: Path, tokenizer: "AutoTokenizer") -> Dataset | None:
    """
    Loads and prepares the dataset based on the finetuning mode specified in the config.

    Args:
        config (dict): The experiment configuration dictionary.
        experiment_run_dir (Path): The directory for the current experiment run.
        tokenizer (AutoTokenizer): The tokenizer to use for processing.

    Returns:
        Dataset or None: The prepared Hugging Face dataset, or None if loading fails.
    """
    mode = config.get("finetune_mode", "ftpo").lower()
    max_seq_length = config['finetune_max_seq_length']
    dpo_dataset_hf = None

    if mode == "dpo":
        # full-sequence preference pairs: rejected is baseline; chosen is the generation made with antislop
        dataset_path = experiment_run_dir / "dpo_pairs_dataset.jsonl"
        if not dataset_path.is_file():
            logger.error(f"DPO dataset not found at {dataset_path}")
            return None

        dpo_dataset_hf = load_dataset(
            "json",
            data_files=str(dataset_path),
            split="train"
        )

        # ----------------------------------------------------------
        #   discard rows whose prompt+continuation would overflow
        # ----------------------------------------------------------
        def _within_len(example):
            prompt_ids = tokenizer(example["prompt"],
                                add_special_tokens=False).input_ids
            chosen_ids = tokenizer(example["chosen"],
                                add_special_tokens=False).input_ids
            rejected_ids = tokenizer(example["rejected"],
                                    add_special_tokens=False).input_ids
            max_len = config['finetune_max_seq_length']
            return (
                len(prompt_ids) + len(chosen_ids) <= max_len
                and
                len(prompt_ids) + len(rejected_ids) <= max_len
            )

        before = len(dpo_dataset_hf)
        dpo_dataset_hf = dpo_dataset_hf.filter(_within_len)
        after  = len(dpo_dataset_hf)
        logger.info(f"DPO length filter: kept {after}/{before} examples "
                    f"(max_seq_len = {config['finetune_max_seq_length']})")

        if after == 0:
            raise ValueError("every DPO sample exceeded finetune_max_seq_length")


        dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))
        max_train = config.get("finetune_max_train_examples")
        if isinstance(max_train, int) and max_train > 0 and len(dpo_dataset_hf) > max_train:
            dpo_dataset_hf = dpo_dataset_hf.select(range(max_train))
            logger.info(f"Capped training dataset to {max_train} examples.")

        # ── filter malformed rows (prompt / chosen / rejected missing) ──
        req_cols = {"prompt", "chosen", "rejected"}
        before_len = len(dpo_dataset_hf)
        dpo_dataset_hf = dpo_dataset_hf.filter(
            lambda x: all(col in x and x[col] for col in req_cols)
        )
        after_len = len(dpo_dataset_hf)
        if after_len == 0:
            logger.error("All rows in DPO dataset were filtered out. Check contents.")
            return None
        if after_len < before_len:
            logger.info(f"Filtered out {before_len - after_len} malformed rows; "
                        f"{after_len} remain.")
        logger.info(f"DPO dataset ready with {after_len} samples.")

    elif mode == "ftpo":
        if config.get("finetune_ftpo_dataset"):
            dataset_path = Path(config["finetune_ftpo_dataset"])
        else:
            ftpo_files = sorted(experiment_run_dir.glob("iter_*_ftpo_pairs.jsonl"))
            if not ftpo_files:
                logger.error("No ftpo files found for ftpo.")
                return None
            dataset_path = ftpo_files[-1]

        # ------------------------------------------------------------------
        # FTPO dataset with dual regularisation + built-in size cap
        # ------------------------------------------------------------------
        dpo_dataset_hf = load_ftpo_multi_dataset(
            dataset_path,
            tokenizer,
            max_seq_len         = max_seq_length,
            # balance *rejected* tokens
            rejected_reg_strength = config.get("ftpo_sample_rejected_regularisation_strength", 0.8),
            # balance *chosen* tokens
            chosen_reg_strength   = config.get("ftpo_sample_chosen_regularisation_strength", 0.2),
            # hard floor on |chosen|
            min_chosen_tokens     = config.get("ftpo_sample_min_chosen_tokens", 3),
            # overall training-set cap (used for per-token quotas too)
            max_train_examples    = config.get("finetune_max_train_examples"),
        )

        # loader already returns a shuffled dataset; an extra shuffle is fine but optional
        dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))


        
        # ──────────────────────────────────────────────────────────────
        # [DEBUG] Inspect last-5 prompt tokens + chosen / rejected token
        #         –– prints up to 50 ftpo examples for a quick sanity check.
        #         –– gated by new config flag `finetune_debug_ftpo_tokens`.
        # ──────────────────────────────────────────────────────────────
        if False:
            sample_n = min(50, len(dpo_dataset_hf))
            print(f"\n🔎 ftpo debug: showing {sample_n} examples "
                "(last-5 prompt tokens, chosen ▸ rejected)\n")
            for i, ex in enumerate(dpo_dataset_hf.select(range(sample_n))):
                tail_prompt = tokenizer.convert_ids_to_tokens(ex["prompt_ids"][-5:])
                chosen_tok  = tokenizer.convert_ids_to_tokens([ex["chosen_ids"][0]])[0]
                rejected_tok = tokenizer.convert_ids_to_tokens([ex["rejected_token_id"]])[0]
                tail_str = " ".join(tail_prompt)
                print(f"{i:03d}: … {tail_str}  →  {chosen_tok} ▸ {rejected_tok}")
            print("\n—— end ftpo debug ——\n")

    elif mode == "dpo_final_token":
        # ------------------------------------------------------------
        # 1. Build the raw dataset **exactly** the same way FTPO does
        # ------------------------------------------------------------
        if config.get("finetune_ftpo_dataset"):
            dataset_path = Path(config["finetune_ftpo_dataset"])
        else:
            ftpo_files = sorted(experiment_run_dir.glob("iter_*_ftpo_pairs.jsonl"))
            if not ftpo_files:
                logger.error("No ftpo files found for dpo_final_token.")
                return None
            dataset_path = ftpo_files[-1]

        ftpo_ds = load_ftpo_multi_dataset(
            dataset_path,
            tokenizer,
            max_seq_len                  = max_seq_length,
            rejected_reg_strength        = config.get("ftpo_sample_rejected_regularisation_strength", 0.8),
            chosen_reg_strength          = config.get("ftpo_sample_chosen_regularisation_strength", 0.2),
            min_chosen_tokens            = config.get("ftpo_sample_min_chosen_tokens", 3),
            max_train_examples           = config.get("finetune_max_train_examples"),
        )

        # ------------------------------------------------------------
        # 2. Convert each row into a *single-token* DPO pair
        # ------------------------------------------------------------
        pairs = []
        pad_id = tokenizer.pad_token_id

        for ex in ftpo_ds:
            # ––– recover the left-padded prompt as text –––
            prompt_ids = [tid for tid in ex["prompt_ids"] if tid != pad_id]
            prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

            # ––– single-token continuations –––
            chosen_txt   = tokenizer.decode(
                [ex["chosen_ids"][0]], skip_special_tokens=False
            )
            rejected_txt = tokenizer.decode(
                [ex["rejected_token_id"]], skip_special_tokens=False
            )

            pairs.append(
                {
                    "prompt":   prompt_txt,
                    "chosen":   chosen_txt,     # continuation only!
                    "rejected": rejected_txt,   # continuation only!
                }
            )

        dpo_dataset_hf = Dataset.from_list(pairs)

        # ── DEBUG: inspect a few prompt / chosen / rejected triples ──────────────
        def _show_examples(ds, n=3):
            for i, ex in enumerate(ds.select(range(n))):
                print(f"\n── example {i} ──")
                print("PROMPT:\n",   ex["prompt"])
                print("CHOSEN:\n",   ex["chosen"])
                print("REJECTED:\n", ex["rejected"])
                print("-" * 40)

        _show_examples(dpo_dataset_hf, n=3)

        # ------------------------------------------------------------
        # 3. Apply the *same* length filter & book-keeping as vanilla DPO
        # ------------------------------------------------------------
        def _within_len(example):
            p = tokenizer(example["prompt"],  add_special_tokens=False).input_ids
            c = tokenizer(example["chosen"],  add_special_tokens=False).input_ids
            r = tokenizer(example["rejected"],add_special_tokens=False).input_ids
            return len(p) + len(c) <= max_seq_length and len(p) + len(r) <= max_seq_length

        before = len(dpo_dataset_hf)
        dpo_dataset_hf = dpo_dataset_hf.filter(_within_len)
        after  = len(dpo_dataset_hf)
        logger.info(f"dpo_final_token length filter: kept {after}/{before} examples "
                    f"(max_seq_len = {max_seq_length})")

        if after == 0:
            raise ValueError("every sample exceeded finetune_max_seq_length")

        dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))
        max_train = config.get("finetune_max_train_examples")
        if isinstance(max_train, int) and max_train > 0 and len(dpo_dataset_hf) > max_train:
            dpo_dataset_hf = dpo_dataset_hf.select(range(max_train))
            logger.info(f"Capped training dataset to {max_train} examples.")

    # ─────────────────────────────────────────────────────────────────────
    #  ORPO — single-token pairs (prompt, chosen, rejected)
    #      Mode value:  "orpo_final_token"
    # ─────────────────────────────────────────────────────────────────────
    elif mode == "orpo_final_token":
        # 1) Construct the FTPO dataset exactly as in the ftpo branch
        if config.get("finetune_ftpo_dataset"):
            dataset_path = Path(config["finetune_ftpo_dataset"])
        else:
            ftpo_files = sorted(experiment_run_dir.glob("iter_*_ftpo_pairs.jsonl"))
            if not ftpo_files:
                logger.error("No ftpo files found for orpo_final_token.")
                return None
            dataset_path = ftpo_files[-1]

        ftpo_ds = load_ftpo_multi_dataset(
            dataset_path,
            tokenizer,
            max_seq_len               = max_seq_length,
            rejected_reg_strength     = config.get("ftpo_sample_rejected_regularisation_strength", 0.8),
            chosen_reg_strength       = config.get("ftpo_sample_chosen_regularisation_strength", 0.2),
            min_chosen_tokens         = config.get("ftpo_sample_min_chosen_tokens", 3),
            max_train_examples        = config.get("finetune_max_train_examples"),
        )

        # 2) Convert to (prompt, chosen, rejected) triples — one per row
        pairs   = []
        pad_id  = tokenizer.pad_token_id

        for ex in ftpo_ds:
            prompt_ids = [tid for tid in ex["prompt_ids"] if tid != pad_id]
            prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

            chosen_txt   = tokenizer.decode([ex["chosen_ids"][0]], skip_special_tokens=False)
            rejected_txt = tokenizer.decode([ex["rejected_token_id"]], skip_special_tokens=False)

            pairs.append({"prompt": prompt_txt,
                        "chosen": chosen_txt,
                        "rejected": rejected_txt})

        dpo_dataset_hf = Dataset.from_list(pairs)

        # 3) Length filter / shuffle / cap  (reuse helper)
        def _within_len(ex):
            p = tokenizer(ex["prompt"],   add_special_tokens=False).input_ids
            c = tokenizer(ex["chosen"],   add_special_tokens=False).input_ids
            r = tokenizer(ex["rejected"], add_special_tokens=False).input_ids
            return len(p) + len(c) <= max_seq_length and len(p) + len(r) <= max_seq_length

        before = len(dpo_dataset_hf)
        dpo_dataset_hf = dpo_dataset_hf.filter(_within_len)
        logger.info(f"orpo_final_token length filter: kept {len(dpo_dataset_hf)}/{before} samples")

        dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))
        max_train = config.get("finetune_max_train_examples")
        if isinstance(max_train, int) and max_train > 0 and len(dpo_dataset_hf) > max_train:
            dpo_dataset_hf = dpo_dataset_hf.select(range(max_train))
            logger.info(f"Capped training dataset to {max_train} examples.")


    # ─────────────────────────────────────────────────────────────────────
    #  KTO — single-token, unpaired (prompt, completion, label)
    #      Mode value:  "kto_final_token"
    # ─────────────────────────────────────────────────────────────────────
    elif mode == "kto_final_token":
        # 1) Same FTPO dataset creation
        if config.get("finetune_ftpo_dataset"):
            dataset_path = Path(config["finetune_ftpo_dataset"])
        else:
            ftpo_files = sorted(experiment_run_dir.glob("iter_*_ftpo_pairs.jsonl"))
            if not ftpo_files:
                logger.error("No ftpo files found for kto_final_token.")
                return None
            dataset_path = ftpo_files[-1]

        ftpo_ds = load_ftpo_multi_dataset(
            dataset_path,
            tokenizer,
            max_seq_len               = max_seq_length,
            rejected_reg_strength     = config.get("ftpo_sample_rejected_regularisation_strength", 0.8),
            chosen_reg_strength       = config.get("ftpo_sample_chosen_regularisation_strength", 0.2),
            min_chosen_tokens         = config.get("ftpo_sample_min_chosen_tokens", 3),
            max_train_examples        = config.get("finetune_max_train_examples"),
        )

        # 2) Flatten into (prompt, completion, label) rows
        rows   = []
        pad_id = tokenizer.pad_token_id

        for ex in ftpo_ds:
            prompt_ids = [tid for tid in ex["prompt_ids"] if tid != pad_id]
            prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

            # negative example
            rej_txt = tokenizer.decode([ex["rejected_token_id"]], skip_special_tokens=False)
            rows.append({"prompt": prompt_txt,
                        "completion": rej_txt,
                        "label": False})

            # positive examples
            for cid in ex["chosen_ids"]:
                ch_txt = tokenizer.decode([cid], skip_special_tokens=False)
                rows.append({"prompt": prompt_txt,
                            "completion": ch_txt,
                            "label": True})

        dpo_dataset_hf = Dataset.from_list(rows)

        # 3) Length filter / shuffle / cap
        def _within_len(ex):
            p = tokenizer(ex["prompt"],     add_special_tokens=False).input_ids
            c = tokenizer(ex["completion"], add_special_tokens=False).input_ids
            return len(p) + len(c) <= max_seq_length

        before = len(dpo_dataset_hf)
        dpo_dataset_hf = dpo_dataset_hf.filter(_within_len)
        logger.info(f"kto_final_token length filter: kept {len(dpo_dataset_hf)}/{before} samples")

        #dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))
        max_train = config.get("finetune_max_train_examples")
        if isinstance(max_train, int) and max_train > 0 and len(dpo_dataset_hf) > max_train:
            dpo_dataset_hf = dpo_dataset_hf.select(range(max_train))
            logger.info(f"Capped training dataset to {max_train} examples.")

    else:
        logger.error(f"Unknown finetune_mode '{mode}'. Use 'dpo' or 'ftpo'.")
        return None

    return dpo_dataset_hf