import os
import torch
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Unsloth and related libraries, but make them optional
try:
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unsloth or its dependencies not found ({e}). DPO finetuning will not be available.")
    UNSLOTH_AVAILABLE = False
    # Define dummy classes if needed for type hinting, or just guard calls
    class FastLanguageModel: pass
    class AutoTokenizer: pass
    class DPOTrainer: pass
    class DPOConfig: pass
    def load_dataset(*args, **kwargs): return None
    def get_chat_template(*args, **kwargs): return None


def run_dpo_finetune(config: dict, experiment_run_dir: Path):
    if not UNSLOTH_AVAILABLE:
        logger.error("Cannot run DPO finetuning: Unsloth library is not installed or has missing dependencies.")
        return

    logger.info("Starting DPO finetuning process...")

    # --- Locate the DPO dataset from the current experiment run ---
    dpo_file = experiment_run_dir / "dpo_pairs_dataset.jsonl"
    if not dpo_file.is_file():
        logger.error(f"DPO dataset not found at {dpo_file}. Run the anti-slop pipeline first to generate it.")
        # Fallback: try to find the latest DPO dataset if current one is missing (optional)
        # This logic was in the notebook, but for CLI, it's better to rely on the current run.
        # If you want the "latest overall" logic, it can be re-added here.
        logger.info("If you intended to use a DPO dataset from a different run, please specify its path directly or ensure it's in the current run directory.")
        return

    logger.info(f"Using DPO dataset: {dpo_file}")
    
    try:
        dpo_dataset_hf = load_dataset("json", data_files=str(dpo_file), split="train")
        if not dpo_dataset_hf or len(dpo_dataset_hf) == 0:
            logger.error(f"Loaded DPO dataset from {dpo_file} is empty or failed to load.")
            return
    except Exception as e:
        logger.error(f"Failed to load DPO dataset from {dpo_file}: {e}")
        return

    # Filter malformed rows
    req_cols = {"prompt", "chosen", "rejected"}
    before_len = len(dpo_dataset_hf)
    dpo_dataset_hf = dpo_dataset_hf.filter(lambda x: all(col in x and x[col] for col in req_cols))
    after_len = len(dpo_dataset_hf)
    if after_len == 0:
        logger.error("All rows in DPO dataset were filtered out (missing prompt, chosen, or rejected). Check dataset contents.")
        return
    if after_len < before_len:
        logger.info(f"Filtered out {before_len - after_len} malformed DPO rows; {after_len} remain.")
    logger.info(f"DPO dataset ready with {after_len} samples.")


    # --- Model and Tokenizer Setup ---
    model_name = config['finetune_base_model_id']
    max_seq_length = config['finetune_max_seq_length']
    
    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=config['finetune_load_in_4bit'],
            # token=config.get('vllm_hf_token'), # Use if model is gated
            dtype=torch.bfloat16 if config['finetune_load_in_4bit'] else None # Common with 4bit
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name) #, token=config.get('vllm_hf_token'))
    except Exception as e:
        logger.error(f"Failed to load base model '{model_name}' or tokenizer for DPO: {e}", exc_info=True)
        return

    model = FastLanguageModel.get_peft_model(
        model,
        r=config['finetune_lora_r'],
        lora_alpha=config['finetune_lora_alpha'],
        lora_dropout=config['finetune_lora_dropout'],
        bias="none",
        target_modules=config['finetune_target_modules'],
        use_gradient_checkpointing=config['finetune_gradient_checkpointing'],
        random_state=3407, # Standard seed
        max_seq_length=max_seq_length,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config['finetune_chat_template'],
        # map_eos_token=True, # Usually handled by get_chat_template or model's default
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer.pad_token to tokenizer.eos_token.")


    # --- DPO Trainer Setup ---
    finetune_output_dir = experiment_run_dir / f"finetuned_model{config['finetune_output_dir_suffix']}"
    finetune_output_dir.mkdir(parents=True, exist_ok=True)

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # Unsloth handles this for LoRA
        train_dataset=dpo_dataset_hf,
        tokenizer=tokenizer,
        args=DPOConfig(
            per_device_train_batch_size=config['finetune_batch_size'],
            gradient_accumulation_steps=config['finetune_gradient_accumulation_steps'],
            warmup_ratio=config['finetune_warmup_ratio'],
            num_train_epochs=config['finetune_num_epochs'],
            learning_rate=config['finetune_learning_rate'],
            logging_steps=10,
            optim="adamw_8bit", # Unsloth optimized
            seed=42,
            output_dir=str(finetune_output_dir),
            max_length=max_seq_length,
            max_prompt_length=max_seq_length // 2,
            beta=config['finetune_beta'],
            report_to="none", # "wandb", "tensorboard"
            lr_scheduler_type="linear",
            # bf16=True if config['finetune_load_in_4bit'] else False, # Enable if dtype was bfloat16
            # fp16=False, # Mutually exclusive with bf16
        ),
    )

    logger.info(f"Starting DPO training. Output will be in {finetune_output_dir}")
    try:
        trainer_stats = dpo_trainer.train()
        logger.info("DPO training finished.")
        if hasattr(trainer_stats, 'metrics'):
            logger.info(f"Trainer metrics: {trainer_stats.metrics}")
    except Exception as e:
        logger.error(f"Error during DPO training: {e}", exc_info=True)
        return


    # --- Saving Model ---
    try:
        lora_save_path = finetune_output_dir / "lora_adapters"
        dpo_trainer.save_model(str(lora_save_path)) # Saves LoRA adapters
        tokenizer.save_pretrained(str(lora_save_path))
        logger.info(f"DPO LoRA adapters and tokenizer saved to {lora_save_path}")

        if config.get('finetune_save_merged_16bit'):
            merged_path = finetune_output_dir / "merged_16bit"
            logger.info(f"Saving merged 16-bit model to {merged_path}...")
            model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit", safe_serialization=True)
            logger.info(f"Merged 16-bit DPO model saved to {merged_path}")

        if config.get('finetune_save_gguf_q8_0'):
            gguf_path = finetune_output_dir / "gguf_q8_0" # Unsloth adds .gguf
            logger.info(f"Saving GGUF Q8_0 model to {gguf_path}.gguf ...")
            model.save_pretrained_gguf(str(gguf_path), tokenizer, quantization_method="q8_0")
            logger.info(f"GGUF Q8_0 DPO model saved to {gguf_path}.gguf")

    except Exception as e:
        logger.error(f"Error saving DPO model: {e}", exc_info=True)

    logger.info("DPO finetuning process completed.")