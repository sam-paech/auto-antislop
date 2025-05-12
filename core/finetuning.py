import os
import torch # Keep torch as it might be used for GPU checks earlier if needed
import logging
from pathlib import Path
# typing.Optional can be imported at the top level if used in type hints outside the function
from typing import Optional 

logger = logging.getLogger(__name__)

# Global flag to check if imports were successful, set within the function
UNSLOTH_LIBS_LOADED = False

def run_dpo_finetune(config: dict, experiment_run_dir: Path):
    global UNSLOTH_LIBS_LOADED # To modify the global flag

    # --- Attempt to import Unsloth and related libraries only when this function is called ---
    if not UNSLOTH_LIBS_LOADED:
        try:
            from unsloth import FastLanguageModel
            from transformers import AutoTokenizer, TextStreamer # Added TextStreamer for potential inference example
            from trl import DPOTrainer, DPOConfig
            from datasets import load_dataset
            from unsloth.chat_templates import get_chat_template
            
            # Make these available in the function's local scope for convenience
            # Or, you can just use them directly as `unsloth.FastLanguageModel` etc.
            # For cleaner code within the function, assigning to local variables is fine.
            globals()['FastLanguageModel'] = FastLanguageModel
            globals()['AutoTokenizer'] = AutoTokenizer
            globals()['TextStreamer'] = TextStreamer
            globals()['DPOTrainer'] = DPOTrainer
            globals()['DPOConfig'] = DPOConfig
            globals()['load_dataset'] = load_dataset
            globals()['get_chat_template'] = get_chat_template
            
            UNSLOTH_LIBS_LOADED = True
            logger.info("Unsloth and DPO finetuning libraries loaded successfully.")
        except ImportError as e:
            logger.error(f"Failed to import Unsloth or its dependencies: {e}. DPO finetuning cannot proceed.")
            logger.error("Please ensure Unsloth, TRL, PEFT, Accelerate, BitsandBytes, Transformers, and Datasets are installed.")
            return # Exit if essential libraries can't be loaded

    if not UNSLOTH_LIBS_LOADED: # Double check, in case of other import issues
        logger.error("Unsloth libraries are not available. Aborting DPO finetuning.")
        return

    logger.info("Starting DPO finetuning process...")

    # --- Locate the DPO dataset from the current experiment run ---
    dpo_file = experiment_run_dir / "dpo_pairs_dataset.jsonl"
    if not dpo_file.is_file():
        logger.error(f"DPO dataset not found at {dpo_file}. Run the anti-slop pipeline first to generate it.")
        return

    logger.info(f"Using DPO dataset: {dpo_file}")
    
    try:
        # Use the globally available load_dataset (or from local scope if preferred)
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
            dtype=torch.bfloat16 if config['finetune_load_in_4bit'] and torch.cuda.is_bf16_supported() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config['finetune_chat_template'],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer.pad_token to tokenizer.eos_token.")


    # --- DPO Trainer Setup ---
    finetune_output_dir = experiment_run_dir / f"finetuned_model{config['finetune_output_dir_suffix']}"
    finetune_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine bf16/fp16 flags based on config and capabilities
    use_bf16 = False
    use_fp16 = False
    if config['finetune_load_in_4bit']: # Often implies bfloat16 if supported
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_bf16 = True
            logger.info("Using bfloat16 for DPO training (4-bit model and bfloat16 supported).")
        # else: fp16 might be an option if not 4-bit, but 4-bit usually goes with bf16 or no explicit fp16/bf16
    # else if not 4-bit, user could specify fp16 in config if desired.
    # For simplicity, this example prioritizes bf16 with 4-bit.

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=dpo_dataset_hf,
        tokenizer=tokenizer,
        args=DPOConfig(
            per_device_train_batch_size=config['finetune_batch_size'],
            gradient_accumulation_steps=config['finetune_gradient_accumulation_steps'],
            warmup_ratio=config['finetune_warmup_ratio'],
            num_train_epochs=config['finetune_num_epochs'],
            learning_rate=config['finetune_learning_rate'],
            logging_steps=10,
            optim="adamw_8bit",
            seed=42,
            output_dir=str(finetune_output_dir),
            max_length=max_seq_length,
            max_prompt_length=max_seq_length // 2,
            beta=config['finetune_beta'],
            report_to="tensorboard", # Changed to tensorboard for local runs
            lr_scheduler_type="linear",
            bf16=use_bf16,
            fp16=use_fp16, # Ensure only one is true or both false
        ),
    )

    logger.info(f"Starting DPO training. Output will be in {finetune_output_dir}. Check tensorboard for progress.")
    try:
        trainer_stats = dpo_trainer.train()
        logger.info("DPO training finished.")
        if hasattr(trainer_stats, 'metrics'):
            logger.info(f"Trainer metrics: {trainer_stats.metrics}")
    except Exception as e:
        logger.error(f"Error during DPO training: {e}", exc_info=True)
        return

    # --- Saving Model ---
    # (Saving logic remains the same)
    try:
        lora_save_path = finetune_output_dir / "lora_adapters"
        dpo_trainer.save_model(str(lora_save_path)) 
        tokenizer.save_pretrained(str(lora_save_path))
        logger.info(f"DPO LoRA adapters and tokenizer saved to {lora_save_path}")

        if config.get('finetune_save_merged_16bit'):
            merged_path = finetune_output_dir / "merged_16bit"
            logger.info(f"Saving merged 16-bit model to {merged_path}...")
            model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit", safe_serialization=True)
            logger.info(f"Merged 16-bit DPO model saved to {merged_path}")

        if config.get('finetune_save_gguf_q8_0'):
            gguf_path = finetune_output_dir / "gguf_q8_0" 
            logger.info(f"Saving GGUF Q8_0 model to {gguf_path}.gguf ...")
            model.save_pretrained_gguf(str(gguf_path), tokenizer, quantization_method="q8_0")
            logger.info(f"GGUF Q8_0 DPO model saved to {gguf_path}.gguf")

    except Exception as e:
        logger.error(f"Error saving DPO model: {e}", exc_info=True)

    logger.info("DPO finetuning process completed.")