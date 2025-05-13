import logging
logger = logging.getLogger(__name__)

# --- Attempt to import Unsloth and related libraries only when this function is called ---
#if not UNSLOTH_LIBS_LOADED:
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
    #return # Exit if essential libraries can't be loaded

#if not UNSLOTH_LIBS_LOADED: # Double check, in case of other import issues
#    logger.error("Unsloth libraries are not available. Aborting DPO finetuning.")
#    return


import os
import torch # Keep torch as it might be used for GPU checks earlier if needed
from pathlib import Path
# typing.Optional can be imported at the top level if used in type hints outside the function
from typing import Optional 
from trl import DPOTrainer
from datasets import load_dataset
from torch.utils.data import default_collate
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ── QUIET-MODE FOR DATASETS / TRANSFORMERS ────────────────────────────
import datasets, transformers, warnings, contextlib, io, os
# kill progress bars & debug prints
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# route any stray `print` that slips through to /dev/null during finetune
null_fh = open(os.devnull, "w")
suppress_stdout = contextlib.redirect_stdout(null_fh)
suppress_stderr = contextlib.redirect_stderr(null_fh)
# ───────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  1. Quiet 🤗 Datasets (the D2 / T4 object-dumps)
# ─────────────────────────────────────────────────────────────
try:
    # Must be done **before** the first `datasets` import
    import datasets.utils.logging as hf_datasets_logging
    hf_datasets_logging.set_verbosity_error()  # or WARNING if you still want HF progress bars
except ModuleNotFoundError:
    pass  # datasets not installed yet – fine

# Belt-and-braces: silence its individual loggers too
for name in (
    "datasets",               # umbrella
    "datasets.arrow_dataset", # the shard concatenation prints
):
    l = logging.getLogger(name)
    l.setLevel(logging.ERROR)
    l.propagate = False

# ─────────────────────────────────────────────────────────────
#  2. Silence *all* remaining torch.compile / dynamo spam
# ─────────────────────────────────────────────────────────────
_noisy_torch_loggers = [
    # earlier ones
    "torch._functorch",
    "torch._functorch._aot_autograd",
    "torch._functorch._aot_autograd.jit_compile_runtime_wrappers",
    "torch._inductor",
    "torch._dynamo",

    # new offenders
    "torch._functorch._aot_autograd.dispatch_and_compile_graph",
    "torch.fx",
    "torch.fx.experimental",
    "torch.fx.experimental.symbolic_shapes",
    "torch._utils_internal",
]

for name in _noisy_torch_loggers:
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False  # critical – stops bubbling up to the root logger

# Global flag to check if imports were successful, set within the function
UNSLOTH_LIBS_LOADED = False


class LastTokenDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_unused_columns = False          # keep custom cols
        # tell Trainer not to replace our collator
        self.data_collator = self.tdpo_collator

    # ------------------------------------------------------------
    @staticmethod
    def tdpo_collator(features):
        """
        Convert a list of dicts with keys
          prompt_ids, attention_mask, chosen_token_id, rejected_token_id
        into a batch of PyTorch tensors.
        """
        batch = {}
        for k in ("prompt_ids", "attention_mask",
                  "chosen_token_id", "rejected_token_id"):
            batch[k] = default_collate([f[k] for f in features])
        return batch
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        # -- Input Tensors --------------------------------------------------
        # inputs["prompt_ids"] is a list of lists/tensors from tdpo_collator
        # We need to pad them here to form a batch.
        prompt_ids_list = [torch.tensor(x, device=model.device) for x in inputs["prompt_ids"]]
        
        # Pad the sequences to the longest sequence in the batch
        # The padding value should be the tokenizer's pad_token_id
        # model.config.pad_token_id might be None if not set, tokenizer.pad_token_id is safer
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # Fallback if tokenizer.pad_token is not set, though it should be.
            pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0 
            logger.warning(f"Using pad_token_id: {pad_token_id} as tokenizer.pad_token_id was None.")

        padded_prompt_ids = pad_sequence(
            prompt_ids_list,
            batch_first=True,
            padding_value=pad_token_id
        )

        # Create a 2D attention mask based on the padding
        # This is the standard mask format Hugging Face models expect.
        # 1 for tokens to attend to, 0 for padding tokens.
        attention_mask = (padded_prompt_ids != pad_token_id).long() # .to(model.device) already done by tensor creation

        chosen_token_ids = torch.tensor(inputs["chosen_token_id"], device=model.device)
        rejected_token_ids = torch.tensor(inputs["rejected_token_id"], device=model.device)

        # -- Policy Model Forward Pass --------------------------------------
        # Model expects: input_ids, attention_mask (2D)
        # Unsloth / FlashAttention will handle causal masking internally if attention_mask is 2D
        # or if attention_mask=None (but providing the 2D padding mask is safer and standard)
        
        # Make sure gradient checkpointing is handled correctly by Unsloth
        # The extensive disabling in run_dpo_finetune should cover this.
        # If still issues, the temp_disable_ckpt context manager could be used here.
        # with temp_disable_ckpt(model) if mode == "tdpo" else contextlib.nullcontext():
        
        policy_outputs = model(
            input_ids=padded_prompt_ids,
            attention_mask=attention_mask,
            use_cache=False, # Important for training
            output_hidden_states=False # Not needed for logits
        )
        # Get logits for the *next* token prediction, i.e., at the last position of the prompt
        # For each sequence in the batch, find its actual length (before padding)
        # to get the logits at the correct position.
        # sequence_lengths = attention_mask.sum(dim=1) # Number of non-padded tokens
        # policy_logits_last_token = policy_outputs.logits[torch.arange(padded_prompt_ids.size(0)), sequence_lengths - 1, :]
        
        # Simpler: if all prompts are processed up to their actual end,
        # the logits for the *next* token are at index -1 of the *output* sequence.
        # This assumes the model's output sequence length matches input.
        # For causal LMs, logits[b, t, :] are for predicting token t+1 given tokens 0...t.
        # So, for a prompt of length L_prompt (non-padded), we need logits at index L_prompt-1.
        
        # The `DPOTrainer` and standard HF practice for getting chosen/rejected logps
        # often involves passing the full chosen/rejected sequences.
        # Here, we only care about the single next token.
        # `policy_outputs.logits` is [B, L, V]
        # We need the logits at the position *before* the chosen/rejected token would be.
        
        # Get the indices of the last actual token in each prompt
        # (Batch_size, Prompt_length_in_batch)
        # Example: [[1,2,3,PAD,PAD], [4,5,PAD,PAD,PAD]] -> lengths = [3,2] -> indices = [2,1]
        prompt_lengths = torch.sum(attention_mask, dim=1)
        last_token_indices = prompt_lengths - 1

        # Gather the logits from the policy model at the end of each prompt
        # logits are [Batch, SeqLen, VocabSize]
        # We need to select [Batch, VocabSize] using last_token_indices
        policy_logits_at_last_token = policy_outputs.logits[torch.arange(padded_prompt_ids.size(0), device=model.device), last_token_indices, :] # [B, V]

        policy_logps_chosen = F.log_softmax(policy_logits_at_last_token, dim=-1).gather(
            dim=-1, index=chosen_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        policy_logps_rejected = F.log_softmax(policy_logits_at_last_token, dim=-1).gather(
            dim=-1, index=rejected_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        # -- Reference Model Forward Pass -----------------------------------
        if self.ref_model is not None:
            with torch.no_grad():
                self.ref_model.eval() # Ensure ref model is in eval mode
                ref_outputs = self.ref_model(
                    input_ids=padded_prompt_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False
                )
                # ref_logits_last_token = ref_outputs.logits[torch.arange(padded_prompt_ids.size(0)), sequence_lengths - 1, :]
                ref_logits_at_last_token = ref_outputs.logits[torch.arange(padded_prompt_ids.size(0), device=model.device), last_token_indices, :] # [B, V]


                ref_logps_chosen = F.log_softmax(ref_logits_at_last_token, dim=-1).gather(
                    dim=-1, index=chosen_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                ref_logps_rejected = F.log_softmax(ref_logits_at_last_token, dim=-1).gather(
                    dim=-1, index=rejected_token_ids.unsqueeze(-1)
                ).squeeze(-1)
        else:
            # This case should ideally not happen if DPO trainer is initialized correctly
            # or handle it by setting ref_logps to zero (as in original DPO paper if no ref model)
            # For simplicity here, let's assume ref_model is always present.
            # If you are using the same model weights for policy and ref (before fine-tuning)
            # and only applying LoRA to policy, then ref_logps are effectively the initial state.
            # The DPOTrainer handles ref_model creation if None is passed.
            # For TDPO, the logic is log pi(t+) - log rho(t+) - (log pi(t-) - log rho(t-))
            # If ref_model is None, DPOTrainer usually uses initial weights of the model.
            # The `self.label_smoothed_nll_loss` in original DPOTrainer handles this.
            # Here, we need explicit ref_logps.
            # If you truly have no separate ref_model and want to use the initial policy model's
            # logprobs (before this batch's update), that's more complex (store initial logprobs).
            # The standard DPO setup implies self.ref_model is a frozen copy.
            logger.warning("ref_model is None. Reference log probabilities will not be used, which deviates from DPO.")
            # This will make the loss: -log_sigmoid(beta * (policy_logps_chosen - policy_logps_rejected))
            # which is SLiC loss if beta=1.
            # To adhere to DPO, ensure ref_model is properly configured.
            # For now, to prevent crash, let's make them zero, but this is NOT DPO.
            ref_logps_chosen = torch.zeros_like(policy_logps_chosen)
            ref_logps_rejected = torch.zeros_like(policy_logps_rejected)


        # -- DPO Loss Calculation -------------------------------------------
        # log π(t⁺|prompt) – log ρ(t⁺|prompt)
        pi_logratios_chosen = policy_logps_chosen - ref_logps_chosen
        # log π(t⁻|prompt) – log ρ(t⁻|prompt)
        pi_logratios_rejected = policy_logps_rejected - ref_logps_rejected

        logits = pi_logratios_chosen - pi_logratios_rejected # This is the term inside σ: β * ( ... )
        
        # The DPO loss is -log σ(β * logits)
        # Equivalent to F.binary_cross_entropy_with_logits with targets = 1
        # loss = -F.logsigmoid(self.beta * logits).mean()
        # Or, if using the DPO trainer's built-in loss computation style:
        # (taken from TRL's DPOTrainer `dpo_loss` method)
        loss = -F.logsigmoid(self.beta * logits).mean() # If all pairs are y_w=1, y_l=0
        # For more general preference modeling (e.g. with margins or different weighting):
        # loss = -F.logsigmoid(self.beta * logits).mean() # Common case
        # loss = - (F.logsigmoid(self.beta * logits) * (1-self.label_smoothing) + F.logsigmoid(-self.beta*logits)*self.label_smoothing).mean() # with label smoothing

        # -- Metrics --------------------------------------------------------
        chosen_rewards = self.beta * pi_logratios_chosen.detach()
        rejected_rewards = self.beta * pi_logratios_rejected.detach()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            "loss": loss.item(), # Keep .item() for logging scalar values
            "accuracy": accuracy.item(),
            "margin": (chosen_rewards - rejected_rewards).mean().item(),
            "rewards_chosen_mean": chosen_rewards.mean().item(),
            "rewards_rejected_mean": rejected_rewards.mean().item(),
        }
        # Add chosen_wins from your original code if you prefer that name
        metrics["chosen_wins"] = accuracy.item() 

        if return_outputs:
            return (loss, metrics) # TRL expects a dict for the second element usually
        return loss
    
    def _prepare_dataset(
        self,
        dataset,
        processing_class=None,
        args=None,
        split="train",
    ):
        """
        Bypass DPOTrainer's default preprocessing, which expects
        'prompt'/'chosen'/'rejected' columns.  Our TDPO dataset is already
        tokenised and has columns:
          prompt_ids, attention_mask, chosen_token_id, rejected_token_id
        So we simply return it unchanged.
        """
        return dataset


def load_tdpo_dataset(path: Path, tokenizer):
    ds = load_dataset("json", data_files=str(path), split="train")

    def _tok(ex):
        enc          = tokenizer(
            ex["context_with_chat_template"],
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
            padding="max_length"   # or "longest" if you prefer
        )
        return {
            "prompt_ids":        enc.input_ids,
            "attention_mask":    enc.attention_mask,
            "chosen_token_id":   tokenizer(ex["chosen_decoded"],   add_special_tokens=False).input_ids[0],
            "rejected_token_id": tokenizer(ex["rejected_decoded"], add_special_tokens=False).input_ids[0],
        }

    return ds.map(_tok, remove_columns=ds.column_names)



def run_dpo_finetune(config: dict, experiment_run_dir: Path):
    #global UNSLOTH_LIBS_LOADED # To modify the global flag

    

    logger.info("Starting DPO finetuning process...")

    model_name = config['finetune_base_model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if config['finetune_chat_template']:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=config['finetune_chat_template'],
        )
    if tokenizer.pad_token is None:
        # this may not always be desired. adjust to the model you are finetuning.
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer.pad_token to tokenizer.eos_token.")

    
    # ── Select dataset path according to finetune_mode ───────────────────
    mode = config.get("finetune_mode", "dpo").lower()   # expect "dpo" or "tdpo"
    dataset_path = None

    if mode == "dpo":
        # full-sequence preference pairs
        dataset_path = experiment_run_dir / "dpo_pairs_dataset.jsonl"
        if not dataset_path.is_file():
            logger.error(f"DPO dataset not found at {dataset_path}")
            return

        dpo_dataset_hf = load_dataset(
            "json",
            data_files=str(dataset_path),
            split="train"
        )

        # ── filter malformed rows (prompt / chosen / rejected missing) ──
        req_cols = {"prompt", "chosen", "rejected"}
        before_len = len(dpo_dataset_hf)
        dpo_dataset_hf = dpo_dataset_hf.filter(
            lambda x: all(col in x and x[col] for col in req_cols)
        )
        after_len = len(dpo_dataset_hf)
        if after_len == 0:
            logger.error("All rows in DPO dataset were filtered out. Check contents.")
            return
        if after_len < before_len:
            logger.info(f"Filtered out {before_len - after_len} malformed rows; "
                        f"{after_len} remain.")
        logger.info(f"DPO dataset ready with {after_len} samples.")

    elif mode == "tdpo":
        # single-token preference pairs
        if config.get("finetune_tdpo_dataset"):
            dataset_path = Path(config["finetune_tdpo_dataset"])
        else:
            tdpo_files = sorted(experiment_run_dir.glob("iter_*_tdpo_pairs.jsonl"))
            if not tdpo_files:
                logger.error("No iter_*_tdpo_pairs.jsonl files found for TDPO.")
                return
            dataset_path = tdpo_files[-1]          # most recent iteration

        if not dataset_path.is_file():
            logger.error(f"TDPO dataset not found at {dataset_path}")
            return

        # defer actual loading until tokenizer is ready
        tdpo_dataset_path = dataset_path
        dpo_dataset_hf = load_tdpo_dataset(tdpo_dataset_path, tokenizer)

    else:
        logger.error(f"Unknown finetune_mode '{mode}'. Use 'dpo' or 'tdpo'.")
        return



    # --- Model and Tokenizer Setup ---    
    max_seq_length = config['finetune_max_seq_length']
    
    
    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=config['finetune_load_in_4bit'],
            dtype=torch.bfloat16 if config['finetune_load_in_4bit'] and torch.cuda.is_bf16_supported() else None,
        )
        
    except Exception as e:
        logger.error(f"Failed to load base model '{model_name}' or tokenizer for DPO: {e}", exc_info=True)
        return
    
    # Hard-disable gradient-checkpointing for TDPO
    if mode == "tdpo":
        model.config._attn_implementation = "flash_attention_2"
        # turn off every ckpt flag Unsloth uses
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            print('gradient checkpointing disabled!')
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_disable()
            print("✓ Disabled gradient checkpointing for better compilation")
        for mod in model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                print('module disabled gradient checkpointing!')
                mod.gradient_checkpointing = False
        if hasattr(model.config, "gradient_checkpointing"):
            print('attempting to disable gradient checkpionting in config')
            model.config.gradient_checkpointing = False
        #model = model.to(model.device)
        if False:
            # 1. HF flag
            if getattr(model, "gradient_checkpointing", False):
                model.gradient_checkpointing_disable()           # HF helper
            # 2. Unsloth compiled blocks keep their own flag
            if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
                model.model.gradient_checkpointing = False
            if hasattr(model.config, "gradient_checkpointing"):
                model.config.gradient_checkpointing = False

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

    if mode == "tdpo":
        model.config._attn_implementation = "flash_attention_2"
        # turn off every ckpt flag Unsloth uses
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        for mod in model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = False
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = False

    


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

    TrainerClass = LastTokenDPOTrainer if mode == "tdpo" else DPOTrainer

    dpo_trainer = TrainerClass(
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
            remove_unused_columns=False,
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