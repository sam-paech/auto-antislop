from typing import TYPE_CHECKING

# placeholders for later lazy loading
if TYPE_CHECKING:
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    from transformers import TextStreamer
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from trl import (
        DPOTrainer, DPOConfig,
        KTOTrainer, KTOConfig,
        ORPOTrainer, ORPOConfig,
    )
    from datasets import Dataset
    from unsloth.chat_templates import get_chat_template
    import torch # Keep torch as it might be used for GPU checks earlier if needed    
    # typing.Optional can be imported at the top level if used in type hints outside the function
    from torch.utils.data import default_collate
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence    
    

import logging
from pathlib import Path
import os
import math
import json
from utils.dataset_helpers import load_ftpo_multi_dataset
from utils.model_helpers import fix_gemma3_checkpoint, detie_lm_head, prepare_gemma3_for_save
# Import the new dataloader function
from utils.trainer_dataloaders import load_and_prepare_dataset
logger = logging.getLogger(__name__)

def load_imports(use_unsloth):
    # --- Attempt to import Unsloth and related libraries only when this function is called ---
    try:
        if use_unsloth:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
            globals()['get_chat_template'] = get_chat_template
            globals()['FastLanguageModel'] = FastLanguageModel

        from transformers import AutoTokenizer, TextStreamer # Added TextStreamer for potential inference example
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        from trl import (
            DPOTrainer, DPOConfig,
            KTOTrainer, KTOConfig,
            ORPOTrainer, ORPOConfig,
        )
        from datasets import load_dataset        
        import torch
        from torch.utils.data import default_collate
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence
        import os
        import transformers
        



        # Make all imports available in the global scope        
        globals()['AutoTokenizer'] = AutoTokenizer
        globals()['TextStreamer'] = TextStreamer
        globals()['AutoModelForCausalLM'] = AutoModelForCausalLM
        globals()['PeftModel'] = PeftModel
        globals()['DPOTrainer'] = DPOTrainer
        globals()['DPOConfig'] = DPOConfig
        globals()['KTOTrainer']  = KTOTrainer
        globals()['KTOConfig']   = KTOConfig
        globals()['ORPOTrainer'] = ORPOTrainer
        globals()['ORPOConfig']  = ORPOConfig
        globals()['load_dataset'] = load_dataset        
        globals()['torch'] = torch
        globals()['default_collate'] = default_collate
        globals()['F'] = F
        globals()['pad_sequence'] = pad_sequence
        
        logger.info("Unsloth and DPO finetuning libraries loaded successfully.")
    except ImportError as e:
        logger.error(f"Failed to import Unsloth or its dependencies: {e}. DPO finetuning cannot proceed.")
        logger.error("Please ensure Unsloth, TRL, PEFT, Accelerate, BitsandBytes, Transformers, and Datasets are installed.")
        #return # Exit if essential libraries can't be loaded


    
    


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







def freeze_early_layers(model, n_unfrozen: int = 4, verbose: bool = True):
    # ── unwrap PEFT wrappers ──────────────────────────────────────────
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()

    # extra candidate paths for newer HF models
    candidate_paths = [
        "model.layers",
        "model.decoder.layers",
        "model.transformer.layers",   # Gemma-3, Mixtral, etc.
        "layers",
    ]

    block_list = None
    for path in candidate_paths:
        obj = model
        for name in path.split("."):
            if not hasattr(obj, name):
                obj = None
                break
            obj = getattr(obj, name)
        if isinstance(obj, (list, torch.nn.ModuleList)):
            block_list = obj
            break

    # fall-back: scan for the first ModuleList that looks like transformer blocks
    if block_list is None:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) and hasattr(mod[0], "self_attn"):
                block_list = mod
                break

    if block_list is None:
        raise RuntimeError("Could not locate transformer layers list")

    total = len(block_list)
    cut   = total - n_unfrozen
    if verbose:
        print(f"Freezing layers 0 … {cut-1} of {total} (keeping {n_unfrozen}).")

    for i, blk in enumerate(block_list):
        if i < cut:
            blk.requires_grad_(False)




def run_dpo_finetune(config: dict, experiment_run_dir: Path):
    use_unsloth = config.get("finetune_use_unsloth", False)

    # honour per-stage GPU mask – must precede any torch import
    gpu_mask = config.get("finetune_cuda_visible_devices")
    if gpu_mask:
        import os, logging
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_mask)
        logging.getLogger(__name__).info(
            f"Finetune stage limited to CUDA_VISIBLE_DEVICES={gpu_mask}"
        )

    load_imports(use_unsloth)

    logger.info("Starting finetuning process...")

    from core.ftpo_trainer import FTPOTrainer, ThresholdStop
    


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

    
    # ── Load and prepare dataset using the dedicated utility function ───
    dpo_dataset_hf = load_and_prepare_dataset(
        config=config,
        experiment_run_dir=experiment_run_dir,
        tokenizer=tokenizer
    )

    if dpo_dataset_hf is None:
        logger.error("Dataset loading and preparation failed. Aborting finetune.")
        return
    
        

    import torch
    
    if use_unsloth:
        try:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=config['finetune_max_seq_length'],
                load_in_4bit=config['finetune_load_in_4bit'],
                dtype=torch.bfloat16 if config['finetune_load_in_4bit'] and torch.cuda.is_bf16_supported() else None,
            )
   
        except Exception as e:
            logger.error(f"Failed to load base model '{model_name}' or tokenizer for DPO: {e}", exc_info=True)
            return
    else:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        import torch

        base_model_id = config["finetune_base_model_id"]
        max_len       = config["finetune_max_seq_length"]

        # 1. base model --------------------------------------------------
        if config["finetune_load_in_4bit"]:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit          = True,
                bnb_4bit_quant_type   = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype    = torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config = bnb_cfg,
                device_map          = {"": 0},
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype = torch.bfloat16,             # Qwen-3 was trained in bf16
                device_map  = {"": 0},
                trust_remote_code=True,
            )

        #model.config._attn_implementation = "eager"        # avoid flash-attn fp16 path
        model.config._attn_implementation = "flash_attention_2"
        model.train()

        # 2. LoRA --------------------------------------------------------
        lora_cfg = LoraConfig(
            r                = config["finetune_lora_r"],
            lora_alpha       = config["finetune_lora_alpha"],
            lora_dropout     = config["finetune_lora_dropout"],
            bias             = "none",
            target_modules   = config["finetune_target_modules"],
        )
        model = get_peft_model(model, lora_cfg)        
        
        

    if use_unsloth:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config['finetune_lora_r'],
            lora_alpha=config['finetune_lora_alpha'],
            lora_dropout=config['finetune_lora_dropout'],
            bias="none",
            target_modules=config['finetune_target_modules'],
            use_gradient_checkpointing=config['finetune_gradient_checkpointing'],
            random_state=3407,
            max_seq_length=config['finetune_max_seq_length'],
        )



    print("⇢  trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad))


    CALC_VAL_STATS = False
    N_VAL_ITEMS = 50
    if CALC_VAL_STATS:
        pad_id = tokenizer.pad_token_id
        def _collate_ftpo(features, pad_id: int, max_len: int):
            """
            Validation-time collator – left-pads so the final real token is
            always at position -1, matching the training collator.
            """
            B = len(features)

            # ── build [B, max_len] prompt tensor ─────────────────────────
            prompt_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
            attn_mask  = torch.zeros_like(prompt_ids, dtype=torch.bool)

            for i, f in enumerate(features):
                seq = torch.tensor(f["prompt_ids"], dtype=torch.long)
                if seq.size(0) > max_len:          # truncate if over-long
                    seq = seq[-max_len:]
                prompt_ids[i, -seq.size(0):] = seq    # left-pad
                attn_mask [i, -seq.size(0):] = True

            # ── ftpo path ──────────────────────────
            max_c = max(len(f["chosen_ids"]) for f in features)
            chosen_pad  = torch.full((B, max_c), -100, dtype=torch.long)
            chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
            for i, f in enumerate(features):
                ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
                chosen_pad [i, :ids.size(0)] = ids
                chosen_mask[i, :ids.size(0)] = True
            batch = dict(chosen_ids=chosen_pad, chosen_mask=chosen_mask)


            # ── always include rejected & prompt tensors ─────────────────
            batch.update(
                prompt_ids        = prompt_ids,
                attention_mask    = attn_mask,
                rejected_token_id=torch.tensor([f["rejected_token_id" if "rejected_token_id" in f else "rejected_id"] for f in features]),
            )
            return batch



        
        def _gap_stats(model, dataset, collate_fn, tag,
               batch_size=2, device=None):
            from torch.utils.data import DataLoader

            model.eval()
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)
            device = device or next(model.parameters()).device

            rows, tot_delta, wins = [], 0.0, 0
            with torch.no_grad():
                for batch in loader:
                    ids   = batch["prompt_ids"].to(device)
                    attn  = batch["attention_mask"].to(device)
                    last  = attn.sum(1) - 1
                    rej   = batch["rejected_token_id"].to(device)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        #logits = model(ids, attention_mask=attn).logits
                        logits_next = model(ids, attention_mask=attn).logits[:, -1, :]
                        logp_all    = torch.log_softmax(logits_next, -1)


                    ch_ids  = batch["chosen_ids"].to(device)          # [B, C]
                    ch_mask = batch["chosen_mask"].to(device)         # [B, C]  bool

                    # log-prob of every chosen token
                    safe_ch_ids = ch_ids.clone()
                    safe_ch_ids[~ch_mask] = pad_id        # any valid id is fine
                    lp_chosen   = logp_all.gather(-1, safe_ch_ids)


                    # log-prob of the rejected token (same for all C columns)
                    lp_bad = logp_all.gather(-1, rej.unsqueeze(-1)).squeeze(-1)  # [B]

                    # per-token wins (True if chosen beats rejected)
                    wins_tok = (lp_chosen > lp_bad.unsqueeze(-1)) & ch_mask      # [B, C] bool

                    # fraction of winners for each training example
                    frac_win = wins_tok.float().sum(-1) / ch_mask.sum(-1)        # [B]

                    delta = frac_win                        # use fraction as the “margin”

                    tot_delta += delta.sum().item()     # for mean_delta
                    wins      += delta.sum().item()     # for chosen_win

                    # record first chosen id for debugging
                    first_ch = (
                        batch["chosen_ids"][:,0] if "chosen_ids" in batch
                        else batch["chosen_token_id"]
                    )
                    rows.extend(
                        {"delta": round(d.item(), 6),
                        "chosen_id": int(c.item()),
                        "rejected_id": int(r.item())}
                        for d, c, r in zip(delta, first_ch, rej)
                    )

            mean = tot_delta / len(dataset)
            acc  = wins / len(dataset)
            return rows, {"tag": tag,
                        "mean_delta": mean,
                        "chosen_win": acc,
                        "n": len(dataset)}


        # ────────────────────────────────────────────────────────────────────
        # 1) train / validation split  (after max-train cap, before model load)
        # ────────────────────────────────────────────────────────────────────
        VAL_N    = min(1000, int(0.1 * len(dpo_dataset_hf)))
        train_ds = dpo_dataset_hf.select(range(len(dpo_dataset_hf) - VAL_N))
        val_ds   = dpo_dataset_hf.select(range(len(dpo_dataset_hf) - VAL_N, len(dpo_dataset_hf)))

        logger.info(f"Split → train {len(train_ds)}  | val {len(val_ds)}")

        # Save a copy for the trainer
        dpo_dataset_hf = train_ds
        # (val_ds is only for analysis; we’re not doing eval during training.)
        # ────────────────────────────────────────────────────────────────────
        # 2) PRE-TRAIN statistics
        # ────────────────────────────────────────────────────────────────────
        analysis_dir = experiment_run_dir / "logprob_gap_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        collate = lambda feats: _collate_ftpo(feats, pad_id, config['finetune_max_seq_length'])


        if True: # skip this check for now
            pre_train_rows, pre_train_stats = _gap_stats(model, train_ds.select(range(min(N_VAL_ITEMS, len(train_ds)))), collate, "train_pre")
            pre_val_rows , pre_val_stats   = _gap_stats(model, val_ds.select(range(min(N_VAL_ITEMS, len(val_ds)))),   collate, "val_pre")


            with open(analysis_dir / "train_pre.jsonl", "w") as f:
                for r in pre_train_rows: f.write(json.dumps(r) + "\n")
            with open(analysis_dir / "val_pre.jsonl", "w") as f:
                for r in pre_val_rows:  f.write(json.dumps(r) + "\n")

            print("\n— PRE-TRAIN SUMMARY —")
            print(pre_train_stats)
            print(pre_val_stats)
            #print("sample train rows:", pre_train_rows[:10])
            #print("sample val rows  :", pre_val_rows [:10])


    import gc
    #gc.collect()
    #torch.cuda.empty_cache()
    #torch.cuda.reset_peak_memory_stats()


    if config.get("finetune_freeze_early_layers", False):
        n_unfrozen = config.get("finetune_n_layers_unfrozen", 10)
        freeze_early_layers(model, n_unfrozen = n_unfrozen, verbose = True)


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


    # --- derive LR automatically --------------------------------------------    

    # ── learning‑rate schedule parameters ───────────────────────────────────
    N_SWITCH = 5_000          # centre of transition (examples)
    WIDTH    = 1_000          # half‑width of tanh ramp (examples)
    LR_BASE_SCALE_CONST = 0.15
    LR_USER_SCALE_CONST = config.get("finetune_auto_learning_rate_adjustment_scaling", 1.0)
    eta0     = 1e-3           # base LR before scaling

    # ── auto‑scale LR ───────────────────────────────────────────────────────
    if config.get("finetune_auto_learning_rate", False):
        N      = len(dpo_dataset_hf)
        B_eff  = config["finetune_batch_size"] * config["finetune_gradient_accumulation_steps"]
        rank   = config["finetune_lora_r"]

        # common pre‑factor
        base = (
            eta0
            * (B_eff / 256) ** 0.5
            * (rank  /   8) ** 0.5
            * LR_BASE_SCALE_CONST
            * LR_USER_SCALE_CONST
        )

        # square‑root branch (α = 0.5)
        lr_small = base * (1e4 / N) ** 0.5

        # linear tail anchored to value at N_SWITCH (α = 1.0 beyond switch)
        anchor   = (1e4 / N_SWITCH) ** 0.5
        lr_large = base * anchor * (N_SWITCH / N)

        # smooth blend with tanh ramp
        w  = 0.5 * (1 + math.tanh((N - N_SWITCH) / WIDTH))  # 0 → 1 over ~2*WIDTH
        lr = (1 - w) * lr_small + w * lr_large

        config["finetune_learning_rate"] = lr
        print(f"Auto‑scaled LR (N={N}, w={w:.3f}) = {lr:.3e}")

    trainer_lookup = {
        "ftpo":           FTPOTrainer,
        "dpo":            DPOTrainer,
        "dpo_final_token":DPOTrainer,
        "orpo_final_token":ORPOTrainer,
        "kto_final_token":KTOTrainer,
    }
    TrainerClass = trainer_lookup[config.get("finetune_mode", "ftpo").lower()]

    config_lookup = {
        DPOTrainer:  DPOConfig,
        FTPOTrainer: DPOConfig,   # FTPO subclasses DPO → same config
        ORPOTrainer: ORPOConfig,
        KTOTrainer:  KTOConfig,
    }
    ConfigClass = config_lookup[TrainerClass]

    if use_unsloth:
        optimiser_str = "adamw_8bit"
    else:        
        optimiser_str = "paged_adamw_32bit"

    # pass the right args to the dpo trainer depending on what it expects
    import inspect
    init_params = inspect.signature(DPOTrainer.__init__).parameters
    kw = {"processing_class" if "processing_class" in init_params else "tokenizer": tokenizer}

    dpo_trainer = TrainerClass(
        model=model,
        ref_model=None,
        train_dataset=dpo_dataset_hf,
        **kw,
        args=ConfigClass(
            per_device_train_batch_size=config['finetune_batch_size'],
            gradient_accumulation_steps=config['finetune_gradient_accumulation_steps'],
            warmup_ratio=config['finetune_warmup_ratio'],
            num_train_epochs=config['finetune_num_epochs'],
            learning_rate=config['finetune_learning_rate'],
            logging_steps=5,
            optim=optimiser_str,
            seed=42,
            output_dir=str(finetune_output_dir),
            max_length=config['finetune_max_seq_length'],
            max_prompt_length=config['finetune_max_seq_length'] // 2,
            beta=config['finetune_beta'],
            weight_decay=config['finetune_weight_decay'],
            report_to="tensorboard", # Changed to tensorboard for local runs
            lr_scheduler_type="linear",
            bf16=use_bf16,
            fp16=use_fp16, # Ensure only one is true or both false
            remove_unused_columns=False,
            disable_tqdm=False,
            max_grad_norm=2.5, # be nice to the baseline model
        ),
    )

    # ------------------------------------------------------------------
    # Inject FTPO-specific h-params from config (if we’re in FTPO mode)
    # ------------------------------------------------------------------
    if config.get("finetune_mode", "ftpo").lower() == "ftpo":
        # names used inside FTPOTrainer.compute_loss
        _ftpo_keys = [
            "loss_mode",
            "beta",
            "lambda_kl_target_agg",
            "tau_kl_target_agg",
            "lambda_kl_target_tokenwise",
            "tau_kl_target_tokenwise",
            "lambda_kl",
            "loss_calc_mode",
            "clip_epsilon_probs",
            "clip_epsilon_logits",
        ]
        for k in _ftpo_keys:
            cfg_key = f"ftpo_{k}"
            if cfg_key in config and config[cfg_key] is not None:
                setattr(dpo_trainer, k, config[cfg_key])
    # ------------------------------------------------------------------

    if config.get("finetune_early_stopping_wins", None):
        # for ftpo
        dpo_trainer.add_callback(
            ThresholdStop("choice_win",
                        threshold=config["finetune_early_stopping_wins"],
                        higher_is_better=True)
        )

        # for dpo
        dpo_trainer.add_callback(
            ThresholdStop("rewards/accuracies",
                        threshold=config["finetune_early_stopping_wins"],
                        higher_is_better=True)
        )

    if config.get("finetune_early_stopping_loss", None) != None:
        dpo_trainer.add_callback(
            ThresholdStop("loss",
                        threshold=config["finetune_early_stopping_loss"],
                        higher_is_better=False)
        )


    if not use_unsloth:
        # replace the optimiser to fix nan issues when training qwen3

        from bitsandbytes.optim import PagedAdamW32bit
        from transformers.optimization import get_scheduler
        import torch        

        # build optimiser on trainable params only
        optim = PagedAdamW32bit(
            (p for p in model.parameters() if p.requires_grad),
            lr = config["finetune_learning_rate"],
        )
        
        # 1. replace the optimizer
        dpo_trainer.optimizer = optim

        # 2. compute the true number of *optimizer* steps
        steps_per_epoch = math.ceil(
            len(dpo_trainer.get_train_dataloader())
            / dpo_trainer.args.gradient_accumulation_steps
        )
        total_steps  = steps_per_epoch * dpo_trainer.args.num_train_epochs
        warmup_steps = int(total_steps * dpo_trainer.args.warmup_ratio)

        # 3. build a fresh scheduler on the new param groups
        dpo_trainer.lr_scheduler = get_scheduler(
            name=dpo_trainer.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    CLIP_GRADS = False
    # gradient-clip each step    
        
    if CLIP_GRADS: # can help if running into model collapse, but will slow training
        from transformers.trainer_callback import TrainerCallback
        class GradClipCb(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                torch.nn.utils.clip_grad_norm_(kw["model"].parameters(), 1.0)
                return control
        dpo_trainer.add_callback(GradClipCb())


    logger.info(f"Starting training. Output will be in {finetune_output_dir}. Check tensorboard for progress.")

    def lora_snapshot(m):
        return {n: p.detach().cpu().clone()
                for n,p in m.named_parameters() if p.requires_grad}

    before = lora_snapshot(model)          # snapshot *after* training

    try:
        trainer_stats = dpo_trainer.train()
        logger.info("DPO training finished.")
        if hasattr(trainer_stats, 'metrics'):
            logger.info(f"Trainer metrics: {trainer_stats.metrics}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return
    

    
    #dpo_trainer.train(resume_from_checkpoint=None, max_steps=1)  # 1 extra step
    after  = lora_snapshot(model)

    delta = {n: (after[n] - before[n]).abs().mean().item() for n in before}
    print("mean |Δ| across LoRA mats:", sum(delta.values()) / len(delta))

    if CALC_VAL_STATS:
        # ────────────────────────────────────────────────────────────────────
        # 3) POST-TRAIN statistics  (same API as above)
        # ────────────────────────────────────────────────────────────────────
        post_train_rows, post_train_stats = _gap_stats(model, train_ds.select(range(min(N_VAL_ITEMS, len(train_ds)))), collate, "train_post")
        post_val_rows , post_val_stats   = _gap_stats(model, val_ds.select(range(min(N_VAL_ITEMS, len(val_ds)))),   collate, "val_post")


        with open(analysis_dir / "train_post.jsonl", "w") as f:
            for r in post_train_rows: f.write(json.dumps(r) + "\n")
        with open(analysis_dir / "val_post.jsonl", "w") as f:
            for r in post_val_rows:  f.write(json.dumps(r) + "\n")

        print("\n— POST-TRAIN SUMMARY —")
        print(post_train_stats)
        print(post_val_stats)
        #print("sample train rows:", post_train_rows[:10])
        #print("sample val rows  :", post_val_rows [:10])

    
    

    # ── Quick sanity-check inference BEFORE merging ───────────────────────
    if False:
        try:
            test_prompt = (
                config.get("finetune_quick_test_prompt")                    # optional YAML/CLI override
                or "You are a creative storyteller.\n\n"
                "# User\n"
                "Write a short, engaging story about a princess named Elara in summertime.\n"
                "# Assistant\n"
            )
            model.eval()
            input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=600,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            generated = tokenizer.decode(
                gen_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            logger.info("\n—— quick inference sample (pre-merge) ——\n%s\n——————————", generated)
        except Exception as e:
            logger.warning("Quick inference test failed: %s", e)

    
    
    lora_dir   = finetune_output_dir / "lora_adapters"
    merged_dir = finetune_output_dir / "merged_16bit"

    # 1. always save the adapter (tiny, 4-bit or not)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    logger.info(f"LoRA adapters saved -> {lora_dir}")

    # 2. build a 16-bit merged checkpoint
    if config["finetune_load_in_4bit"]:                      # TRAINED IN 4-BIT
        logger.info("Reloading base model on CPU for fp16 merge …")

        # move current 4-bit graph away to free VRAM
        model.cpu(); torch.cuda.empty_cache(); gc.collect()

        if use_unsloth:
            # Unsloth’s loader already knows about Gemma-3 / Mistral-3 configs
            base_fp16, _ = FastLanguageModel.from_pretrained(
                model_name      = config["finetune_base_model_id"],
                max_seq_length  = config['finetune_max_seq_length'],      # same var used earlier
                load_in_4bit    = False,               # we want full-precision
                dtype           = torch.float16,
                device_map      = {"": "cpu"},
            )
        else:
            base_fp16 = AutoModelForCausalLM.from_pretrained(
                config["finetune_base_model_id"],
                torch_dtype = torch.float16,
                device_map  = {"": "cpu"},
                trust_remote_code = True,
            )
        model_fp16 = PeftModel.from_pretrained(
            base_fp16,
            lora_dir,                           # plug in the saved adapter
            device_map={"": "cpu"},
        )
        merged = model_fp16.merge_and_unload()  # pure fp16 torch.nn.Linear
        #detie_lm_head(merged)
        prepare_gemma3_for_save(merged)
        logger.info("Untied lm_head.weight from embed_tokens.weight")
    else:                                                   # TRAINED IN 16-BIT
        logger.info("Training was fp16/bf16 – merging in-place …")
        merged = model.merge_and_unload()       # still on GPU
        merged = merged.to(torch.float16).cpu() # push to CPU for writing
        
        if (getattr(merged.config, "model_type", "") or "").lower() == "gemma3":
            detie_lm_head(merged)          # clone lm_head so it no longer points at embeddings
            prepare_gemma3_for_save(merged)  # remove alias, set tie_word_embeddings=False

    # 3. write the merged checkpoint
    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,                # *.safetensors shards
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(merged_dir)
    fix_gemma3_checkpoint(merged_dir) # gemma3 models are saving weird.
    logger.info(f"Merged 16-bit model saved -> {merged_dir}")

    # --- Saving Model ---
    # (Saving logic remains the same)
    if False:
        try:
            lora_save_path = finetune_output_dir / "lora_adapters"
            dpo_trainer.save_model(str(lora_save_path)) 
            tokenizer.save_pretrained(str(lora_save_path))
            logger.info(f"LoRA adapters and tokenizer saved to {lora_save_path}")

            if config.get('finetune_save_merged_16bit'):
                merged_path = finetune_output_dir / "merged_16bit"
                logger.info(f"Saving merged 16-bit model to {merged_path}...")
                model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit", safe_serialization=True)
                logger.info(f"Merged 16-bit model saved to {merged_path}")

            if config.get('finetune_save_gguf_q8_0'):
                gguf_path = finetune_output_dir / "gguf_q8_0" 
                logger.info(f"Saving GGUF Q8_0 model to {gguf_path}.gguf ...")
                model.save_pretrained_gguf(str(gguf_path), tokenizer, quantization_method="q8_0")
                logger.info(f"GGUF Q8_0 model saved to {gguf_path}.gguf")

        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)

    logger.info("Finetuning process completed.")