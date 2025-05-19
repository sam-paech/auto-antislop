from typing import TYPE_CHECKING

# placeholders for later lazy loading
if TYPE_CHECKING:
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    from transformers import TextStreamer
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from trl import DPOTrainer, DPOConfig
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
from utils.dataset_helpers import load_tdpo_dataset, load_tdpo_multi_dataset
logger = logging.getLogger(__name__)

def load_imports():
    # --- Attempt to import Unsloth and related libraries only when this function is called ---
    try:
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer, TextStreamer # Added TextStreamer for potential inference example
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        from trl import DPOTrainer, DPOConfig
        from datasets import load_dataset
        from unsloth.chat_templates import get_chat_template
        import torch
        from torch.utils.data import default_collate
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence

        os.environ["UNSLOTH_DISABLE_COMPILATION"]     = "1"   # no Triton kernels
        os.environ["UNSLOTH_DISABLE_GRADIENT_OFFLOAD"] = "1"  # keep grads on GPU


        # Make all imports available in the global scope
        globals()['FastLanguageModel'] = FastLanguageModel
        globals()['AutoTokenizer'] = AutoTokenizer
        globals()['TextStreamer'] = TextStreamer
        globals()['AutoModelForCausalLM'] = AutoModelForCausalLM
        globals()['PeftModel'] = PeftModel
        globals()['DPOTrainer'] = DPOTrainer
        globals()['DPOConfig'] = DPOConfig
        globals()['load_dataset'] = load_dataset
        globals()['get_chat_template'] = get_chat_template
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
    load_imports()

    logger.info("Starting finetuning process...")

    from core.last_token_dpo_trainer import LastTokenDPOTrainer
    


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
    mode = config.get("finetune_mode", "tdpo-multi").lower()   # expect "dpo" or "tdpo" or "tdpo-multi"
    dataset_path = None
    # --- Model and Tokenizer Setup ---    
    max_seq_length = config['finetune_max_seq_length']

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
        dpo_dataset_hf = load_tdpo_dataset(
            tdpo_dataset_path, tokenizer,
            max_seq_len          = max_seq_length,
            rule_reg_strength    = config.get("finetune_tdpo_sample_regularisation_strength", 0.0),
        )
        dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))
        max_train = config.get("finetune_max_train_examples")
        if isinstance(max_train, int) and max_train > 0 and len(dpo_dataset_hf) > max_train:
            dpo_dataset_hf = dpo_dataset_hf.select(range(max_train))
            logger.info(f"Capped training dataset to {max_train} examples.")

        
        # ──────────────────────────────────────────────────────────────
        # [DEBUG] Inspect last-5 prompt tokens + chosen / rejected token
        #         –– prints up to 50 TDPO examples for a quick sanity check.
        #         –– gated by new config flag `finetune_debug_tdpo_tokens`.
        # ──────────────────────────────────────────────────────────────
        sample_n = min(50, len(dpo_dataset_hf))
        print(f"\n🔎 TDPO debug: showing {sample_n} examples "
            "(last-5 prompt tokens, chosen ▸ rejected)\n")
        for i, ex in enumerate(dpo_dataset_hf.select(range(sample_n))):
            tail_prompt = tokenizer.convert_ids_to_tokens(ex["prompt_ids"][-5:])
            chosen_tok  = tokenizer.convert_ids_to_tokens([ex["chosen_token_id"]])[0]
            rejected_tok = tokenizer.convert_ids_to_tokens([ex["rejected_token_id"]])[0]
            tail_str = " ".join(tail_prompt)
            print(f"{i:03d}: … {tail_str}  →  {chosen_tok} ▸ {rejected_tok}")
        print("\n—— end TDPO debug ——\n")

    elif mode == "tdpo-multi":
        if config.get("finetune_tdpo_dataset"):
            dataset_path = Path(config["finetune_tdpo_dataset"])
        else:
            tdpo_files = sorted(experiment_run_dir.glob("iter_*_tdpo_pairs.jsonl"))
            if not tdpo_files:
                logger.error("No TDPO files found for TDPO-MULTI.")
                return
            dataset_path = tdpo_files[-1]

        dpo_dataset_hf = load_tdpo_multi_dataset(
            dataset_path, tokenizer,
            max_seq_len=max_seq_length,
            rule_reg_strength    = config.get("finetune_tdpo_sample_regularisation_strength", 0.0),
        )
        
        dpo_dataset_hf = dpo_dataset_hf.shuffle(seed=config.get("finetune_shuffle_seed", 3407))
        max_train = config.get("finetune_max_train_examples")
        if isinstance(max_train, int) and max_train > 0 and len(dpo_dataset_hf) > max_train:
            dpo_dataset_hf = dpo_dataset_hf.select(range(max_train))
            logger.info(f"Capped training dataset to {max_train} examples.")

        
        # ──────────────────────────────────────────────────────────────
        # [DEBUG] Inspect last-5 prompt tokens + chosen / rejected token
        #         –– prints up to 50 TDPO examples for a quick sanity check.
        #         –– gated by new config flag `finetune_debug_tdpo_tokens`.
        # ──────────────────────────────────────────────────────────────
        sample_n = min(50, len(dpo_dataset_hf))
        print(f"\n🔎 TDPO-multi debug: showing {sample_n} examples "
            "(last-5 prompt tokens, chosen ▸ rejected)\n")
        for i, ex in enumerate(dpo_dataset_hf.select(range(sample_n))):
            tail_prompt = tokenizer.convert_ids_to_tokens(ex["prompt_ids"][-5:])
            chosen_tok  = tokenizer.convert_ids_to_tokens([ex["chosen_ids"][0]])[0]
            rejected_tok = tokenizer.convert_ids_to_tokens([ex["rejected_token_id"]])[0]
            tail_str = " ".join(tail_prompt)
            print(f"{i:03d}: … {tail_str}  →  {chosen_tok} ▸ {rejected_tok}")
        print("\n—— end TDPO-multi debug ——\n")

    else:
        logger.error(f"Unknown finetune_mode '{mode}'. Use 'dpo' or 'tdpo'.")
        return
    
        

    import torch
    
    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=config['finetune_load_in_4bit'],
            dtype=torch.bfloat16 if (not config['finetune_load_in_4bit']) and torch.cuda.is_bf16_supported() else None,
            compile_model      = False,
        )
        
    except Exception as e:
        logger.error(f"Failed to load base model '{model_name}' or tokenizer for DPO: {e}", exc_info=True)
        return
    

    
    # Hard-disable gradient-checkpointing for TDPO
    if mode in ["tdpo", "tdpo-multi"]:
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

    if mode in ["tdpo", "tdpo-multi"]:
        model.config._attn_implementation = "flash_attention_2"
        # turn off every ckpt flag Unsloth uses
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        for mod in model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = False
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = False


    

    #model.config._attn_implementation = "sdpa"




    CALC_VAL_STATS = False
    if CALC_VAL_STATS:
        def _collate_tdpo(features, pad_id: int, max_len: int):
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

            # ── TDPO-MULTI vs single-token path ──────────────────────────
            if "chosen_ids" in features[0]:              # multi-chosen
                max_c = max(len(f["chosen_ids"]) for f in features)
                chosen_pad  = torch.full((B, max_c), -100, dtype=torch.long)
                chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
                for i, f in enumerate(features):
                    ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
                    chosen_pad [i, :ids.size(0)] = ids
                    chosen_mask[i, :ids.size(0)] = True
                batch = dict(chosen_ids=chosen_pad, chosen_mask=chosen_mask)
            else:                                        # single-token TDPO
                batch = dict(chosen_token_id=torch.tensor(
                            [f["chosen_token_id"] for f in features]))

            # ── always include rejected & prompt tensors ─────────────────
            batch.update(
                prompt_ids        = prompt_ids,
                attention_mask    = attn_mask,
                rejected_token_id=torch.tensor([f["rejected_token_id" if "rejected_token_id" in f else "rejected_id"] for f in features]),
            )
            return batch



        
        def _gap_stats(model, dataset, collate_fn, tag,
               batch_size=2, device=None):
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

                    # ----- variant detection -----
                    if "chosen_ids" in batch:                      # TDPO-MULTI
                        ch_ids  = batch["chosen_ids"].to(device)
                        ch_mask = batch["chosen_mask"].to(device)
                        gathered = logp_all.gather(-1, ch_ids).masked_fill(~ch_mask, -1e9)
                        lp_good = torch.logsumexp(gathered, dim=-1)        # [B]
                    else:                                            # TDPO
                        ch      = batch["chosen_token_id"].to(device)
                        lp_good = logp_all.gather(-1, ch.unsqueeze(-1)).squeeze(-1)

                    lp_bad  = logp_all.gather(-1, rej.unsqueeze(-1)).squeeze(-1)
                    delta   = lp_good - lp_bad

                    tot_delta += delta.sum().item()
                    wins      += (delta > 0).sum().item()

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

        pad_id = tokenizer.pad_token_id
        collate = lambda feats: _collate_tdpo(feats, pad_id, max_seq_length)


        if True: # skip this check for now
            pre_train_rows, pre_train_stats = _gap_stats(model, train_ds.select(range(min(200, len(train_ds)))), collate, "train_pre")
            pre_val_rows , pre_val_stats   = _gap_stats(model, val_ds.select(range(min(200, len(val_ds)))),   collate, "val_pre")


            with open(analysis_dir / "train_pre.jsonl", "w") as f:
                for r in pre_train_rows: f.write(json.dumps(r) + "\n")
            with open(analysis_dir / "val_pre.jsonl", "w") as f:
                for r in pre_val_rows:  f.write(json.dumps(r) + "\n")

            print("\n— PRE-TRAIN SUMMARY —")
            print(pre_train_stats)
            print(pre_val_stats)
            print("sample train rows:", pre_train_rows[:10])
            print("sample val rows  :", pre_val_rows [:10])


    import gc
    #gc.collect()
    #torch.cuda.empty_cache()
    #torch.cuda.reset_peak_memory_stats()



    #freeze_early_layers(model, n_unfrozen = 8, verbose = True)


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
    LR_SCALE_CONST = 0.15
    eta0     = 2e-4           # base LR before scaling

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
            * LR_SCALE_CONST
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




    TrainerClass = LastTokenDPOTrainer if mode.lower() in ["tdpo", "tdpo-multi"] else DPOTrainer

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
            optim="paged_adamw_32bit",
            seed=42,
            output_dir=str(finetune_output_dir),
            max_length=max_seq_length,
            max_prompt_length=max_seq_length // 2,
            beta=config['finetune_beta'],
            weight_decay=config['finetune_weight_decay'],
            report_to="tensorboard", # Changed to tensorboard for local runs
            lr_scheduler_type="linear",
            bf16=use_bf16,
            fp16=use_fp16, # Ensure only one is true or both false
            remove_unused_columns=False,
            disable_tqdm=False,
            max_grad_norm=0.5, # be nice to the baseline model
        ),
    )


    # -----------------------------------------------------------
    # after you create dpo_trainer  (just before .train())
    # -----------------------------------------------------------
    from bitsandbytes.optim import PagedAdamW32bit
    from transformers.trainer_callback import TrainerCallback
    import torch

    # 1) build the optimiser on the trainable parameters only
    optim = PagedAdamW32bit(
        (p for p in dpo_trainer.model.parameters() if p.requires_grad),
        lr=config["finetune_learning_rate"],
    )

    # 2) inject it into the trainer / accelerator
    dpo_trainer.optimizer = optim
    dpo_trainer.accelerator.optimizer = optim   # keeps scheduler in sync

    # 3) lightweight grad-norm clip each step
    def _clip_before_step(trainer, args, state, control, **kw):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)

    class GradClipCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # kwargs always contains "model" when called from Trainer
            torch.nn.utils.clip_grad_norm_(kwargs["model"].parameters(), 1.0)
            return control                         # must return the control object

    # after you build dpo_trainer and inject the optimiser
    dpo_trainer.add_callback(GradClipCallback())

    print("⇢  using optimiser:", optim.__class__.__name__)   # should say PagedAdamW32bit
    # -----------------------------------------------------------





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
        post_train_rows, post_train_stats = _gap_stats(model, train_ds.select(range(min(200, len(train_ds)))), collate, "train_post")
        post_val_rows , post_val_stats   = _gap_stats(model, val_ds.select(range(min(200, len(val_ds)))),   collate, "val_post")


        with open(analysis_dir / "train_post.jsonl", "w") as f:
            for r in post_train_rows: f.write(json.dumps(r) + "\n")
        with open(analysis_dir / "val_post.jsonl", "w") as f:
            for r in post_val_rows:  f.write(json.dumps(r) + "\n")

        print("\n— POST-TRAIN SUMMARY —")
        print(post_train_stats)
        print(post_val_stats)
        print("sample train rows:", post_train_rows[:10])
        print("sample val rows  :", post_val_rows [:10])

    
    

    # ── Quick sanity-check inference BEFORE merging ───────────────────────
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

        base_fp16 = AutoModelForCausalLM.from_pretrained(
            config["finetune_base_model_id"],
            torch_dtype=torch.float16,          # or bfloat16
            device_map={"": "cpu"},             # load straight to CPU
        )
        model_fp16 = PeftModel.from_pretrained(
            base_fp16,
            lora_dir,                           # plug in the saved adapter
            device_map={"": "cpu"},
        )
        merged = model_fp16.merge_and_unload()  # pure fp16 torch.nn.Linear
    else:                                                   # TRAINED IN 16-BIT
        logger.info("Training was fp16/bf16 – merging in-place …")
        merged = model.merge_and_unload()       # still on GPU
        merged = merged.to(torch.float16).cpu() # push to CPU for writing

    # 3. write the merged checkpoint
    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,                # *.safetensors shards
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(merged_dir)
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