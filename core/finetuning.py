from typing import TYPE_CHECKING

# placeholders for later lazy loading
if TYPE_CHECKING:
    from unsloth import FastLanguageModel as FastLanguageModelType
    from transformers import AutoTokenizer as AutoTokenizerType
    from transformers import TextStreamer as TextStreamerType
    from transformers import AutoModelForCausalLM as AutoModelForCausalLMType
    from peft import PeftModel as PeftModelType
    from trl import DPOTrainer as DPOTrainerType, DPOConfig as DPOConfigType
    from datasets import Dataset as DatasetType
    from unsloth.chat_templates import get_chat_template as get_chat_template_type
    import torch # Keep torch as it might be used for GPU checks earlier if needed    
    # typing.Optional can be imported at the top level if used in type hints outside the function
    from torch.utils.data import default_collate
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence    
    

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import json
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




# ---------------------------------------------------------------------
# 1.  Dataset loader for “final-token DPO”
# ---------------------------------------------------------------------


from pathlib import Path
from typing import Collection, Optional

from datasets import load_dataset


def load_tdpo_dataset(
    path: Path,
    tokenizer,
    *,
    max_seq_len: int = 4096,
    rule_reg_strength: float = 0.0,
    stop_words: Optional[Collection[str]] = None,
):
    """
    Read a TDPO jsonl file → HF Dataset of prompt-ids & final-token ids.
    If rule_reg_strength > 0, re-sample the examples so that over-frequent
    validator.rules are down-weighted.  Output size stays identical.

    Additional rule: drop any sample whose *rejected* suffix (after
    whitespace-trimming) is a stop-word.
    """

    import random
    import numpy as np
    from collections import Counter

    # ── stop-word setup ----------------------------------------------------
    if stop_words is None:
        stop_words = {
            "the", "a", "an", "in", "on", "at", "by", "for", "to", "of", "and",
            "or", "but", "if", "then", "else", "when", "where", "how", "why",
            "what", "who", "whom", "this", "that", "these", "those", "is", "are",
            "was", "were", "be", "being", "been", "have", "has", "had", "do",
            "does", "did", "will", "would", "shall", "should", "can", "could",
            "may", "might", "must"
        }
    stop_words = set(w.lower() for w in stop_words)

    # ── raw load -----------------------------------------------------------
    ds = load_dataset("json", data_files=str(path), split="train").select(range(20000))

    # ── optional rule-balanced re-sample ----------------------------------
    if rule_reg_strength and rule_reg_strength > 0:
        rules = [
            ex["validator"]["rule"] if isinstance(ex.get("validator"), dict) else None
            for ex in ds
        ]
        counts = Counter(r for r in rules if r is not None)
        if counts:
            thresh = np.median(list(counts.values()))
            w_rule = {
                r: 1.0 if c <= thresh else (thresh / c) ** rule_reg_strength
                for r, c in counts.items()
            }
            w_example = [w_rule.get(r, 1.0) for r in rules]
            probs = np.asarray(w_example, dtype=np.float64)
            probs /= probs.sum()

            rng = np.random.default_rng(3407)            # reproducible
            idx = rng.choice(len(ds), size=len(ds), replace=False, p=probs)
            idx.sort()
            ds = ds.select(idx.tolist())

    # ── tokenisation / sanity checks --------------------------------------
    tokenizer.truncation_side = "left"

    def _tok(ex):
        prompt_ids = tokenizer(
            ex["context_with_chat_template"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        ).input_ids

        # tokenize the candidate suffixes *as-is* (could include the ▁ boundary)
        ch_ids = tokenizer(ex["chosen_decoded"],   add_special_tokens=False).input_ids
        rj_ids = tokenizer(ex["rejected_decoded"], add_special_tokens=False).input_ids

        # ── Reject rows where suffix is not exactly ONE token --------------
        if len(ch_ids) != 1 or len(rj_ids) != 1:
            _tok.multi_tok_rows += 1
            logger.error('! failed tokenisation -- ' + str(len(ch_ids)) + ' -- ' + str(len(rj_ids)) + ' -- ' +str(ex["chosen_decoded"]) + ' -- ' +str(ex["rejected_decoded"]))
            return {
                "prompt_ids":        [],
                "chosen_token_id":    0,
                "rejected_token_id":  0,
                "__valid":           False,
            }

        # ── New rule: rejected token must not be a stop-word ---------------
        # Regex patterns sometimes ban patterns beginning with a stop word.
        # This can cause issues with model coherence if we train away from those probabilities.
        # So we'll filter any of those out.
        # If you want to include these rules with full effect, you can comment this section out.
        rj_text = ex["rejected_decoded"].strip().lower()
        if rj_text in stop_words:
            return {
                "prompt_ids":        [],
                "chosen_token_id":    0,
                "rejected_token_id":  0,
                "__valid":           False,
            }
        # ──

        ok = (
            len(prompt_ids) + 1 <= max_seq_len
            and ch_ids and rj_ids
            and ch_ids[-1] != rj_ids[-1]
        )
        if not ok:                     # ← prompt too long, empty suffix, etc.
            return {
                "prompt_ids":        [],
                "chosen_token_id":    0,
                "rejected_token_id":  0,
                "__valid":           False,
            }

        return {
            "prompt_ids":       prompt_ids,
            "chosen_token_id":   ch_ids[-1],
            "rejected_token_id": rj_ids[-1],
            "__valid":           True,
        }

    _tok.multi_tok_rows = 0            # attribute needed before first call
    ds = ds.map(_tok, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: ex["__valid"]).remove_columns("__valid")

    if len(ds) == 0:
        raise ValueError("no TDPO samples survived length / sanity checks")

    return ds.shuffle(seed=3407)




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


    # putting the class here because we're lazy loading unsloth + other imports
    # yes it's messy =/
    class LastTokenDPOTrainer(DPOTrainer):
        """
        Trainer for “single-next-token” (TDPO) preference learning.
        Replaces TRL’s standard loss with a log-ratio on the **last**
        autoregressive position and stays agnostic to model head names.
        """

        # ──────────────────────────────────────────────────────────────
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.remove_unused_columns = False
            self.data_collator = self.tdpo_collator                    # override

        # ──────────────────────────────────────────────────────────────
        @staticmethod
        def _get_proj(model):
            """
            Return the output-projection module in a model-agnostic way.
            Falls back to `lm_head` if `get_output_embeddings()` is None.
            """
            proj = model.get_output_embeddings()
            if proj is None:
                proj = getattr(model, "lm_head", None)
            if proj is None:
                raise AttributeError(
                    "Model lacks both get_output_embeddings() and lm_head."
                )
            return proj

        # ──────────────────────────────────────────────────────────────
        def tdpo_collator(self, features):
            pad_id = self.padding_value
            max_len = self.args.max_length        # 4096 from your config

            # regular left-pad so we keep right-alignment semantics
            prompt_tensors = [torch.tensor(f["prompt_ids"]) for f in features]
            prompt_ids = pad_sequence(prompt_tensors,
                                    batch_first=True,
                                    padding_value=pad_id)

            # force-pad to static length to kill shape sprawl
            if prompt_ids.size(1) < max_len:
                pad_cols = max_len - prompt_ids.size(1)
                prompt_ids = F.pad(prompt_ids, (pad_cols, 0), value=pad_id)

            attention_mask = prompt_ids.ne(pad_id)

            chosen   = torch.tensor([f["chosen_token_id"]   for f in features])
            rejected = torch.tensor([f["rejected_token_id"] for f in features])

            return dict(prompt_ids=prompt_ids,
                        attention_mask=attention_mask,
                        chosen_token_id=chosen,
                        rejected_token_id=rejected)

        # 
        # ──────────────────────────────────────────────────────────────
        # this version maybe worked ok if constrained to lm_head and given a v. high lr
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            ids      = inputs["prompt_ids"].to(model.device)           # [B, L]
            attn     = inputs["attention_mask"].to(model.device)       # [B, L]
            chosen   = inputs["chosen_token_id"].to(model.device)      # [B]
            rejected = inputs["rejected_token_id"].to(model.device)    # [B]

            # ── policy forward ───────────────────────────────────────
            out = model(ids, attention_mask=attn,
                        use_cache=False, output_hidden_states=False)

            last_index  = attn.sum(1) - 1                              # [B]
            logits_last = out.logits[torch.arange(ids.size(0),
                                                device=model.device),
                                    last_index]                       # [B, V]

            logp_good = F.log_softmax(logits_last, -1).gather(-1,
                        chosen.unsqueeze(-1)).squeeze(-1)
            logp_bad  = F.log_softmax(logits_last, -1).gather(-1,
                        rejected.unsqueeze(-1)).squeeze(-1)

            # ── reference forward ────────────────────────────────────
            with torch.no_grad():
                if self.ref_model is None:
                    # temporarily disable adapters to get base model behaviour
                    with self.null_ref_context():
                        ref_out = model(ids, attention_mask=attn, use_cache=False)
                else:
                    ref_out = self.ref_model(ids, attention_mask=attn, use_cache=False)

                ref_logits_last = ref_out.logits[torch.arange(ids.size(0), device=model.device),
                                                last_index]   # [B, V]

                ref_good = F.log_softmax(ref_logits_last, -1).gather(-1,
                        chosen.unsqueeze(-1)).squeeze(-1)
                ref_bad  = F.log_softmax(ref_logits_last, -1).gather(-1,
                        rejected.unsqueeze(-1)).squeeze(-1)

            # ── DPO scalar loss ──────────────────────────────────────
            delta = (logp_good - ref_good) - (logp_bad - ref_bad)
            loss  = -F.logsigmoid(self.beta * delta).mean()

            if return_outputs:
                return loss, {"chosen_wins": (delta > 0).float().mean().detach()}
            return loss


        # ---------------------------------------------------------------------
        # LastTokenDPOTrainer.compute_loss
        # ---------------------------------------------------------------------
        # – Adds a token-wise KL regulariser that *ignores* the last position
        #   so we still tether the prompt/context to the reference model but
        #   let the single decision token drift freely.
        # – Keeps the linear KL-ramp (0 → kl_coeff_target over kl_ramp_pct of
        #   total steps).
        # ---------------------------------------------------------------------

        # this version attempts to minimise the effect of the context on what we're training
        # to focus on the chosen/rejected final tokens
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            # ──────────────────────────────────────────────────────────────
            # 1. Unpack batch tensors
            # ──────────────────────────────────────────────────────────────
            ids      = inputs["prompt_ids"].to(model.device)           # [B,L]
            attn     = inputs["attention_mask"].to(model.device)       # [B,L]  (1 = real token)
            chosen   = inputs["chosen_token_id"].to(model.device)      # [B]
            rejected = inputs["rejected_token_id"].to(model.device)    # [B]

            # ──────────────────────────────────────────────────────────────
            # 2. Forward pass – current policy
            # ──────────────────────────────────────────────────────────────
            out = model(ids, attention_mask=attn, use_cache=False)
            last_idx = attn.sum(1) - 1                                 # [B]
            logits_last = out.logits[torch.arange(ids.size(0), device=model.device),
                                    last_idx]                         # [B,V]

            logp_good = F.log_softmax(logits_last, -1)\
                        .gather(-1, chosen.unsqueeze(-1)).squeeze(-1)    # [B]
            logp_bad  = F.log_softmax(logits_last, -1)\
                        .gather(-1, rejected.unsqueeze(-1)).squeeze(-1)  # [B]

            # ──────────────────────────────────────────────────────────────
            # 3. Forward pass – reference policy
            #    (either a separate ref_model or the base model with adapters
            #     disabled if self.ref_model is None)
            # ──────────────────────────────────────────────────────────────
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_out = model(ids, attention_mask=attn, use_cache=False)
                else:
                    ref_out = self.ref_model(ids, attention_mask=attn, use_cache=False)

                ref_logits_last = ref_out.logits[torch.arange(ids.size(0), device=model.device),
                                                last_idx]

                ref_good = F.log_softmax(ref_logits_last, -1)\
                            .gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
                ref_bad  = F.log_softmax(ref_logits_last, -1)\
                            .gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

            # ──────────────────────────────────────────────────────────────
            # 4. TDPO preference loss  (only last token)
            # ──────────────────────────────────────────────────────────────
            delta      = (logp_good - ref_good) - (logp_bad - ref_bad)
            pref_loss  = -F.logsigmoid(self.beta * delta).mean()

            # ──────────────────────────────────────────────────────────────
            # 5. Token-wise KL on *prompt/context* only
            #    • Compute KL(policy ‖ ref) at every position.
            #    • Mask out padding tokens AND the last position so the
            #      regulariser cannot fight the preference update.
            # ──────────────────────────────────────────────────────────────
            logp_all = F.log_softmax(out.logits,  dim=-1)              # [B,L,V]
            ref_prob = F.softmax   (ref_out.logits, dim=-1)            # [B,L,V]

            kl_tok = F.kl_div(logp_all, ref_prob, reduction="none",
                            log_target=False).sum(-1)                # [B,L]

            kl_tok *= attn                                            # zero pads
            batch_idx = torch.arange(ids.size(0), device=model.device)
            kl_tok[batch_idx, last_idx] = 0.0                         # zero last token

            kl_loss = kl_tok.sum() / attn.sum()                       # mean over real tokens

            # ──────────────────────────────────────────────────────────────
            # 6. Linear KL-coefficient ramp  (0 → kl_coeff_target)
            # ──────────────────────────────────────────────────────────────
            if not hasattr(self, "_kl_setup_done"):
                self.kl_coeff_target = 5e-2
                ramp_frac  = getattr(self, "kl_ramp_pct", 0.4)
                total_steps = (self.args.num_train_epochs
                            * len(self.train_dataset)
                            // (self.args.per_device_train_batch_size
                                * self.args.gradient_accumulation_steps))
                self.kl_ramp_steps = max(1, int(ramp_frac * total_steps))
                self._kl_setup_done = True

            step = self.state.global_step
            kl_coeff = ( self.kl_coeff_target * step / self.kl_ramp_steps
                        if step < self.kl_ramp_steps else
                        self.kl_coeff_target )

            # ──────────────────────────────────────────────────────────────
            # 7. Total loss
            # ──────────────────────────────────────────────────────────────
            loss = pref_loss + kl_coeff * kl_loss

            # ──────────────────────────────────────────────────────────────
            # 8. Log component metrics for Trainer.log()
            # ──────────────────────────────────────────────────────────────
            metrics = {
                "pref_loss":  pref_loss.detach(),
                "kl_loss":    kl_loss.detach(),
                "kl_coeff":   torch.tensor(kl_coeff),
                "chosen_win": (delta > 0).float().mean().detach(),
            }
            self.store_metrics(metrics, train_eval="train")

            if return_outputs:
                return loss, metrics
            return loss

        # version targeting only the last token position
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            ids      = inputs["prompt_ids"].to(model.device)         # [B,L]
            attn     = inputs["attention_mask"].to(model.device)     # [B,L]
            chosen   = inputs["chosen_token_id"].to(model.device)    # [B]
            rejected = inputs["rejected_token_id"].to(model.device)  # [B]

            # ── split positions -----------------------------------------------------
            last_idx   = attn.sum(1) - 1                             # [B]
            ctx_ids    = ids.clone()
            tok_ids    = torch.zeros_like(ids)                       # will hold only the last token
            for b, idx in enumerate(last_idx):
                ctx_ids[b, idx] = self.padding_value                 # mask last token out
                tok_ids[b, idx] = ids[b, idx]                        # keep only last token
            ctx_attn   = ctx_ids.ne(self.padding_value)
            tok_attn   = tok_ids.ne(0)                               # 1 at last token

            # ── 1 ▸ forward context *without* grad -------------------------------
            with torch.no_grad():
                ctx_out = model(
                    ctx_ids,
                    attention_mask=ctx_attn,
                    use_cache=True,            # we need past_key_values
                    return_dict=True,
                )
                past_kv = ctx_out.past_key_values

            # ── 2 ▸ forward last token *with* grad -------------------------------
            tok_out = model(
                tok_ids,
                attention_mask=tok_attn,
                past_key_values=past_kv,
                use_cache=False,
                return_dict=True,
            )
            logits_last = tok_out.logits[torch.arange(ids.size(0), device=model.device), last_idx]

            #del past_kv, ctx_out, tok_out
            #torch.cuda.empty_cache()

            # ── log-probs ----------------------------------------------------------
            logp_good = F.log_softmax(logits_last, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            logp_bad  = F.log_softmax(logits_last, -1).gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

            # ── reference model (no grad) -----------------------------------------
            if False: # not needed when not using ref
                with torch.no_grad():
                    if self.ref_model is None:
                        # OLD VERSION: Wrong because didn't recompute whole context with lora detached:                    
                        with self.null_ref_context():
                            ref_tok_out = model(
                                tok_ids,
                                attention_mask=tok_attn,
                                past_key_values=past_kv,
                                use_cache=False,
                                return_dict=True,
                            )
                        if False:
                            with self.null_ref_context():          # LoRA disabled
                                ref_ctx_out = model(
                                    ctx_ids,
                                    attention_mask=ctx_attn,
                                    use_cache=True,
                                    return_dict=True,
                                )
                                ref_tok_out = model(
                                    tok_ids,
                                    attention_mask=tok_attn,
                                    past_key_values=ref_ctx_out.past_key_values,
                                    use_cache=False,
                                    return_dict=True,
                                )
                    else:
                        ref_tok_out = self.ref_model(
                            tok_ids,
                            attention_mask=tok_attn,
                            past_key_values=past_kv,
                            use_cache=False,
                            return_dict=True,
                        )
                    ref_logits_last = ref_tok_out.logits[torch.arange(ids.size(0), device=model.device), last_idx]
                    ref_good = F.log_softmax(ref_logits_last, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
                    ref_bad  = F.log_softmax(ref_logits_last, -1).gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

            if False: # normal path
                # ── preference loss (only last-token path) ----------------------------
                delta     = (logp_good - ref_good) - (logp_bad - ref_bad)
                #pref_loss = -F.logsigmoid(self.beta * delta).mean()
                pref_loss = -F.logsigmoid(3.0 * delta).mean()

                # ── (optional) disable prompt-level KL completely ---------------------
                kl_loss   = torch.tensor(0.0, device=model.device)
            
            else: # testing path with reference term disabled
                # delta = (logp_good - ref_good) - (logp_bad - ref_bad)
                delta = logp_good - logp_bad            # <- raw gap, no tether
                pref_loss = -F.logsigmoid(self.beta * delta).mean()
                kl_loss   = 0.0


            loss = pref_loss + kl_loss                       # kl_loss is zero, but left for clarity

            rho         = logp_good - logp_bad
            choice_win  = (rho > 0).float().mean().detach()
            metrics = {
                "pref_loss":  pref_loss.detach(),
                #"chosen_win": (delta > 0).float().mean().detach(),
                "choice_win": choice_win,              # ← new behaviour metric
            }
            self.store_metrics(metrics, train_eval="train")

            if return_outputs:
                return loss, metrics
            return loss
        # --- end new compute_loss ----------------------------------------------------

        # --------------------------------------------------------------------- #
        #  Allowed modes:
        #     "ref"   – reference-tethered preference loss
        #     "free"  – untethered (raw Δ)
        #     "clip"  – odds-ratio clipping with ε
        # --------------------------------------------------------------------- #
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            mode = getattr(self, "loss_mode", "clip")   # set externally
            eps  = getattr(self, "clip_eps", 0.2)       # only used in "clip"
            beta = getattr(self, "beta", 0.1)           # reused everywhere

            # ── unpack ---------------------------------------------------------
            ids      = inputs["prompt_ids"].to(model.device)       # [B,L]
            attn     = inputs["attention_mask"].to(model.device)   # [B,L]
            chosen   = inputs["chosen_token_id"].to(model.device)  # [B]
            rejected = inputs["rejected_token_id"].to(model.device)  # [B]

            # ── split context vs last token -----------------------------------
            last_idx = attn.sum(1) - 1                             # [B]
            ctx_ids  = ids.clone()
            tok_ids  = torch.zeros_like(ids)
            for b, idx in enumerate(last_idx):
                ctx_ids[b, idx] = self.padding_value
                tok_ids[b, idx] = ids[b, idx]
            ctx_attn = ctx_ids.ne(self.padding_value)
            tok_attn = tok_ids.ne(0)

            # ── 1 ▸ forward context (no grad) ---------------------------------
            with torch.no_grad():
                ctx_out = model(
                    ctx_ids,
                    attention_mask=ctx_attn,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = ctx_out.past_key_values

            # ── 2 ▸ forward last token (grad) ---------------------------------
            tok_out = model(
                tok_ids,
                attention_mask=tok_attn,
                past_key_values=past_kv,
                use_cache=False,
                return_dict=True,
            )
            logits_last = tok_out.logits[torch.arange(ids.size(0), device=model.device), last_idx]

            # ── log-probs ------------------------------------------------------
            logp_good = F.log_softmax(logits_last, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            logp_bad  = F.log_softmax(logits_last, -1).gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

            # ───────────────────────────────────────────────────────────────────
            #  Variant-specific preference term
            # ───────────────────────────────────────────────────────────────────
            if mode == "ref":
                # compute reference logits once, only if needed
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.null_ref_context():
                            ref_tok_out = model(
                                tok_ids,
                                attention_mask=tok_attn,
                                past_key_values=past_kv,
                                use_cache=False,
                                return_dict=True,
                            )
                    else:
                        ref_tok_out = self.ref_model(
                            tok_ids,
                            attention_mask=tok_attn,
                            past_key_values=past_kv,
                            use_cache=False,
                            return_dict=True,
                        )
                    ref_logits_last = ref_tok_out.logits[torch.arange(ids.size(0), device=model.device), last_idx]
                    ref_good = F.log_softmax(ref_logits_last, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
                    ref_bad  = F.log_softmax(ref_logits_last, -1).gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

                delta = (logp_good - ref_good) - (logp_bad - ref_bad)
                pref_loss = -F.logsigmoid(beta * delta).mean()
                kl_loss   = 0.0                                # keep prompt-KL off for now

            elif mode == "clip":
                # PPO-style odds-ratio clipping
                delta  = logp_good - logp_bad                 # log-odds
                ratio  = torch.exp(delta)                     # odds ratio
                clipped = torch.minimum(ratio, torch.tensor(1.0 + eps, device=ratio.device))
                pref_loss = -torch.log(clipped / (1.0 + eps)).mean()
                kl_loss   = 0.0

            else:  # "free"
                delta = logp_good - logp_bad
                pref_loss = -F.logsigmoid(beta * delta).mean()
                kl_loss   = 0.0

            # ── total loss & metrics ------------------------------------------
            loss = pref_loss + kl_loss

            rho        = logp_good - logp_bad
            choice_win = (rho > 0).float().mean().detach()
            metrics = {
                "pref_loss":  pref_loss.detach(),
                "choice_win": choice_win,
            }
            self.store_metrics(metrics, train_eval="train")

            #with torch.no_grad():
            #    hist = ratio.cpu().log10().numpy()
            #    print("log10 ratio stats:", hist.min(), hist.mean(), hist.max())

            if return_outputs:
                return loss, metrics
            return loss



        # ----------------------------------------------------------
        def _prepare_dataset(self, dataset, *args, **_):
            return dataset


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

    else:
        logger.error(f"Unknown finetune_mode '{mode}'. Use 'dpo' or 'tdpo'.")
        return


    
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
        #lora_dropout=config['finetune_lora_dropout'],
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






    CALC_VAL_STATS = True
    if CALC_VAL_STATS:
        def _collate_tdpo(features, pad_id: int, max_len: int):
            # tensors → left-pad → static length
            prompt_tensors = [torch.tensor(f["prompt_ids"]) for f in features]
            prompt_ids = pad_sequence(prompt_tensors,
                                    batch_first=True,
                                    padding_value=pad_id)
            if prompt_ids.size(1) < max_len:
                pad_cols = max_len - prompt_ids.size(1)
                prompt_ids = F.pad(prompt_ids, (0, pad_cols), value=pad_id)

            attn = prompt_ids.ne(pad_id)
            chosen   = torch.tensor([f["chosen_token_id"]   for f in features])
            rejected = torch.tensor([f["rejected_token_id"] for f in features])

            return dict(prompt_ids=prompt_ids,
                        attention_mask=attn,
                        chosen_token_id=chosen,
                        rejected_token_id=rejected)

        from torch.utils.data import DataLoader
        def _gap_stats(model, dataset, collate_fn, tag,
               ref_model=None, use_null_ref=False, batch_size=2):
            model.eval()
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)

            rows, tot_delta, wins = [], 0.0, 0
            with torch.no_grad():
                for batch in loader:
                    ids   = batch["prompt_ids"     ].to(model.device)
                    attn  = batch["attention_mask" ].to(model.device)
                    good  = batch["chosen_token_id"].to(model.device)
                    bad   = batch["rejected_token_id"].to(model.device)
                    last  = attn.sum(1) - 1

                    # --- policy forward ------------------------------------------------
                    #with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    #logits = model(ids, attention_mask=attn).logits
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits = model(ids, attention_mask=attn).logits
                    logits_last = logits[torch.arange(ids.size(0)), last]
                    lp_good = torch.log_softmax(logits_last, -1).gather(-1, good.unsqueeze(-1)).squeeze(-1)
                    lp_bad  = torch.log_softmax(logits_last, -1).gather(-1, bad .unsqueeze(-1)).squeeze(-1)

                    
                    delta = lp_good - lp_bad

                    tot_delta += delta.sum().item()
                    wins      += (delta > 0).sum().item()

                    #rows.extend([{"delta": round(float(d), 6), "chosen_id": g.item(), "rejected_id": b.item()}
                    rows.extend([{"delta": round(d.item(), 6),  # or round(d.item(), 6)
                        "chosen_id": g.item(),
                        "rejected_id": b.item()}
                        for d, g, b in zip(delta, good, bad)])

            mean = tot_delta / len(dataset)
            acc  = wins / len(dataset)
            return rows, {"tag": tag, "mean_delta": mean, "chosen_win": acc, "n": len(dataset)}

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


        if False: # skip this check for now
            pre_train_rows, pre_train_stats = _gap_stats(model, train_ds, collate, "train_pre", ref_model=None, use_null_ref=True)
            pre_val_rows , pre_val_stats   = _gap_stats(model, val_ds,   collate, "val_pre", ref_model=None, use_null_ref=True)


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



    #freeze_early_layers(model, n_unfrozen = 4, verbose = True)


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
            disable_tqdm=False,
        ),
    )

    logger.info(f"Starting training. Output will be in {finetune_output_dir}. Check tensorboard for progress.")

    try:
        trainer_stats = dpo_trainer.train()
        logger.info("DPO training finished.")
        if hasattr(trainer_stats, 'metrics'):
            logger.info(f"Trainer metrics: {trainer_stats.metrics}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return
    


    if CALC_VAL_STATS:
        # ────────────────────────────────────────────────────────────────────
        # 3) POST-TRAIN statistics  (same API as above)
        # ────────────────────────────────────────────────────────────────────
        post_train_rows, post_train_stats = _gap_stats(model, train_ds, collate, "train_post", ref_model=None, use_null_ref=False)
        post_val_rows , post_val_stats   = _gap_stats(model, val_ds,   collate, "val_post", ref_model=None, use_null_ref=False)


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