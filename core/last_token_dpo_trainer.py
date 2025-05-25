from core.finetuning import DPOTrainer, torch, pad_sequence, F


# ---------------------------------------------------------------------
# Early-stopping on any logged scalar (loss, choice_win, etc.)
# ---------------------------------------------------------------------
from transformers.trainer_callback import TrainerCallback


class ThresholdStop(TrainerCallback):
    """
    Stop training immediately when `monitor` crosses `threshold`.

    If `higher_is_better` is True  → stop when metric >= threshold.  
    If False                       → stop when metric <= threshold.
    """
    def __init__(self, monitor: str, threshold: float, higher_is_better: bool):
        self.monitor           = monitor
        self.threshold         = threshold
        self.higher_is_better  = higher_is_better

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return
        value = logs[self.monitor]
        stop  = (value >= self.threshold) if self.higher_is_better else (value <= self.threshold)
        if stop:
            control.should_training_stop = True
            print(f"[ThresholdStop] {self.monitor}={value:.4f} "
                  f"crossed {'≥' if self.higher_is_better else '≤'} "
                  f"{self.threshold} – stopping.")
            
class EarlyStoppingByMetric(TrainerCallback):
    """
    Stop training when a monitored metric has stopped improving.

    Args
    ----
    monitor: str
        Key that appears in the `logs` dict (e.g. "loss", "choice_win").
    higher_is_better: bool
        True  → metric should increase (e.g. choice_win)  
        False → metric should decrease (e.g. loss / pref_loss)
    patience: int
        How many *log events* with no improvement to wait before stopping.
    min_delta: float
        Minimum change that counts as an improvement.
    """
    def __init__(self,
                 monitor: str,
                 higher_is_better: bool,
                 patience: int = 10,
                 min_delta: float = 0.0):
        self.monitor          = monitor
        self.higher_is_better = higher_is_better
        self.patience         = patience
        self.min_delta        = min_delta
        self.best             = None
        self.counter          = 0                     # events since last improv.

    # ── invoked every time trainer logs metrics ─────────────────────
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        # first observation
        if self.best is None:
            self.best = current
            return

        # compute signed improvement
        if self.higher_is_better:
            improvement = current - self.best
        else:
            improvement = self.best - current

        # has the metric improved “enough”?
        if improvement > self.min_delta:
            self.best    = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                #  signal the Trainer to halt
                control.should_training_stop = True
                print(f"[EarlyStopping] '{self.monitor}' plateaued "
                      f"(best={self.best:.5f}) – stopping training.")


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
        """
        Left-pads every prompt to `self.args.max_length`, so the last real
        token is always at position -1.  That lets the loss read logits
        with a single slice ([:, -1, :]).
        """
        pad_id   = self.padding_value
        max_len  = self.args.max_length
        batch_sz = len(features)

        # ── build [B, L] prompt tensor ───────────────────────────────
        prompt_ids   = torch.full((batch_sz, max_len), pad_id, dtype=torch.long)
        attention_ms = torch.zeros_like(prompt_ids, dtype=torch.bool)

        for i, feat in enumerate(features):
            seq = torch.tensor(feat["prompt_ids"], dtype=torch.long)
            if seq.size(0) > max_len:
                seq = seq[-max_len:]                    # truncate left if over-long
            prompt_ids[i, -seq.size(0):] = seq          # left-pad
            attention_ms[i, -seq.size(0):] = True

        # ── universal fields ─────────────────────────────────────────
        batch = dict(
            prompt_ids      = prompt_ids,
            attention_mask  = attention_ms,
            rejected_token_id = torch.tensor([f["rejected_token_id"] for f in features]),
        )

        # ── TDPO-MULTI vs single-token branch ────────────────────────
        if "chosen_ids" in features[0]:
            max_c = max(len(f["chosen_ids"]) for f in features)
            chosen_pad  = torch.full((batch_sz, max_c), -100, dtype=torch.long)
            chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
            for i, f in enumerate(features):
                ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
                chosen_pad [i, :ids.size(0)] = ids
                chosen_mask[i, :ids.size(0)] = True
            batch.update(chosen_ids = chosen_pad,
                        chosen_mask = chosen_mask)
        else:
            batch.update(chosen_token_id = torch.tensor([f["chosen_token_id"]
                                                        for f in features]))

        return batch


        

    # --------------------------------------------------------------------- #
    #  Allowed modes:
    #     "ref"   – reference-tethered preference loss
    #     "free"  – untethered (raw Δ)
    #     "clip"  – odds-ratio clipping with ε
    # --------------------------------------------------------------------- #
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        mode = getattr(self, "loss_mode", "clip")
        clip_epsilon_probs  = getattr(self, "clip_epsilon_probs", 0.2)
        clip_epsilon_logits  = getattr(self, "clip_epsilon_logits", 2)
        beta = getattr(self, "beta", 0.1)

        LOSS_CALC_MODE = getattr(self, "loss_calc_mode", "logits")    # probs / logits. Use probs or logits in the loss function. logits is more surgical (doesn't affect the whole logit distribution).
        USE_KL_LOSS=True # tether all the logits other than the ones we are interested in moving to the reference

        # ── unpack ---------------------------------------------------------
        ids   = inputs["prompt_ids"].to(model.device)      # [B,L]
        attn  = inputs["attention_mask"].to(model.device)  # [B,L]
        B, L  = ids.shape
        
        # Find the last real token position for each sequence
        seq_len  = attn.sum(1)                         # [B] - number of real tokens

        # Position encoding setup
        pad_off  = (L - seq_len).unsqueeze(1)
        arange_L = torch.arange(L, device=ids.device).unsqueeze(0)
        pos_full = (arange_L - pad_off).clamp(min=0)
        pos_full = pos_full.masked_fill(attn == 0, 0)

        # ------------- Single forward pass ---------------------------------
        outputs = model(
            ids,
            attention_mask=attn,
            position_ids=pos_full,
            use_cache=False,
            return_dict=True,
        )
        
        # With left padding, the last token is always at position -1
        logits_last = outputs.logits[:, -1, :]  # [B, V]
        logp_all = F.log_softmax(logits_last, dim=-1)  # [B, V]


        if inputs.get("chosen_ids") is not None: # tdpo-multi path
            # --- unpack ----------------------------------------------------------------
            ch_ids  = inputs["chosen_ids"].to(model.device)       # [B,C]
            ch_mask = inputs["chosen_mask"].to(model.device)      # [B,C] bool
            
            # --- rejected token ---------------------------------------------------------
            rej     = inputs["rejected_token_id"].to(model.device)  # [B]
            logp_bad = logp_all.gather(-1, rej.unsqueeze(-1)).squeeze(-1)  # [B]


            # ---------------------------------------------------------------------------
            #  Weight rule
            #     • margin = p(chosen) − p(rejected)
            #     • if margin ≥ ε                → weight = 0
            #     • if margin ≤ −ε or large loss → weight = 1
            #     • otherwise                    → linear taper  (ε − margin) / ε
            # ---------------------------------------------------------------------------
            batch_rows = torch.arange(B, device=logp_all.device).unsqueeze(1)
            
            if LOSS_CALC_MODE == "probs":
                gathered      = logp_all[batch_rows, ch_ids]      # log-p
                probs_good    = gathered.exp()
                prob_bad      = logp_bad.unsqueeze(-1).exp()
                margin        = probs_good - prob_bad                
                weights = torch.clamp((clip_epsilon_probs - margin) / clip_epsilon_probs, 0.0, 1.0) * ch_mask
            else:   # "logits"
                gathered      = logits_last[batch_rows, ch_ids]   # raw logits
                logit_bad     = logits_last.gather(-1, rej.unsqueeze(-1))
                margin        = gathered - logit_bad
                # need probs_good for the weighted mean later
                logZ          = logits_last.logsumexp(-1, keepdim=True)
                probs_good    = (gathered - logZ).exp()
                weights = torch.clamp((clip_epsilon_logits - margin) / clip_epsilon_logits, 0.0, 1.0) * ch_mask

            

            # fall-back: if *all* weights in a row are zero, keep every chosen token
            zero_row = weights.sum(dim=-1, keepdim=True) < 1e-12
            weights  = torch.where(zero_row, ch_mask.float(), weights)

            # ---------------------------------------------------------------------------
            #  Per-token preference loss (TDPO-MULTI) – honours "probs" vs "logits"
            # ---------------------------------------------------------------------------
            weights_sum = weights.sum(dim=-1, keepdim=True)               # [B,1]
            batch_rows  = torch.arange(B, device=ids.device).unsqueeze(1)

            if LOSS_CALC_MODE == "probs":
                logp_chosen = logp_all.gather(-1, ch_ids)                 # [B,C]
                logp_bad    = logp_all.gather(-1, rej.unsqueeze(-1))      # [B,1]
                delta_tok   = logp_chosen - logp_bad                      # [B,C]
            else:  # "logits" – completely localised, no softmax in loss
                l_chosen    = logits_last[batch_rows, ch_ids]             # [B,C]
                l_bad       = logits_last.gather(-1, rej.unsqueeze(-1))   # [B,1]
                delta_tok   = l_chosen - l_bad                            # [B,C]

            if mode == "clip":
                eps         = clip_epsilon_logits if LOSS_CALC_MODE == "logits" else clip_epsilon_probs
                ratio_tok   = torch.exp(delta_tok)
                clipped_tok = torch.minimum(ratio_tok,
                                            torch.tensor(1.0 + eps, device=ratio_tok.device))
                per_tok_loss = -torch.log(clipped_tok / (1.0 + eps))
            else:  # "free"
                per_tok_loss = -F.logsigmoid(beta * delta_tok)

            pref_loss = (per_tok_loss * weights).sum() / weights_sum.sum()   # ← scalar


        else:
            # single-token path
            chosen = inputs["chosen_token_id"].to(model.device)
            rejected = inputs["rejected_token_id"].to(model.device)
            logp_good = logp_all.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            logp_bad = logp_all.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)


        # ───────────────────────────────────────────────────────────────────
        #  Variant-specific preference term (single-token TDPO)
        # ───────────────────────────────────────────────────────────────────
        if inputs.get("chosen_ids") is None:
            if LOSS_CALC_MODE == "probs":
                delta = logp_good - logp_bad
            else:  # "logits"
                l_good = logits_last.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
                l_bad  = logits_last.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)
                delta  = l_good - l_bad

            if mode == "clip":
                eps    = clip_epsilon_logits if LOSS_CALC_MODE == "logits" else clip_epsilon_probs
                ratio  = torch.exp(delta)
                clipped = torch.minimum(ratio,
                                        torch.tensor(1.0 + eps, device=ratio.device))
                pref_loss = -torch.log(clipped / (1.0 + eps)).mean()
            else:  # "free"
                pref_loss = -F.logsigmoid(beta * delta).mean()



        # ── total loss & metrics ------------------------------------------
        if USE_KL_LOSS:
            # applies kl loss to every logit *except* those in chosen/rejected.
            # this allows our target logits to move freely while constraining weight updates globally
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_logits_last = model(
                            ids, attention_mask=attn, position_ids=pos_full,
                            use_cache=False, return_dict=True,
                        ).logits[:, -1, :]
                else:
                    ref_logits_last = self.ref_model(
                        ids, attention_mask=attn, position_ids=pos_full,
                        use_cache=False, return_dict=True,
                    ).logits[:, -1, :]

            freeze_mask = torch.ones_like(logits_last, dtype=torch.bool)
            if "chosen_ids" in inputs:
                freeze_mask.scatter_(1, ch_ids,  False)
            else:
                freeze_mask.scatter_(1, chosen.unsqueeze(-1), False)
            freeze_mask.scatter_(1, rej.unsqueeze(-1), False)

            if LOSS_CALC_MODE == "logits":
                # --- L2 penalty on raw logits (no softmax) ---------------------
                diff     = logits_last - ref_logits_last
                kl_part  = (freeze_mask * diff.pow(2)).sum()
                kl_loss  = kl_part / freeze_mask.sum()               # MSE on frozen logits
            else:
                # --- standard forward-KL in prob space ------------------------
                ref_logp_all = F.log_softmax(ref_logits_last, dim=-1)
                logp_all     = F.log_softmax(logits_last,      dim=-1)
                kl_part      = (freeze_mask * ref_logp_all.exp()
                                        * (ref_logp_all - logp_all)).sum()
                kl_loss      = kl_part / freeze_mask.sum()

            # ── extra diagnostics ─────────────────────────────────────────────
            freeze_frac    = freeze_mask.float().mean()                # how much of V is KL-regularised
            kl_pref_ratio  = kl_loss / (pref_loss + 1e-8)              # relative weight of KL vs pref

            # ── diagnostic ratios ─────────────────────────────────────────────
            lambda_kl           = 0.1                                        # <-- keep in sync with the line in `loss = ...`
            pref_eps            = 1e-4          # or smaller if you like
            safe_pref_loss      = torch.clamp(pref_loss.detach(), min=pref_eps)

            kl_pref_ratio_raw   = kl_loss / safe_pref_loss
            kl_pref_ratio       = lambda_kl * kl_pref_ratio_raw

            extra_metrics = {
                "kl_loss"            : kl_loss.detach(),        # un-scaled
                "freeze_frac"        : freeze_frac.detach(),
                "kl_pref_ratio"      : kl_pref_ratio.detach(),  # scaled (what actually matters)
            }

            loss = pref_loss + lambda_kl * kl_loss                           # λ≈0.02 usually enough

        else:
            loss = pref_loss

        if inputs.get("chosen_ids") is not None:        # TDPO-MULTI
            # log-probs for each chosen token, same shape as ch_ids
            lp_chosen = logp_all.gather(-1, ch_ids)               # [B,C]
            lp_bad    = logp_bad.unsqueeze(-1)                    # [B,1]

            # boolean wins per token (requires same mask as ch_mask)
            wins_tok  = (lp_chosen > lp_bad) & ch_mask            # [B,C] bool
            frac_win  = wins_tok.float().sum(-1) / ch_mask.sum(-1).clamp(min=1e-8)  # [B]
            choice_win = frac_win.mean().detach()                 # scalar
        else:                                        # TDPO single
            choice_win = (logp_good > logp_bad).float().mean().detach()        

        metrics = {
            "pref_loss":  pref_loss.detach(),
            "choice_win": choice_win,
            **extra_metrics,
        }
        self.store_metrics(metrics, train_eval="train")

        #self.log({f"train/{k}": (v if not torch.is_tensor(v) else v.cpu().float().item())
        #  for k, v in extra_metrics.items()})

        if return_outputs:
            return loss, metrics
        return loss



    # ----------------------------------------------------------
    def _prepare_dataset(self, dataset, *args, **_):
        return dataset