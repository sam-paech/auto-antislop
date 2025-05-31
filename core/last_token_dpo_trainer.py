from core.finetuning import DPOTrainer, torch, pad_sequence, F

# ---------------------------------------------------------------
#  Add Adaptive Gradient Clipping to *every* parameter in-place
# ---------------------------------------------------------------
def attach_agc(model, clip: float = 0.01, eps: float = 1e-3):
    """
    Registers a per-parameter hook that applies Brock et al.’s
    Adaptive Gradient Clipping:

        ||g||₂ > clip * (||θ||₂ + eps)  →  g ← g * (threshold / ||g||₂)

    Works with params in fp32, bf16, or bitsandbytes int4.
    """

    def _agc_hook(grad, param):
        #print('agc hook')
        if grad is None:
            return grad
        param_norm = param.detach().norm()            # ||θ||
        grad_norm  = grad.norm()                      # ||g||
        max_norm   = clip * (param_norm + eps)
        if grad_norm > max_norm:
            grad = grad * (max_norm / (grad_norm + 1e-6))
        return grad

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g, p=p: _agc_hook(g, p))


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
        max_c = max(len(f["chosen_ids"]) for f in features)
        chosen_pad  = torch.full((batch_sz, max_c), -100, dtype=torch.long)
        chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
        for i, f in enumerate(features):
            ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
            chosen_pad [i, :ids.size(0)] = ids
            chosen_mask[i, :ids.size(0)] = True
        batch.update(chosen_ids = chosen_pad,
                    chosen_mask = chosen_mask)

        return batch


        

    # --------------------------------------------------------------------- #
    #  Allowed modes:
    #     "free"  – untethered (raw Δ)
    #     "clip"  – odds-ratio clipping with ε
    # --------------------------------------------------------------------- #
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        mode = getattr(self, "loss_mode", "clip") # "clip" | "free": determines whether we clip loss contributions per the ratio of rejected/chosen logits (like orpo)        
        beta = getattr(self, "beta", 0.1) # loss scaling
        lambda_kl_target   = getattr(self, "lambda_kl_target", 0.1) # how strongly target (chosen/rejected) logits are tethered to reference via kl loss
        tau_kl_target = getattr(self, "tau_kl_target", 2.0)  # allow the target logits some grace to move before kl loss kicks in
        lambda_kl = getattr(self, "lambda_kl", 0.4) # how strongly the remaining vocab (other than chosen/rejected) is tethered to reference via kl loss

        LOSS_CALC_MODE = getattr(self, "loss_calc_mode", "logits")    # probs / logits. Use probs or logits in the loss function. logits is more surgical (doesn't affect the whole logit distribution).
        clip_epsilon_probs  = getattr(self, "clip_epsilon_probs", 0.2) # (when in probs mode): loss contribution is clipped if (chosen - rejected) probs delta is above this
        clip_epsilon_logits  = getattr(self, "clip_epsilon_logits", 2) # (when in logits mode): loss contribution is clipped if (chosen - rejected) logits delta is above this

        USE_KL_LOSS=True # tether all the logits other than the ones we are interested in moving to the reference

        # ── unpack ---------------------------------------------------------
        device   = next(model.parameters()).device            # works for DP / DDP
        ids   = inputs["prompt_ids"].to(device)               # [B,L]
        attn  = inputs["attention_mask"].to(device)           # [B,L]
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
            ch_ids  = inputs["chosen_ids"].to(device)       # [B,C]
            ch_mask = inputs["chosen_mask"].to(device)      # [B,C] bool
            
            # --- rejected token ---------------------------------------------------------
            rej     = inputs["rejected_token_id"].to(device)  # [B]
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
            chosen = inputs["chosen_token_id"].to(device)
            rejected = inputs["rejected_token_id"].to(device)
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


        extra_metrics = {}

        # ── total loss & metrics ------------------------------------------
        if USE_KL_LOSS:
            # --------------------------------------------------------------
            # 1. reference logits (no-grad)
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # 2. element-wise KL on *non-target* vocab  (old behaviour)
            # --------------------------------------------------------------
            freeze_mask = torch.ones_like(logits_last, dtype=torch.bool)
            if "chosen_ids" in inputs:                 # TDPO-MULTI
                freeze_mask.scatter_(1, ch_ids,  False)
            else:                                      # single-token
                freeze_mask.scatter_(1, chosen.unsqueeze(-1), False)
            freeze_mask.scatter_(1, rej.unsqueeze(-1), False)

            if LOSS_CALC_MODE == "logits":
                diff        = logits_last - ref_logits_last
                kl_elem_raw = (freeze_mask * diff.pow(2)).sum() / freeze_mask.sum()
            else:
                ref_lp      = F.log_softmax(ref_logits_last, dim=-1)
                cur_lp      = F.log_softmax(logits_last,      dim=-1)
                kl_elem_raw = (freeze_mask * ref_lp.exp() * (ref_lp - cur_lp)).sum()
                kl_elem_raw = kl_elem_raw / freeze_mask.sum()

            # --------------------------------------------------------------
            # 3. *Optional* aggregate KL on the target tokens only
            #       This pulls target token logits towards basline
            #       but only *on aggregate* so they can still move
            #       independently of one another.
            # --------------------------------------------------------------            

            if lambda_kl_target:
                tgt_mask = torch.zeros_like(logits_last, dtype=torch.bool)
                if "chosen_ids" in inputs:
                    tgt_mask.scatter_(1, ch_ids, True)
                else:
                    tgt_mask.scatter_(1, chosen.unsqueeze(-1), True)
                tgt_mask.scatter_(1, rej.unsqueeze(-1), True)

                tgt_sz      = tgt_mask.sum(-1).clamp(min=1)        # avoid /0
                mean_cur    = (logits_last * tgt_mask).sum(-1) / tgt_sz
                mean_ref    = (ref_logits_last * tgt_mask).sum(-1) / tgt_sz

                # give the target logits some grace to move around before kl penalty kicks in                
                diff = mean_cur - mean_ref                      # [B]
                excess = torch.clamp(diff.abs() - tau_kl_target, min=0.0) # zero inside ±tau
                kl_tgt_raw = excess.pow(2).mean()               # quadratic outside band

            else:
                kl_tgt_raw  = logits_last.new_tensor(0.0)

            # --------------------------------------------------------------
            # 4. combine
            # --------------------------------------------------------------
            kl_loss = lambda_kl * kl_elem_raw + lambda_kl_target * kl_tgt_raw
            loss    = pref_loss + kl_loss

            # diagnostics
            extra_metrics.update({
                "kl_elem"        : kl_elem_raw.detach(),
                "kl_tgt"         : kl_tgt_raw.detach(),
                "kl_pref_ratio"  : (kl_loss / (pref_loss + 1e-8)).detach(),
            })

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
    

class AGCTrainer(LastTokenDPOTrainer):
    def __init__(self, *args, agc_clip: float = 0.01, agc_eps: float = 1e-3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.agc_clip = agc_clip
        self.agc_eps  = agc_eps

    # HF calls this right after .backward() and before optimizer.step()
    def clip_gradients(self, accelerator, model, max_grad_norm=None):
        print('clipping grads')
        clip = self.agc_clip
        eps  = self.agc_eps

        for p in model.parameters():
            if p.grad is None:
                continue
            # ||θ|| and ||g||
            param_norm = p.detach().norm()
            grad_norm  = p.grad.norm()

            max_norm = clip * (param_norm + eps)
            if grad_norm > max_norm:
                p.grad.mul_(max_norm / (grad_norm + 1e-6))