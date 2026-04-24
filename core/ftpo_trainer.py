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
# Early-stopping on any logged scalar (loss, chosen_win, etc.)
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
        Key that appears in the `logs` dict (e.g. "loss", "chosen_win").
    higher_is_better: bool
        True  → metric should increase (e.g. chosen_win)  
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



class FTPOTrainer(DPOTrainer):
    """
    Trainer for final token preference optimisation (ftpo).
    Replaces TRL’s standard loss with a log-ratio on the **last**
    autoregressive position.
    """

    # ──────────────────────────────────────────────────────────────
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_unused_columns = False
        self.data_collator = self.ftpo_collator                    # override

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
    def ftpo_collator(self, features):
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

        # ── ftpo vs single-token branch ────────────────────────
        max_c = max(len(f["chosen_ids"]) for f in features)
        chosen_pad = torch.full((batch_sz, max_c), pad_id, dtype=torch.long)
        chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
        for i, f in enumerate(features):
            ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
            chosen_pad [i, :ids.size(0)] = ids
            chosen_mask[i, :ids.size(0)] = True
        batch.update(chosen_ids = chosen_pad,
                    chosen_mask = chosen_mask)

        return batch

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        # We use 2 separate MSE loss terms
        # 1. A lightly applied tokenwise MSE loss applied to only the target tokens
        lambda_mse_target   = getattr(self, "lambda_mse_target", 0.05)  # strength
        tau_mse_target      = getattr(self, "tau_mse_target", 1.0)      # grace region (zero cost movement)

        # 2. A strongly applied tokenwise MSE loss applied to the remaining (non-target) vocab
        lambda_mse = getattr(self, "lambda_mse", 0.4) # how strongly the remaining vocab (other than chosen/rejected) is tethered to reference via mse loss

        # loss contribution is clipped if (chosen - rejected) logits delta is above this
        clip_epsilon_logits  = getattr(self, "clip_epsilon_logits", 2) 

        USE_MSE_LOSS=True # tether all the logits other than the ones we are interested in moving to the reference

        # ── unpack ---------------------------------------------------------
        device   = next(model.parameters()).device            # works for DP / DDP
        ids   = inputs["prompt_ids"].to(device)               # [B,L]
        attn  = inputs["attention_mask"].to(device)           # [B,L]
        B, L  = ids.shape
        
        seq_len  = attn.sum(1)
        pad_off  = (L - seq_len).unsqueeze(1)
        arange_L = torch.arange(L, device=ids.device).unsqueeze(0)
        pos_full = (arange_L - pad_off).clamp(min=0)
        pos_full = pos_full.masked_fill(attn == 0, 0)

        outputs = model(
            ids,
            attention_mask=attn,
            position_ids=pos_full,
            use_cache=False,
            return_dict=True,
        )
        
        logits_last = outputs.logits[:, -1, :]  # [B, V]
        logp_all = F.log_softmax(logits_last, dim=-1)  # [B, V]

        ch_ids  = inputs["chosen_ids"].to(device)       
        ch_mask = inputs["chosen_mask"].to(device)      
        rejected     = inputs["rejected_token_id"].to(device)  
        logp_bad = logp_all.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)  

        batch_rows = torch.arange(B, device=logp_all.device).unsqueeze(1)
        delta_tok = logits_last[batch_rows, ch_ids] - logits_last.gather(
            -1, rejected.unsqueeze(-1)
        )
        weights = (
            torch.clamp(
                (clip_epsilon_logits - delta_tok) / clip_epsilon_logits,
                0.0,
                1.0,
            )
            * ch_mask
        )

        tau = 1.0
        gap = clip_epsilon_logits - delta_tok
        per_tok_loss = F.softplus(gap / tau)

        chosen_counts = ch_mask.sum(dim=-1).clamp(min=1)
        pref_loss = ((per_tok_loss * weights).sum(dim=-1) / chosen_counts).mean()

        extra_metrics = {}

        if USE_MSE_LOSS:
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

            tether_mask = torch.ones_like(logits_last, dtype=torch.bool)
            rows = torch.arange(B, device=ch_ids.device).unsqueeze(1).expand_as(ch_ids)
            tether_mask[rows[ch_mask], ch_ids[ch_mask]] = False
            tether_mask.scatter_(1, rejected.unsqueeze(-1), False)

            diff        = logits_last - ref_logits_last
            mse_elem_raw = (tether_mask * diff.pow(2)).sum() / tether_mask.sum()

            tgt_mask = torch.zeros_like(logits_last, dtype=torch.bool)
            rows = torch.arange(B, device=ch_ids.device).unsqueeze(1).expand_as(ch_ids)
            tgt_mask[rows[ch_mask], ch_ids[ch_mask]] = True
            tgt_mask.scatter_(1, rejected.unsqueeze(-1), True)
            
            if lambda_mse_target:
                diff_tok = logits_last - ref_logits_last          
                diff_tok = diff_tok * tgt_mask
                excess_tok = torch.clamp(diff_tok.abs() - tau_mse_target, min=0.0)
                mse_target_raw = (excess_tok.pow(2)).sum() / tgt_mask.sum()
            else:
                mse_target_raw = logits_last.new_tensor(0.0)

            mse_loss = (
                lambda_mse         * mse_elem_raw   
                + lambda_mse_target   * mse_target_raw
            )
            loss    = pref_loss + mse_loss

            extra_metrics.update({
                "mse_elem"        : mse_elem_raw.detach(),
                "mse_tgt_tokenwise"         : mse_target_raw.detach(),
            })

        else:
            loss = pref_loss

        lp_chosen = logp_all.gather(-1, ch_ids)               
        lp_bad    = logp_bad.unsqueeze(-1)                    

        wins_tok  = (lp_chosen > lp_bad) & ch_mask            
        frac_win  = wins_tok.float().sum(-1) / ch_mask.sum(-1).clamp(min=1e-8)  
        chosen_win = frac_win.mean().detach()                 

        active_delta = delta_tok[ch_mask]
        active_weights = weights[ch_mask]
        margin_win = (
            ((delta_tok >= clip_epsilon_logits) & ch_mask).float().sum()
            / ch_mask.float().sum().clamp(min=1e-8)
        ).detach()
        mean_delta = active_delta.mean().detach()
        median_delta = active_delta.median().detach()
        active_weight = active_weights.mean().detach()

        metrics = {
            "pref_loss":  pref_loss.detach(),
            "chosen_win": chosen_win,
            "margin_win": margin_win,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "active_weight": active_weight,
            **extra_metrics,
        }
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss




    # ----------------------------------------------------------
    def _prepare_dataset(self, dataset, *args, **_):
        return dataset
