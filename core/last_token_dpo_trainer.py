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
        mode = getattr(self, "loss_mode", "clip")   # set externally
        eps  = getattr(self, "clip_eps", 0.2)       # only used in "clip"
        beta = getattr(self, "beta", 0.1)           # reused everywhere

        # ── unpack ---------------------------------------------------------
        ids   = inputs["prompt_ids"].to(model.device)      # [B,L]
        attn  = inputs["attention_mask"].to(model.device)  # [B,L]
        B, L  = ids.shape
        pad_id = self.padding_value

        # -------- logical position indices (ignore pads) ------------------
        seq_len  = attn.sum(1)                         # [B]  (number of actual tokens, n)
        # Assuming seq_len >= 1. If seq_len can be 0, this needs careful handling.
        # For now, proceed with the assumption from typical tokenized inputs.

        # `pos_full` calculates the 0-indexed position for each token in the sequence.
        # Pads are 0, first actual token is 0, second is 1, ..., last actual token is (seq_len-1).
        # This calculation seems correct as is:
        pad_off  = (L - seq_len).unsqueeze(1)          # [B,1]
        arange_L = torch.arange(L, device=ids.device).unsqueeze(0)  # [1,L]
        pos_full = (arange_L - pad_off).clamp(min=0)   # 0 … n-1, pads map to 0
        pos_full = pos_full.masked_fill(attn == 0, 0)  # Ensure pads are definitively 0

        # --- Corrected logic for two-pass ---
        
        # The actual last token of the prompt is always at column L-1 in the `ids` tensor
        # due to the left-padding in `tdpo_collator`.
        actual_last_token_col_idx = L - 1 # Scalar: column index

        # ------------- pass 1 : context (no grad) --------------------------
        # Goal: Get past_key_values for the prompt *excluding* its very last token.
        
        ctx_ids  = ids.clone()
        # Mask out the actual last token in the tensor by replacing it with pad_id
        ctx_ids[:, actual_last_token_col_idx] = pad_id
        
        ctx_attn = attn.clone()
        # Zero out attention for this masked token
        ctx_attn[:, actual_last_token_col_idx] = 0

        pos_ctx = pos_full.clone()
        # Set the position_id of this masked token to 0 (or your designated pad position_id)
        pos_ctx[:, actual_last_token_col_idx] = 0

        with torch.no_grad():
            ctx_out = model(
                ctx_ids,
                attention_mask = ctx_attn,
                position_ids   = pos_ctx,
                use_cache      = True, # We need past_key_values
                return_dict    = True,
            )
        past_kv = ctx_out.past_key_values

        # ------------- pass 2 : last prompt token (grad) -------------------
        # Goal: Get logits for the token *after* the actual last prompt token, using past_kv.

        # The input token for this pass is the actual last token of the original prompt.
        # It's located at ids[:, actual_last_token_col_idx].
        # Its shape should be [B, 1].
        tok_ids_for_pass2 = ids[:, actual_last_token_col_idx].unsqueeze(1)

        # Attention mask for this single token. Since it's an actual token (assuming seq_len >= 1),
        # its original attention mask (attn[:, actual_last_token_col_idx]) was 1.
        # Shape: [B, 1].
        tok_attn_for_pass2 = attn[:, actual_last_token_col_idx].unsqueeze(1)
        # Alternative, if absolutely sure it's always a non-pad token:
        # tok_attn_for_pass2 = torch.ones_like(tok_ids_for_pass2, dtype=attn.dtype)


        # The position_id for this token. The original position of the last token is (seq_len - 1).
        # seq_len is [B], so (seq_len - 1) is [B]. Unsqueeze to [B, 1].
        # Shape: [B, 1].
        pos_tok_for_pass2 = (seq_len - 1).unsqueeze(1)
        # Ensure pos_tok_for_pass2 is not negative if seq_len can be 0.
        # If seq_len is guaranteed >= 1, this is fine. Otherwise, .clamp(min=0) might be needed.
        # pos_tok_for_pass2 = (seq_len - 1).clamp(min=0).unsqueeze(1) # Safer if seq_len can be 0

        tok_out = model(
            tok_ids_for_pass2,         # Correct token
            attention_mask = tok_attn_for_pass2, # Correct attention for this token
            position_ids   = pos_tok_for_pass2,  # Correct position for this token
            past_key_values= past_kv,
            use_cache      = False, # We don't need past_kv from this pass
            return_dict    = True,
        )

        # Logits for the next token prediction, after the last prompt token.
        # tok_out.logits is [B, 1, V], so squeeze out the dim 1.
        logp_all = F.log_softmax(tok_out.logits.squeeze(1), dim=-1)   # [B,V]

        # ───────────────────────── DEBUG BLOCK ────────────────────────────
        # (Keep your debug block as is, it should now show much smaller |Δ log-p|)
        DEBUG = True
        if DEBUG and not getattr(self, "_debug_ran", False):
            self._debug_ran = True

            assert tok_ids_for_pass2.shape[1] == 1, f"tok_ids_for_pass2 shape {tok_ids_for_pass2.shape}"
            assert logp_all.dim()   == 2,  f"logp_all dim {logp_all.dim()}"

            with torch.no_grad():
                one_logits = model(
                    ids,
                    attention_mask = attn,
                    position_ids   = pos_full,
                    use_cache      = False,
                    return_dict    = True,
                ).logits[:, -1, :] # Logits at the last position (L-1), predicting token after ids[:, L-1]
                one_logp = F.log_softmax(one_logits, -1)

            max_abs_diff = (one_logp - logp_all).abs().max().item()
            print(f"[DBG] max |Δ log-p| two-pass vs one-pass : {max_abs_diff:.3e}")

            batch_rows = torch.arange(ids.size(0), device=ids.device)
            if "chosen_ids" in inputs:
                ch_ids  = inputs["chosen_ids" ].to(ids.device)
                ch_mask = inputs["chosen_mask"].to(ids.device)
                rej     = inputs["rejected_token_id"].to(ids.device)
                probe_good = one_logp[batch_rows.unsqueeze(1), ch_ids]
                probe_bad  = one_logp[batch_rows, rej].unsqueeze(1)
                wins_probe = (probe_good > probe_bad) & ch_mask
                probe_win  = (wins_probe.float().sum(-1) / ch_mask.sum(-1).clamp(min=1e-8)).mean().item() # Added clamp for safety
                int_good = logp_all[batch_rows.unsqueeze(1), ch_ids]
                int_bad  = logp_all[batch_rows, rej].unsqueeze(1)
                wins_int = (int_good > int_bad) & ch_mask
                int_win  = (wins_int.float().sum(-1) / ch_mask.sum(-1).clamp(min=1e-8)).mean().item() # Added clamp for safety
                same_id = ((ch_ids == rej.unsqueeze(1)) & ch_mask).sum().item()
            else:
                ch_ids = inputs["chosen_token_id"].to(ids.device)
                rej    = inputs["rejected_token_id"].to(ids.device)
                probe_good = one_logp[batch_rows, ch_ids]
                probe_bad  = one_logp[batch_rows, rej]
                probe_win  = (probe_good > probe_bad).float().mean().item()
                int_good = logp_all[batch_rows, ch_ids]
                int_bad  = logp_all[batch_rows, rej]
                int_win  = (int_good > int_bad).float().mean().item()
                same_id = (ch_ids == rej).sum().item()
            print(f"[DBG] probe  win (one-pass) = {probe_win:.4f}")
            print(f"[DBG] internal win (two-pass)= {int_win:.4f}")
            if same_id: print(f"[WARN] {same_id} examples where chosen == rejected!")
            raise RuntimeError("Debug run complete – stopping trainer.")
        # ────────────────────── END DEBUG BLOCK ───────────────────────────

        # ... (rest of your loss calculation logic using the corrected logp_all) ...
        # Make sure to use the correct `rejected_token_id` from inputs, not a local `rejected`
        # variable if it was defined differently before.
        # The original code snippet had `rejected = inputs["rejected_token_id"].to(model.device)`
        # This should be fine.

        # Example:
        # if "chosen_ids" in inputs: # TDPO-MULTI
        #     ch_ids  = inputs["chosen_ids"].to(model.device)
        #     ch_mask = inputs["chosen_mask"].to(model.device)
        #     rej_ids = inputs["rejected_token_id"].to(model.device) # Ensure this is used

        #     # ... calculations for logp_good using logp_all and ch_ids ...
        #     # logp_bad = logp_all.gather(-1, rej_ids.unsqueeze(-1)).squeeze(-1)
        # else: # single-token
        #     chosen_ids = inputs["chosen_token_id"].to(model.device)
        #     rej_ids    = inputs["rejected_token_id"].to(model.device)
        #     # logp_good = logp_all.gather(-1, chosen_ids.unsqueeze(-1)).squeeze(-1)
        #     # logp_bad  = logp_all.gather(-1, rej_ids.unsqueeze(-1)).squeeze(-1)

        # The rest of your code for calculating logp_good, logp_bad, delta, pref_loss, etc.
        # should now work with a `logp_all` that is consistent with a single pass.
        # You'll need to adapt the parts where `logp_good` and `logp_bad` are derived
        # from `logp_all` based on whether it's the multi-token or single-token case.

        # --- Corrected logp_good / logp_bad extraction ---
        rejected_token_ids = inputs["rejected_token_id"].to(model.device) # [B]

        if "chosen_ids" in inputs: # TDPO-MULTI
            # This part of your code for TDPO-MULTI seems to calculate logp_good
            # based on a weighted average of probabilities of chosen tokens.
            # It should use the corrected `logp_all`.
            ch_ids_multi  = inputs["chosen_ids"].to(model.device)       # [B,C]
            ch_mask_multi = inputs["chosen_mask"].to(model.device)      # [B,C] bool

            batch_rows_idx = torch.arange(B, device=logp_all.device).unsqueeze(1) # [B,1] for gathering
            
            # Log-probabilities of all chosen tokens according to the policy model
            gathered_logps_chosen = logp_all[batch_rows_idx, ch_ids_multi] # [B, C]
            probs_chosen = gathered_logps_chosen.exp()                     # p_i for chosen tokens

            # Your weighting logic for TDPO-MULTI
            # Ensure `eps` here is the one for weighting, not the DPO clipping `eps`
            # Assuming `eps` for weighting is a small probability threshold like 0.01 or similar,
            # not the DPO clipping epsilon (which is often 0.1-0.2 for odds ratio).
            # Let's call it `prob_weighting_eps` if it's different.
            # For now, I'll assume `eps` from `getattr(self, "clip_eps", 0.2)` is NOT for this.
            # Let's assume a fixed small epsilon for this weighting or make it configurable.
            # For simplicity, I'll use a placeholder value or assume it's defined elsewhere.
            # If `eps` from `clip_eps` was intended, then it's fine.
            prob_weighting_eps = 0.01 # Example, adjust as needed or make configurable

            weights  = torch.clamp((prob_weighting_eps - probs_chosen) / prob_weighting_eps, min=0.0) * ch_mask_multi
            zero_row = weights.sum(dim=-1, keepdim=True) < 1e-12
            weights  = torch.where(zero_row, ch_mask_multi.float(), weights) # Use mask if all weights zero

            weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8) # Avoid div by zero
            weighted_mean_prob = (probs_chosen * weights).sum(dim=-1, keepdim=True) / weights_sum
            logp_good = weighted_mean_prob.log().squeeze(-1)      # [B]

            logp_bad = logp_all.gather(-1, rejected_token_ids.unsqueeze(-1)).squeeze(-1)  # [B]

        else: # single-token path
            chosen_token_ids = inputs["chosen_token_id"].to(model.device) # [B]
            logp_good = logp_all.gather(-1, chosen_token_ids.unsqueeze(-1)).squeeze(-1) # [B]
            logp_bad = logp_all.gather(-1, rejected_token_ids.unsqueeze(-1)).squeeze(-1)  # [B]


        # ───────────────────────────────────────────────────────────────────
        #  Variant-specific preference term (DPO modes: ref, clip, free)
        # ───────────────────────────────────────────────────────────────────
        if mode == "ref":
            # Reference model pass needs to be consistent for chosen/rejected logp calculation
            with torch.no_grad():
                # Determine ref_model or use self.null_ref_context()
                current_model_for_ref = self.ref_model if self.ref_model is not None else model
                with self.null_ref_context() if self.ref_model is None else torch.no_grad(): # Ensure no_grad for self.model if it's the ref
                    ref_out = current_model_for_ref(
                        ids, # Full original prompt
                        attention_mask=attn,
                        position_ids=pos_full, # Use consistent position_ids
                        use_cache=False,
                        return_dict=True,
                    )

                ref_logits_last = ref_out.logits[:, -1, :] # Logits at the last position
                ref_logp_all    = F.log_softmax(ref_logits_last, dim=-1)

                if "chosen_ids" in inputs: # TDPO-MULTI for ref model
                    # Consistent logp_good calculation for ref model
                    ref_gathered_logps_chosen = ref_logp_all[batch_rows_idx, ch_ids_multi] # [B, C]
                    ref_probs_chosen = ref_gathered_logps_chosen.exp()
                    
                    ref_weights  = torch.clamp((prob_weighting_eps - ref_probs_chosen) / prob_weighting_eps, min=0.0) * ch_mask_multi
                    ref_zero_row = ref_weights.sum(dim=-1, keepdim=True) < 1e-12
                    ref_weights  = torch.where(ref_zero_row, ch_mask_multi.float(), ref_weights)
                    
                    ref_weights_sum = ref_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    ref_weighted_mean_prob = (ref_probs_chosen * ref_weights).sum(dim=-1, keepdim=True) / ref_weights_sum
                    ref_good = ref_weighted_mean_prob.log().squeeze(-1) # [B]
                else: # single-token for ref model
                    ref_good = ref_logp_all.gather(-1, chosen_token_ids.unsqueeze(-1)).squeeze(-1) # [B]
                
                ref_bad = ref_logp_all.gather(-1, rejected_token_ids.unsqueeze(-1)).squeeze(-1) # [B]

            delta     = (logp_good - ref_good) - (logp_bad - ref_bad)
            pref_loss = -F.logsigmoid(beta * delta).mean()
            kl_loss   = 0.0 # Placeholder, KL not implemented here

        elif mode == "clip":
            delta     = logp_good - logp_bad
            ratio     = torch.exp(delta)
            # `eps` here is the DPO clipping epsilon, e.g., 0.2
            clipped_ratio = torch.clip(ratio, 1.0 - eps, 1.0 + eps) # Common PPO clipping
            # The loss formulation in your code was: -torch.log(clipped / (1.0 + eps)).mean()
            # A more standard DPO-like clipping or PPO loss:
            # loss1 = ratio * (beta * delta) # or just ratio * delta if beta is in delta
            # loss2 = clipped_ratio * (beta * delta)
            # pref_loss = -torch.min(loss1, loss2).mean() # This is for advantage * ratio
            # For DPO, often it's simpler:
            # log_ratio = beta * delta
            # clipped_log_ratio = torch.clip(log_ratio, -eps_clip_log, eps_clip_log) # if eps is for log_ratio
            # pref_loss = -F.logsigmoid(clipped_log_ratio).mean()
            # Given your original: -torch.log(clipped / (1.0 + eps)).mean()
            # This seems unusual. If `ratio` is preferred (chosen prob / rejected prob),
            # then `log_odds = logp_good - logp_bad`.
            # `pi_ratio = exp(log_odds_policy - log_odds_ref)`
            # `loss = -log_sigmoid(beta * (log_odds_policy - log_odds_ref))`
            # For "clip" mode without ref, it's often on `log_odds_policy` itself.
            # Let's stick to your original formulation for "clip" for now, assuming it's intended:
            # delta = logp_good - logp_bad (log-odds of policy)
            # ratio = torch.exp(delta)     (odds ratio of policy)
            # clipped = torch.minimum(ratio, torch.tensor(1.0 + eps, device=ratio.device)) # This clips if ratio > 1+eps
            # pref_loss = -torch.log(clipped / (1.0 + eps)).mean() # This term is 0 if ratio <= 1+eps, negative if ratio > 1+eps (undesirable)
                                                                # Or if ratio < 1+eps, log(...) is negative, so -log(...) is positive.
            # This loss seems to penalize when `clipped / (1.0 + eps)` is small, i.e. when `ratio` is small.
            # This is more like -log(min(ratio, 1+eps)/(1+eps)).
            # If ratio is high, say 2, eps=0.2. min(2, 1.2) = 1.2. 1.2/1.2=1. log(1)=0.
            # If ratio is low, say 0.5. min(0.5, 1.2) = 0.5. 0.5/1.2 = 0.41. log(0.41) = -0.87. -log = 0.87.
            # This seems to be a valid loss that encourages higher ratios, capped.
            pref_loss = -torch.log(torch.minimum(ratio, 1.0 + eps) / (1.0 + eps) + 1e-8).mean() # Added 1e-8 for stability
            kl_loss   = 0.0

        else:  # "free" mode (standard DPO-like loss without ref model, on policy log_odds)
            delta = logp_good - logp_bad
            pref_loss = -F.logsigmoid(beta * delta).mean()
            kl_loss   = 0.0

        loss = pref_loss + kl_loss

        # Metrics calculation
        with torch.no_grad():
            if "chosen_ids" in inputs: # TDPO-MULTI
                # Log-probs for each chosen token under the policy
                lp_chosen_policy = logp_all.gather(-1, ch_ids_multi) # [B,C]
                # Log-prob for the rejected token (needs to be broadcastable for comparison)
                lp_bad_policy = logp_bad.unsqueeze(-1) # [B,1]
                
                wins_tok  = (lp_chosen_policy > lp_bad_policy) & ch_mask_multi # [B,C] bool
                # Average win rate per example (fraction of chosen tokens better than rejected)
                frac_win_per_example  = wins_tok.float().sum(-1) / ch_mask_multi.sum(-1).clamp(min=1e-8) # [B]
                choice_win = frac_win_per_example.mean() # scalar
            else: # TDPO single
                choice_win = (logp_good > logp_bad).float().mean()

        metrics = {
            "pref_loss":  pref_loss.detach(),
            "choice_win": choice_win.detach(), # Ensure detached
        }
        if mode == "clip": metrics["debug_ratio_mean"] = ratio.mean().detach()
        if mode == "ref" or mode == "free": metrics["debug_delta_mean"] = delta.mean().detach()
            
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss



    # ----------------------------------------------------------
    def _prepare_dataset(self, dataset, *args, **_):
        return dataset