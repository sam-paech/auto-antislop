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
        eps  = getattr(self, "clip_eps", 0.2)
        beta = getattr(self, "beta", 0.1)

        # ── unpack ---------------------------------------------------------
        ids   = inputs["prompt_ids"].to(model.device)      # [B,L]
        attn  = inputs["attention_mask"].to(model.device)  # [B,L]
        B, L  = ids.shape
        
        # Find the last real token position for each sequence
        seq_len  = attn.sum(1)                         # [B] - number of real tokens
        last_idx = seq_len - 1                         # [B] - last position index


        # Add this debug right after getting ids and attn in compute_loss:
        with torch.no_grad():
            if not getattr(self, "_debug_tails", False):
                self._debug_tails = True
                print("\n[DEBUG] Checking last tokens of all prompts in batch:")
                print(f"Batch size: {B}, Padded length: {L}")
                print(f"Padding token ID: {self.padding_value}")
                
                for i in range(min(10, B)):  # Look at up to 10 examples
                    seq_len_i = seq_len[i].item()
                    last_idx_i = last_idx[i].item()
                    
                    # Get the last 20 tokens (or however many are available)
                    num_to_show = min(20, seq_len_i)
                    start_pos = L - seq_len_i  # Where real tokens start (after left padding)
                    end_pos = L  # Where sequence ends
                    
                    # Extract the actual tokens
                    last_tokens = ids[i, end_pos - num_to_show:end_pos].tolist()
                    
                    print(f"\nExample {i}:")
                    print(f"  Seq length: {seq_len_i}, Last idx: {last_idx_i}")
                    print(f"  Last {num_to_show} tokens: {last_tokens}")
                    
                    # Decode them if possible
                    if hasattr(self, 'tokenizer'):
                        decoded = self.tokenizer.decode(last_tokens)
                        print(f"  Decoded: '{decoded}'")
                        
                        # Also decode just the last token
                        # Also decode just the last token (which is at position -1 with left padding)
                        actual_last_token_id = ids[i, -1].item()
                        last_token_decoded = self.tokenizer.decode([actual_last_token_id])
                        print(f"  Last token only: ID={actual_last_token_id} -> '{last_token_decoded}'")
                        

        
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



        # ───────────────────────── DEBUG BLOCK ────────────────────────────
        # This block validates that our training loss computation matches validation
        DEBUG = True
        if DEBUG:
            # Add right after computing logp_all:
            with torch.no_grad():
                if not getattr(self, "_debug_check", False):
                    self._debug_check = True
                    print("\n[DEBUG] Checking logit extraction:")
                    print(f"Batch size: {B}, Padded length: {L}")
                    
                    for i in range(min(3, B)):
                        print(f"\nExample {i}:")
                        print(f"  Sequence length: {seq_len[i].item()}")
                        print(f"  Last index: {last_idx[i].item()}")
                        print(f"  Last prompt token ID: {ids[i, last_idx[i]].item()}")
                        
                        # Get chosen/rejected
                        if "chosen_ids" in inputs:
                            ch_mask = inputs["chosen_mask"][i]
                            chosen = inputs["chosen_ids"][i][ch_mask].tolist()
                            rejected = inputs["rejected_token_id"][i].item()
                            print(f"  Chosen IDs: {chosen}")
                        else:
                            chosen = [inputs["chosen_token_id"][i].item()]
                            rejected = inputs["rejected_token_id"][i].item()
                            print(f"  Chosen ID: {chosen[0]}")
                        print(f"  Rejected ID: {rejected}")
                        
                        # Check probabilities
                        probs = logp_all[i].exp()
                        for c in chosen:
                            print(f"  P(token {c}): {probs[c].item():.6f}")
                        print(f"  P(token {rejected}): {probs[rejected].item():.6f}")
                        
                        # Are we even looking at reasonable probability mass?
                        top5 = torch.topk(probs, 5)
                        print(f"  Top 5 tokens: {top5.indices.tolist()} with probs {top5.values.tolist()}")
        # ────────────────────── END DEBUG BLOCK ───────────────────────────

        # Continue with your existing loss computation logic...
        if inputs.get("chosen_ids") is not None:
            # --- unpack ----------------------------------------------------------------
            ch_ids  = inputs["chosen_ids"].to(model.device)       # [B,C]
            ch_mask = inputs["chosen_mask"].to(model.device)      # [B,C] bool
            
            # --- rejected token ---------------------------------------------------------
            rej     = inputs["rejected_token_id"].to(model.device)  # [B]
            logp_bad = logp_all.gather(-1, rej.unsqueeze(-1)).squeeze(-1)  # [B]

            # --- per-token log-p and p --------------------------------------------------
            batch_rows = torch.arange(B, device=logp_all.device).unsqueeze(1)
            gathered   = logp_all[batch_rows, ch_ids]        # [B, C]  log p(chosen_i)
            probs_good = gathered.exp()                      # p(chosen_i)  ∈ (0,1)
            prob_bad   = logp_bad.unsqueeze(-1).exp()        # [B, 1]  broadcast

            # ---------------------------------------------------------------------------
            #  Weight rule
            #     • margin = p(chosen) − p(rejected)
            #     • if margin ≥ ε                → weight = 0
            #     • if margin ≤ −ε or large loss → weight = 1
            #     • otherwise                    → linear taper  (ε − margin) / ε
            # ---------------------------------------------------------------------------
            margin   = probs_good - prob_bad                 # [B, C]
            weights  = torch.clamp((eps - margin) / eps,
                                min=0.0, max=1.0) * ch_mask

            # fall-back: if *all* weights in a row are zero, keep every chosen token
            zero_row = weights.sum(dim=-1, keepdim=True) < 1e-12
            weights  = torch.where(zero_row, ch_mask.float(), weights)

            # ---------------------------------------------------------------------------
            #  Weighted mean p(chosen)  → logp_good
            # ---------------------------------------------------------------------------
            weights_sum        = weights.sum(dim=-1, keepdim=True)
            weighted_mean_prob = (probs_good * weights).sum(dim=-1, keepdim=True) / weights_sum
            logp_good          = weighted_mean_prob.log().squeeze(-1)      # [B]


            

        else:
            # single-token path
            chosen = inputs["chosen_token_id"].to(model.device)
            rejected = inputs["rejected_token_id"].to(model.device)
            logp_good = logp_all.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            logp_bad = logp_all.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

        # ───────────────────────────────────────────────────────────────────
        #  Variant-specific preference term
        # ───────────────────────────────────────────────────────────────────
        if mode == "ref":
            # ── reference pass (no grads) ─────────────────────────────────────
            with torch.no_grad():
                if self.ref_model is None:
                    # fall back to the "frozen copy of self" context manager
                    with self.null_ref_context():
                        ref_out = model(
                            ids,
                            attention_mask=attn,
                            position_ids=pos_full,
                            use_cache=False,
                            return_dict=True,
                        )
                else:
                    ref_out = self.ref_model(
                        ids,
                        attention_mask=attn,
                        position_ids=pos_full,
                        use_cache=False,
                        return_dict=True,
                    )

                # final-token logits → log-probs
                ref_logits_last = ref_out.logits[torch.arange(B), last_idx]    # [B,V]
                ref_logp_all    = F.log_softmax(ref_logits_last, dim=-1)

                if inputs.get("chosen_ids") is not None:
                    # multi-winner branch
                    ref_gathered = ref_logp_all.gather(-1, ch_ids)         # [B,C]
                    ref_probs    = ref_gathered.exp()
                    ref_weights  = torch.clamp((eps - ref_probs) / eps, min=0.0) * ch_mask
                    zero_row     = ref_weights.sum(dim=-1, keepdim=True) < 1e-12
                    ref_weights  = torch.where(zero_row, ch_mask.float(), ref_weights)
                    ref_wsum     = ref_weights.sum(dim=-1, keepdim=True)
                    ref_mean_p   = (ref_probs * ref_weights).sum(dim=-1, keepdim=True) / ref_wsum
                    ref_good     = ref_mean_p.log().squeeze(-1)            # [B]
                else:
                    # single-token branch
                    ref_good = ref_logp_all.gather(
                        -1, chosen.unsqueeze(-1)
                    ).squeeze(-1)                                          # [B]
 
                ref_bad = ref_logp_all.gather(
                    -1, rejected.unsqueeze(-1)
                ).squeeze(-1)                                              # [B]

            # ── preference loss using reference offset ─────────────────────
            delta     = (logp_good - ref_good) - (logp_bad - ref_bad)
            pref_loss = -F.logsigmoid(beta * delta).mean()
            kl_loss   = 0.0  # prompt-KL off for now

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
        }
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss



    # ----------------------------------------------------------
    def _prepare_dataset(self, dataset, *args, **_):
        return dataset