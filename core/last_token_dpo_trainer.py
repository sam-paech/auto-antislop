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
        ids      = inputs["prompt_ids"].to(model.device)       # [B,L]
        attn     = inputs["attention_mask"].to(model.device)   # [B,L]
        #chosen   = inputs["chosen_token_id"].to(model.device)  # [B]
        rejected = inputs["rejected_token_id"].to(model.device)  # [B]

        



        # only enable gradient flow for the final token, not for the rest of the context
        # ──────────────────────────────────────────────────────────────
        #  Two-pass KV-cache: prompt frozen, last token trainable
        # ──────────────────────────────────────────────────────────────
        pad_id   = self.padding_value          # e.g. tokenizer.pad_token_id
        last_idx = attn.sum(1) - 1            # [B]  index of final real token
        B, L = ids.shape
        

        # ------------------------------------------------------------------
        #  Build logical position indices that ignore the left pads
        # ------------------------------------------------------------------
        seq_len  = attn.sum(1)                           # [B]  (= n)
        pad_off  = (L - seq_len).unsqueeze(1)            # [B,1]
        arange_L = torch.arange(L, device=ids.device).unsqueeze(0)  # [1,L]

        pos_ctx = (arange_L - pad_off).clamp(min=0)      # [-pads … n-2,n-1]
        pos_ctx = pos_ctx.masked_fill(attn == 0, 0)      # pads → 0

        pos_tok = (seq_len - 1).unsqueeze(1)             # [B,1]  value n-1

        # ids  : [B,L]  left-padded prompt
        # attn : [B,L]
        B, L = ids.shape
        last_idx = attn.sum(1) - 1                # [B]

        # ---------- context pass (unchanged, no grad) --------------------
        ctx_ids  = ids.clone()
        ctx_ids[torch.arange(B), last_idx] = pad_id
        ctx_attn = attn.clone()
        ctx_attn[torch.arange(B), last_idx] = 0

        with torch.no_grad():
            ctx_out = model(
                ctx_ids,
                attention_mask = ctx_attn,
                use_cache      = True,
                return_dict    = True,
            )
        past_kv = ctx_out.past_key_values          # frozen

        # ---------- last-token pass  (override position_id) --------------
        tok_ids  = ids.gather(1, last_idx.unsqueeze(1))   # [B,1]  token n-1
        tok_attn = torch.ones_like(tok_ids, dtype=attn.dtype)
        pos_tok  = last_idx.unsqueeze(1)                  # [B,1]  value = L-1

        tok_out = model(
            tok_ids,
            attention_mask = tok_attn,
            position_ids   = pos_tok,     # ← key line
            past_key_values= past_kv,
            use_cache      = False,
            return_dict    = True,
        )

        logp_all = F.log_softmax(tok_out.logits.squeeze(1), dim=-1)   # [B,V]




        # ───────────────────────── DEBUG BLOCK ────────────────────────────
        # Paste inside compute_loss immediately after `logp_all` is defined.
        DEBUG = True                         # flip to False when done
        if DEBUG and not getattr(self, "_debug_ran", False):
            self._debug_ran = True           # run only on first training batch

            # 1) shape sanity ------------------------------------------------
            assert tok_ids.shape[1] == 1,  f"tok_ids shape {tok_ids.shape}"
            assert logp_all.dim()   == 2,  f"logp_all dim {logp_all.dim()}"

            # 2) single-pass reference distribution -------------------------
            with torch.no_grad():
                one_logits = model(
                    ids,
                    attention_mask=attn,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]                    # [B, V]
                one_logp = F.log_softmax(one_logits, -1)

            max_abs_diff = (one_logp - logp_all).abs().max().item()
            print(f"[DBG] max |Δ log-p| two-pass vs one-pass : {max_abs_diff:.3e}")

            # 3) choice-win comparison --------------------------------------
            batch_rows = torch.arange(ids.size(0), device=ids.device)

            if "chosen_ids" in inputs:                # TDPO-MULTI
                ch_ids  = inputs["chosen_ids" ].to(ids.device)       # [B,C]
                ch_mask = inputs["chosen_mask"].to(ids.device)       # [B,C]
                rej     = inputs["rejected_token_id"].to(ids.device) # [B]

                probe_good = one_logp[batch_rows.unsqueeze(1), ch_ids]     # [B,C]
                probe_bad  = one_logp[batch_rows, rej].unsqueeze(1)        # [B,1]
                wins_probe = (probe_good > probe_bad) & ch_mask
                probe_win  = (wins_probe.float().sum(-1) /
                            ch_mask.sum(-1)).mean().item()

                int_good = logp_all[batch_rows.unsqueeze(1), ch_ids]
                int_bad  = logp_all[batch_rows, rej].unsqueeze(1)
                wins_int = (int_good > int_bad) & ch_mask
                int_win  = (wins_int.float().sum(-1) /
                            ch_mask.sum(-1)).mean().item()

                same_id = ((ch_ids == rej.unsqueeze(1)) & ch_mask).sum().item()
            else:                                     # single-winner
                ch_ids = inputs["chosen_token_id"].to(ids.device)           # [B]
                rej    = inputs["rejected_token_id"].to(ids.device)         # [B]

                probe_good = one_logp[batch_rows, ch_ids]                   # [B]
                probe_bad  = one_logp[batch_rows, rej]                      # [B]
                probe_win  = (probe_good > probe_bad).float().mean().item()

                int_good = logp_all[batch_rows, ch_ids]
                int_bad  = logp_all[batch_rows, rej]
                int_win  = (int_good > int_bad).float().mean().item()

                same_id = (ch_ids == rej).sum().item()

            print(f"[DBG] probe  win (one-pass) = {probe_win:.4f}")
            print(f"[DBG] internal win (two-pass)= {int_win:.4f}")

            if same_id:
                print(f"[WARN] {same_id} examples where chosen == rejected!")

            # 4) stop after first batch so you can inspect the log ----------
            raise RuntimeError("Debug run complete – stopping trainer.")
        # ────────────────────── END DEBUG BLOCK ───────────────────────────



        

        if inputs.get("chosen_ids") is not None:
            DEBUG = False  
            # --- unpack ----------------------------------------------------------------
            ch_ids  = inputs["chosen_ids"].to(model.device)       # [B,C]
            ch_mask = inputs["chosen_mask"].to(model.device)      # [B,C] bool
            rej     = inputs["rejected_token_id"].to(model.device)  # [B]

            # --- per-token log-p and p --------------------------------------------------
            batch_rows = torch.arange(B, device=logp_all.device).unsqueeze(1)
            gathered   = logp_all[batch_rows, ch_ids]           # [B, C]
            probs     = gathered.exp()                     # p_i

            # --- soft weight: 1 at p≤eps, linear decay to 0 at p≥1 ----------------------
            weights  = torch.clamp((eps - probs) / eps, min=0.0) * ch_mask
            zero_row = weights.sum(dim=-1, keepdim=True) < 1e-12  # all weights zero?
            weights  = torch.where(zero_row, ch_mask.float(), weights)

            weights_sum        = weights.sum(dim=-1, keepdim=True)         # [B,1]
            weighted_mean_prob = (probs * weights).sum(dim=-1, keepdim=True) / weights_sum
            logp_good          = weighted_mean_prob.log().squeeze(-1)      # [B]

            # --- rejected token ---------------------------------------------------------
            logp_bad = logp_all.gather(-1, rej.unsqueeze(-1)).squeeze(-1)  # [B]
            p_bad    = logp_bad.exp()

            # --- optional verbose inspection -------------------------------------------
            if DEBUG:
                torch.set_printoptions(precision=9, sci_mode=True)
                for b in range(ch_ids.size(0)):
                    real = ch_mask[b]
                    ids  = ch_ids[b][real].tolist()
                    lp   = gathered[b][real]          # log p_i
                    p    = lp.exp()
                    w    = weights[b][real]

                    print(f"\n── batch {b} ─────────────────────────────────────────")
                    for i, (tid, lpi, pi, wi) in enumerate(zip(ids, lp, p, w)):
                        print(f"  {i:02d}  id={tid:<6}  logp={lpi.item(): .9f}  "
                            f"p={pi.item():.3e}  w={wi.item():.3f}")
                    print(f"  -----")
                    print(f"  weighted mean p : {weighted_mean_prob[b].item():.9e}")
                    print(f"  logp_good       : {logp_good[b].item(): .9f}")
                    print(f"  logp_bad        : {logp_bad [b].item(): .9f}")
                    print(f"  p_bad           : {p_bad   [b].item():.3e}")
                    print(f"  margin (Δ)      : {(logp_good[b]-logp_bad[b]).item(): .9f}")
                    print("───────────────────────────────────────────────────")



        else:
            # single-token path
            chosen = inputs["chosen_token_id"].to(model.device)
            logp_good = logp_all.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)


        # ── log-probs ------------------------------------------------------        
        #logp_bad  = F.log_softmax(logits_last, -1).gather(-1, rejected.unsqueeze(-1)).squeeze(-1)
        logp_bad = logp_all.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)
        #print(logp_good.detach().cpu().numpy(), logp_bad.detach().cpu().numpy(), logp_good.detach().cpu().numpy() - logp_bad.detach().cpu().numpy())

        # ───────────────────────────────────────────────────────────────────
        #  Variant-specific preference term
        # ───────────────────────────────────────────────────────────────────
        if mode == "ref":
            # ── reference pass (no grads) ─────────────────────────────────────
            with torch.no_grad():
                if self.ref_model is None:
                    # fall back to the “frozen copy of self” context manager
                    with self.null_ref_context():
                        ref_out = model(
                            ids,
                            attention_mask=attn,
                            use_cache=False,
                            return_dict=True,
                        )
                else:
                    ref_out = self.ref_model(
                        ids,
                        attention_mask=attn,
                        use_cache=False,
                        return_dict=True,
                    )

                # final-token logits → log-probs
                ref_logits_last = ref_out.logits[:, -1, :]                 # [B,V]
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
            lp_bad    = logp_bad.unsqueeze(-1)                    # [B,1] -> [B,1]

            # boolean wins per token (requires same mask as ch_mask)
            wins_tok  = (lp_chosen > lp_bad) & ch_mask            # [B,C] bool
            frac_win  = wins_tok.float().sum(-1) / ch_mask.sum(-1)  # [B]
            choice_win = frac_win.mean().detach()                 # scalar
        else:                                        # TDPO single
            choice_win = (logp_good > logp_bad).float().mean().detach()

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