from core.finetuning import DPOTrainer, torch, pad_sequence, F

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

        #torch.autograd.set_detect_anomaly(True)
        

        # ── unpack ---------------------------------------------------------
        ids      = inputs["prompt_ids"].to(model.device)       # [B,L]
        attn     = inputs["attention_mask"].to(model.device)   # [B,L]
        #chosen   = inputs["chosen_token_id"].to(model.device)  # [B]
        rejected = inputs["rejected_token_id"].to(model.device)  # [B]

        # only enable gradient flow for the final token, not for the rest of the context
        # --- obtain final-layer hidden states, independent of model class ----------

        # -- forward -----------------------------------------------------------------
        out = model(ids,
                    attention_mask=attn,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True)

        last_hidden = (out.last_hidden_state
                    if getattr(out, "last_hidden_state", None) is not None
                    else out.hidden_states[-1])                       # [B, L, H]

        # ----- single position ------------------------------------------------------
        proj         = self._get_proj(model)
        target_dtype = proj.weight.dtype
        last_token   = last_hidden[:, -1, :].to(target_dtype)           # [B, H]

        logits_last  = proj(last_token)                                 # [B, V]
        logp_all     = F.log_softmax(logits_last, dim=-1)               # [B, V]

        


        if inputs.get("chosen_ids") is not None:
            DEBUG = False  
            EPS_P   = 1e-12        # clamp floor for probabilities
            # --- unpack ----------------------------------------------------------------
            ch_ids  = inputs["chosen_ids"].to(model.device)       # [B,C]
            ch_mask = inputs["chosen_mask"].to(model.device)      # [B,C] bool
            rej     = inputs["rejected_token_id"].to(model.device)  # [B]

            # --- per-token log-p and p --------------------------------------------------
            gathered = logp_all.gather(-1, ch_ids)                # log p_i            
            probs    = gathered.exp()

            # --- soft weight: 1 at p≤eps, linear decay to 0 at p≥1 ----------------------
            weights  = torch.clamp((eps - probs) / eps, min=0.0) * ch_mask
            zero_row = weights.sum(dim=-1, keepdim=True) < 1e-12  # all weights zero?
            weights  = torch.where(zero_row, ch_mask.float(), weights)

            weights_sum        = weights.sum(dim=-1, keepdim=True)
            weighted_mean_prob = (probs * weights).sum(dim=-1, keepdim=True) / weights_sum
            weighted_mean_prob = weighted_mean_prob.clamp_min(EPS_P)          # NEW
            logp_good          = weighted_mean_prob.log().squeeze(-1)

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
            delta  = (logp_good - logp_bad).clamp(min=-80.0, max=80.0)   # log-odds
            ratio  = torch.exp(delta)                     # odds ratio
            #clipped = torch.minimum(ratio, torch.tensor(1.0 + eps, device=ratio.device))
            #pref_loss = -torch.log(clipped / (1.0 + eps)).mean()
            MIN_R  = 1e-6
            clipped = torch.clamp(ratio, min=MIN_R, max=1.0 + eps)
            pref_loss = -torch.log(clipped / (1.0 + eps)).mean()
            kl_loss   = 0.0

        else:  # "free"
            delta  = (logp_good - logp_bad).clamp(min=-80.0, max=80.0)
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


        def _chk(name, t):
            if not torch.isfinite(t).all():
                bad = t[~torch.isfinite(t)]
                print(f"⚠️  {name} has {bad.numel()} non-finite values "
                    f"(min={bad.min().item()}, max={bad.max().item()})")

        _chk("logp_good",   logp_good)
        _chk("logp_bad",    logp_bad)
        _chk("delta",       delta)
        _chk("ratio",       ratio if 'ratio' in locals() else torch.tensor(0.))
        _chk("pref_loss",   pref_loss)
        _chk("total_loss",  loss)

        # -----------------------------------------------------------------------
        if not torch.isfinite(loss):
            print("non-finite loss detected – skipping backward")
            return loss * 0      # blocks gradient update for this batch
        # -----------------------------------------------------------------------

        #logits_last.retain_grad()
        #loss.backward(retain_graph=True)            # probe pass
        #g = logits_last.grad
        #print(f"∂L/∂logits  finite={torch.isfinite(g).all()}  "
        #    f"max|g|={g.abs().max().item():.4e}")
        

        if return_outputs:
            return loss, metrics
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        ids   = inputs["prompt_ids"].to(model.device)          # [B,L]
        attn  = inputs["attention_mask"].to(model.device)      # [B,L]
        chmat = inputs["chosen_ids"].to(model.device)          # [B,C]
        mask  = inputs["chosen_mask"].to(model.device)         # [B,C]

        # pick the first valid column in each row
        first_idx = mask.float().argmax(dim=1)                 # [B]
        label = chmat[torch.arange(chmat.size(0), device=chmat.device),
                      first_idx]                               # [B]

        out = model(ids, attention_mask=attn,
                    use_cache=False, return_dict=True)
        logits_last = out.logits[:, -1, :]                     # [B,V]
        #with torch.cuda.amp.autocast(enabled=False):
        #    logits_last = proj(last_token.float())      # FP32 mm

        loss = F.cross_entropy(logits_last, label)

        if return_outputs:
            return loss, {"nll": loss.detach()}
        return loss

    # ----------------------------------------------------------
    def _prepare_dataset(self, dataset, *args, **_):
        return dataset