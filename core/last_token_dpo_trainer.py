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
            prompt_ids = F.pad(prompt_ids, (0, pad_cols), value=pad_id)   # right pad

        attention_mask = prompt_ids.ne(pad_id)

        
        rejected = torch.tensor([f["rejected_token_id"] for f in features])
        # — multi-chosen support —
        if "chosen_ids" in features[0]:
            max_c = max(len(f["chosen_ids"]) for f in features)
            chosen_pad  = torch.full((len(features), max_c),
                                        -100, dtype=torch.long)
            chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
            for i, f in enumerate(features):
                ids = torch.tensor(f["chosen_ids"])
                chosen_pad [i, : ids.size(0)] = ids
                chosen_mask[i, : ids.size(0)] = True

            return dict(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                #chosen_token_id=chosen,          # always present
                rejected_token_id=rejected,
                chosen_ids=chosen_pad,           # None for plain tdpo
                chosen_mask=chosen_mask,
            )
        else:
            chosen   = torch.tensor([f["chosen_token_id"]   for f in features])
            return dict(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                chosen_token_id=chosen,          # always present
                rejected_token_id=rejected,
            )

        

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

        logp_all = F.log_softmax(logits_last, -1)

        if False: #inputs.get("chosen_ids") is not None:          # TDPO-MULTI
            ch_ids  = inputs["chosen_ids"].to(model.device)      # [B,C]
            ch_mask = inputs["chosen_mask"].to(model.device)

            # gather per-token log-probs, keep padding at −1e9
            gathered = logp_all.gather(-1, ch_ids).masked_fill(~ch_mask, -1e9)

            # ── NEW: freeze gradients for high-prob tokens ────────────────────
            probs        = gathered.exp()                     # p = e^{log p}
            detach_mask  = (probs > eps) & ch_mask           # ignore padding
            gathered     = torch.where(detach_mask,
                                    gathered.detach(),    # no grad
                                    gathered)             # keep grad

            #logp_good = torch.logsumexp(gathered, dim=-1)     # [B]

            count = ch_mask.sum(dim=-1, keepdim=True) # [B,1]
            logp_good = torch.logsumexp(gathered, dim=-1) - count.log().squeeze(-1)
            
        elif inputs.get("chosen_ids") is not None:
            DEBUG = True  
            ch_ids  = inputs["chosen_ids"].to(model.device)      # [B,C]
            ch_mask = inputs["chosen_mask"].to(model.device)     # [B,C] bool

            # 1. per-token log-p and prob
            gathered = logp_all.gather(-1, ch_ids)               # log p_i
            probs    = gathered.exp()                            # p_i

            # 2. soft down-weight once p_i exceeds eps
            weights  = torch.clamp((eps - probs) / eps, min=0.0) * ch_mask
            zero_rows = weights.sum(dim=-1, keepdim=True) < 1e-12
            weights   = torch.where(zero_rows, ch_mask.float(), weights)

            # 3. weighted mean probability, then log
            weights_sum        = weights.sum(dim=-1, keepdim=True)         # [B,1]
            weighted_mean_prob = (probs * weights).sum(dim=-1, keepdim=True) / weights_sum
            logp_good          = weighted_mean_prob.log().squeeze(-1)      # [B]

            # 4. optional deep-dive prints
            if DEBUG:
                for b in range(ch_ids.size(0)):
                    ids     = ch_ids[b][ch_mask[b]].tolist()
                    p_vals  = probs[b][ch_mask[b]].tolist()
                    w_vals  = weights[b][ch_mask[b]].tolist()
                    print(f"── batch {b} ─────────────────────────────────")
                    print(f"chosen ids      : {ids}")
                    print(f"probs           : {[round(p, 6) for p in p_vals]}")
                    print(f"weights         : {[round(w, 6) for w in w_vals]}")
                    print(f"weighted mean p : {weighted_mean_prob[b].item():.6e}")
                    print(f"logp_good       : {logp_good[b].item():.6f}")
                    print("────────────────────────────────────────────")


        else:
            # single-token path
            chosen = inputs["chosen_token_id"].to(model.device)
            logp_good = logp_all.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)


        # ── log-probs ------------------------------------------------------
        #logp_good = F.log_softmax(logits_last, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        logp_bad  = F.log_softmax(logits_last, -1).gather(-1, rejected.unsqueeze(-1)).squeeze(-1)
        print(logp_good.detach().cpu().numpy(), logp_bad.detach().cpu().numpy(), logp_good.detach().cpu().numpy() - logp_bad.detach().cpu().numpy())

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