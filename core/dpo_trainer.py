# core/dpo_trainer.py

from __future__ import annotations
from typing import List, Optional

from core.finetuning import DPOTrainer, torch, F


class DPOTrainerWithChoiceWin(DPOTrainer):
    """
    Drop-in DPO trainer that adds a token-level `chosen_win` metric for
    dpo_final_token batches without altering DPO loss/behavior.

    Metric (per-row):
        Given the prompt context, compare the model's next-token probabilities:
            win = 1{ log p(chosen_first | prompt) > log p(rejected_first | prompt) }
        chosen_win = mean over rows of win.

    Expected batch keys (dpo_final_token):
        - chosen_input_ids, chosen_attention_mask
        - rejected_input_ids, rejected_attention_mask
        - prompt_input_ids (+ prompt_attention_mask)   OR
          prompt_ids        (+ attention_mask)
    """

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # 1) standard DPO loss (unchanged)
        loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # 2) metric (only if dpo_final_token fields are present)
        needed = {
            "chosen_input_ids", "chosen_attention_mask",
            "rejected_input_ids", "rejected_attention_mask",
        }
        if needed.issubset(inputs.keys()):
            chosen_win = self._metric_dpo_final_token(model, inputs)
            if chosen_win is not None:
                self.store_metrics({"chosen_win": chosen_win}, train_eval="train")

        if return_outputs:
            return loss, {}
        return loss

    # ----------------------------- helpers ---------------------------------

    @staticmethod
    def _first_real_index(mask_row: torch.Tensor) -> Optional[int]:
        """Index of the first non-pad token according to an attention mask row."""
        nz = mask_row.nonzero(as_tuple=False)
        return int(nz[0].item()) if nz.numel() else None

    # ---------------------- core metric (dpo_final_token) -------------------

    def _metric_dpo_final_token(self, model, inputs) -> Optional[torch.Tensor]:
        device = next(model.parameters()).device
        tok = getattr(self, "tokenizer", getattr(self, "processing_class", None))
        pad_id = getattr(tok, "pad_token_id", 0)

        # Continuations
        ch_ids = inputs["chosen_input_ids"].to(device)          # [B, Lc] (continuation only)
        ch_am  = inputs["chosen_attention_mask"].to(device)     # [B, Lc]
        rj_ids = inputs["rejected_input_ids"].to(device)        # [B, Lr]
        rj_am  = inputs["rejected_attention_mask"].to(device)   # [B, Lr]

        # Prompt (prefer explicit prompt_input_ids → fall back to prompt_ids)
        pr_ids_key = "prompt_input_ids" if "prompt_input_ids" in inputs else (
            "prompt_ids" if "prompt_ids" in inputs else None
        )
        pr_am_key = "prompt_attention_mask" if "prompt_attention_mask" in inputs else (
            "attention_mask" if "attention_mask" in inputs else None
        )

        if pr_ids_key is None:
            # No prompt in the batch → cannot compute the metric reliably
            return None

        pr_ids_full = inputs[pr_ids_key].to(device)  # [B, Lp]
        if pr_am_key in inputs:
            pr_am_full = inputs[pr_am_key].to(device)  # [B, Lp]
        else:
            # derive attention mask from pad id
            pr_am_full = pr_ids_full.ne(pad_id).to(pr_ids_full.dtype)

        B = ch_ids.size(0)

        # Assemble per-row prompt and first continuation tokens
        prompts: List[torch.Tensor] = []
        chosen_first: List[int] = []
        rejected_first: List[int] = []
        last_idx: List[int] = []

        for b in range(B):
            # first token of each continuation (continuations are typically 1 token + EOS)
            k_ch = self._first_real_index(ch_am[b])
            k_rj = self._first_real_index(rj_am[b])
            if k_ch is None or k_rj is None:
                continue

            ch_first = int(ch_ids[b, k_ch].item())
            rj_first = int(rj_ids[b, k_rj].item())

            # prompt (all real tokens)
            pr_mask = pr_am_full[b].bool()
            pr_seq = pr_ids_full[b][pr_mask]  # 1D tensor with real prompt ids
            if pr_seq.numel() == 0:
                continue

            prompts.append(pr_seq)
            chosen_first.append(ch_first)
            rejected_first.append(rj_first)
            last_idx.append(pr_seq.numel() - 1)

        n = len(prompts)
        if n == 0:
            return None

        # Right-pad prompts into a dense batch for one forward pass
        max_pr = max(p.numel() for p in prompts)
        prompt_ids = pr_ids_full.new_full((n, max_pr), pad_id)
        prompt_am  = pr_am_full.new_zeros((n, max_pr))

        for i, p in enumerate(prompts):
            Lp = p.numel()
            prompt_ids[i, :Lp] = p
            prompt_am [i, :Lp] = 1

        last_idx_t = torch.tensor(last_idx, device=device, dtype=torch.long)
        chosen_tok = torch.tensor(chosen_first,   device=device, dtype=torch.long)
        reject_tok = torch.tensor(rejected_first, device=device, dtype=torch.long)

        # One no-grad forward on prompts; evaluate next-token distribution at last prompt token
        with torch.no_grad():
            out = model(prompt_ids, attention_mask=prompt_am, use_cache=False, return_dict=True)
            logits_last = out.logits[torch.arange(n, device=device), last_idx_t, :]  # [n, V]
            logp_last   = F.log_softmax(logits_last, dim=-1)

            lp_good = logp_last.gather(1, chosen_tok.unsqueeze(1)).squeeze(1)   # [n]
            lp_bad  = logp_last.gather(1, reject_tok.unsqueeze(1)).squeeze(1)   # [n]
            wins    = (lp_good > lp_bad).float()

        return wins.mean().detach()
