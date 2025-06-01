# ---------------------------------------------------------------------
#  helper: ensure Gemma-3 checkpoints use  language_model.model.…  keys
# ---------------------------------------------------------------------
import os, json, logging
from pathlib import Path
from safetensors.torch import safe_open, save_file

log = logging.getLogger(__name__)

def fix_gemma3_checkpoint(ckpt_dir: str | Path) -> None:
    """
    If `ckpt_dir` is a Gemma-3 checkpoint whose tensor keys look like
        language_model.embed_tokens.weight
    instead of
        language_model.model.embed_tokens.weight
    rewrite the shards and index file in-place.

    No-op when:
      • model_type ≠ 'gemma3'
      • keys are already correct
      • required files are missing
    """
    ckpt_dir = Path(ckpt_dir)
    index_file = ckpt_dir / "model.safetensors.index.json"
    config_file = ckpt_dir / "config.json"
    if not index_file.is_file() or not config_file.is_file():
        return  # nothing to do

    # ── guard: only patch Gemma-3 checkpoints ───────────────────────────
    try:
        with open(config_file) as f:
            cfg = json.load(f)
        if (cfg.get("model_type") or "").lower() != "gemma3":
            return
    except Exception as e:
        log.warning("Could not read %s (%s); skipping fix.", config_file, e)
        return

    # ── scan weight map ─────────────────────────────────────────────────
    with open(index_file) as f:
        idx = json.load(f)

    wm = idx["weight_map"]
    broken = [
        k for k in wm
        if k.startswith("language_model.") and not k.startswith("language_model.model.")
    ]
    if not broken:
        return                                      # already fine

    log.info("Repairing Gemma-3 key prefixes in %s", ckpt_dir)

    def _fixed(k: str) -> str:
        if k.startswith("language_model.") and not k.startswith("language_model.model."):
            return "language_model.model." + k[len("language_model."):]
        return k

    # ── rewrite every shard exactly once ────────────────────────────────
    repaired_shards = set()
    for old_key, shard_name in wm.items():
        wm[_fixed(old_key)] = wm.pop(old_key)       # update key in dict
        if shard_name in repaired_shards:
            continue
        repaired_shards.add(shard_name)

        src = ckpt_dir / shard_name
        tmp = ckpt_dir / (shard_name + ".tmp")

        fixed_tensors = {}
        with safe_open(src, framework="pt", device="cpu") as f:
            for k in f.keys():
                fixed_tensors[_fixed(k)] = f.get_tensor(k)

        save_file(fixed_tensors, tmp, metadata={"format": "pt"})
        tmp.replace(src)                            # atomic overwrite

    # ── write new index ────────────────────────────────────────────────
    with open(index_file, "w") as f:
        json.dump(idx, f, indent=2)

    log.info("✓ Gemma-3 checkpoint repaired.")


# fully detie lm_head from embeddings so safetensors can flatten
def detie_lm_head(model):
    """
    Untie the logits projection from the input embeddings and register it
    *exactly* where the model (and loaders like vLLM) expect it.

    Works with HF models whose output head is either `lm_head` or some
    nested attribute (e.g. `language_model.output_projection` in Gemma-3).
    """
    import torch
    from types import SimpleNamespace

    emb = model.get_input_embeddings()          # nn.Embedding
    old_head = model.get_output_embeddings()    # whatever Linear HF exposes

    # nothing to do if they are already separate tensors
    if old_head.weight.data_ptr() != emb.weight.data_ptr():
        return

    vocab_size, hidden_size = emb.weight.shape
    new_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    new_head.weight = torch.nn.Parameter(emb.weight.detach().clone())
    new_head.to(next(model.parameters()).dtype)

    # ------------------------------------------------------------------
    # find the *attribute path* of the existing output head
    # ------------------------------------------------------------------
    path = None
    for name, module in model.named_modules():
        if module is old_head:
            path = name            # e.g. "lm_head" or "language_model.output_projection"
            break
    if path is None:               # very unusual, but fall back to "lm_head"
        path = "lm_head"

    print('!!', name)
    
    # ------------------------------------------------------------------
    # install the new head at that path
    # ------------------------------------------------------------------
    def set_by_path(root, dotted_name, value):
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], value)

    set_by_path(model, path, new_head)

    # HF convenience: if the public attribute `lm_head` *is not* the main path,
    # mirror it so code expecting `model.lm_head` still works. This does *not*
    # duplicate weights – both names reference the same nn.Linear instance.
    #if path != "lm_head":
    #    model.lm_head = new_head

    model.config.tie_word_embeddings = False

    