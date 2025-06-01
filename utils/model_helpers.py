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


# --------------------------------------------------------------------
#  restore Gemma-3’s weight-tying and ensure only the embed_tokens
#  key lands in the safetensors index (no lm_head, no duplication)
# --------------------------------------------------------------------
# ---------------------------------------------------------------
#  Gemma-3: keep weight-tying *and* give vLLM the path it wants
# ---------------------------------------------------------------
def retie_gemma3_and_prune_alias(model):
    """
    Re-establish tying between embeddings and logits projection and ensure
    the projection is reachable at `language_model.output_projection`.
    Removes the top-level `lm_head` alias so the serializer never emits
    an `lm_head.*` key.

    Call just before `save_pretrained(...)`.
    """
    import torch.nn as nn

    if (getattr(model.config, "model_type", "") or "").lower() != "gemma3":
        return  # skip for anything that isn't Gemma-3

    emb  = model.get_input_embeddings()          # nn.Embedding
    proj = getattr(model, "lm_head", None)       # HF always defines this

    if proj is None or not isinstance(proj, nn.Linear):
        raise RuntimeError("Could not find lm_head on Gemma-3 model")

    # ── tie weights if they were detied earlier ─────────────────────────
    if proj.weight.data_ptr() != emb.weight.data_ptr():
        proj.weight = emb.weight      # share storage again
    model.config.tie_word_embeddings = True

    # ── ensure wrapper + attribute for vLLM ────────────────────────────
    # 1. make / fetch `model.language_model`
    if not hasattr(model, "language_model"):
        wrapper = nn.Module()
        model.add_module("language_model", wrapper)
    else:
        wrapper = model.language_model

    # 2. register projection inside wrapper
    wrapper.add_module("output_projection", proj)

    # ── drop the top-level alias so it won't be serialised ─────────────
    if hasattr(model, "lm_head"):
        delattr(model, "lm_head")


# ------------------------------------------------------------------
#  Gemma-3 helper: detie + relabel head for vLLM + safetensors
# ------------------------------------------------------------------
def prepare_gemma3_for_save(model):
    """
    • Makes the output projection an independent tensor if it still shares
      storage with the embeddings.
    • Registers it at `language_model.lm_head` (the path vLLM uses).
    • Deletes the top-level `lm_head` alias so no `lm_head.*` key is saved.
    • Sets `tie_word_embeddings=False` so Transformers knows they’re untied.
    """
    import torch.nn as nn, torch

    if (getattr(model.config, "model_type", "") or "").lower() != "gemma3":
        return

    emb  = model.get_input_embeddings()
    head = model.get_output_embeddings()           # this is model.lm_head

    # 1. Detie if they still share storage
    if head.weight.data_ptr() == emb.weight.data_ptr():
        vocab, hidden = emb.weight.shape
        new_head = nn.Linear(hidden, vocab, bias=False)
        new_head.weight = nn.Parameter(emb.weight.detach().clone())
        new_head.to(next(model.parameters()).dtype)
        head = new_head

    # 2. Ensure `language_model` wrapper exists
    if not hasattr(model, "language_model"):
        model.add_module("language_model", nn.Module())

    # 3. Register under vLLM path
    model.language_model.add_module("lm_head", head)

    # 4. Drop the alias so no `lm_head.*` key lands in the state-dict
    if hasattr(model, "lm_head"):
        delattr(model, "lm_head")

    model.config.tie_word_embeddings = False
