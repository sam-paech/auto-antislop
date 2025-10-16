#!/usr/bin/env python3
"""
Gemma-3 key repair utility
==========================

Repairs checkpoints whose weight names are in either of the two incorrect
forms:

1.  model.language_model.embed_tokens.weight          (# leading "model.")
2.  language_model.embed_tokens.weight                (# missing ".model.")

to the correct form:

    language_model.model.embed_tokens.weight

Usage:
    python repair_gemma3_keys.py /path/to/checkpoint_dir
"""

import sys, json, shutil
from pathlib import Path
from safetensors.torch import safe_open, save_file

BAD_LEADING  = "model."                    # variant 1
GOOD_PREFIX  = "language_model."
GOOD_FULL    = "language_model.model."
OUT_SUFFIX   = "_repaired"

# ----------------------------------------------------------------------
# key transformation ----------------------------------------------------
# ----------------------------------------------------------------------

def fix_key(key: str) -> str:
    """
    1) strip a leading "model." if present
    2) ensure "language_model." is followed by "model."
    """
    # step 1 – drop wrapper prefix once
    if key.startswith(BAD_LEADING):
        key = key[len(BAD_LEADING):]

    # step 2 – insert ".model." if missing
    if key.startswith(GOOD_PREFIX) and not key.startswith(GOOD_FULL):
        key = GOOD_FULL + key[len(GOOD_PREFIX):]

    return key

# ----------------------------------------------------------------------
# shard processing ------------------------------------------------------
# ----------------------------------------------------------------------

def repair_shard(src: Path, dst: Path) -> None:
    """
    Re-write a .safetensors shard with corrected keys.
    """
    corrected = {}

    with safe_open(src, framework="pt", device="cpu") as f:
        for old_key in f.keys():
            corrected[fix_key(old_key)] = f.get_tensor(old_key)

    save_file(corrected, dst, metadata={"format": "pt"})

# ----------------------------------------------------------------------
# driver ----------------------------------------------------------------
# ----------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: repair_gemma3_keys.py <checkpoint_dir>")

    src_dir = Path(sys.argv[1]).expanduser().resolve()
    if not src_dir.is_dir():
        sys.exit(f"Directory not found: {src_dir}")

    out_dir = src_dir.with_name(src_dir.name + OUT_SUFFIX)
    out_dir.mkdir(exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        sys.exit("model.safetensors.index.json not found in checkpoint dir.")

    # ---- load index ----------------------------------------------------
    with open(index_path, "r") as f:
        index = json.load(f)

    new_weight_map   = {}
    processed_shards = set()

    # ---- process every tensor key -------------------------------------
    for old_key, shard_name in index["weight_map"].items():
        new_key = fix_key(old_key)
        new_weight_map[new_key] = shard_name

        if shard_name in processed_shards:
            continue
        processed_shards.add(shard_name)
        repair_shard(src_dir / shard_name, out_dir / shard_name)

    index["weight_map"] = new_weight_map

    # ---- write new index ----------------------------------------------
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # ---- copy auxiliary files -----------------------------------------
    for fp in src_dir.iterdir():
        if fp.name == "model.safetensors.index.json" or fp.suffix == ".safetensors":
            continue
        shutil.copy2(fp, out_dir / fp.name)

    print(f"✓ Repaired checkpoint written to {out_dir}")

if __name__ == "__main__":
    main()

