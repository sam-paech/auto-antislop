#!/usr/bin/env python3
"""
Merge a LoRA adapter (saved by your finetune run) into the full-precision
base model and write the merged fp16 weights to disk.

Requires:
  pip install unsloth peft transformers accelerate
"""

from pathlib import Path
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoTokenizer

# ---------------------------------------------------------------------
# Adjust these three paths if your directory layout is different.
# ---------------------------------------------------------------------
BASE_MODEL   = "unsloth/gemma-3-4b-it"
ADAPTER_DIR  = (
    "results/auto_antislop_runs/run_20250608_102159/"
    "finetuned_model_ftpo_exp01/lora_adapters"
)
OUT_DIR      = (
    "results/auto_antislop_runs/run_20250608_102159/"
    "finetuned_model_ftpo_exp01/merged_manual_fp16"
)

# ---------------------------------------------------------------------
def main() -> None:
    print("→ loading base model …")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL,
        max_seq_length  = 4096,          # keep consistent with training
        load_in_4bit    = False,         # full-precision
        dtype           = torch.float16,
        device_map      = {"": "cpu"},   # CPU merge; change to {"": 0} for GPU
    )

    print("→ plugging in LoRA adapter …")
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        device_map = {"": "cpu"},
    )

    print("→ merging and unloading …")
    merged_model = peft_model.merge_and_unload()  # returns a plain nn.Module

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"→ saving merged model to {OUT_DIR} …")
    merged_model.save_pretrained(
        OUT_DIR,
        safe_serialization = True,       # *.safetensors shards
        max_shard_size     = "5GB",
    )

    # save the tokenizer so the directory is immediately usable
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(OUT_DIR)

    print("✓ done")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
