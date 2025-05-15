
import os
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_latest_finetuned_model() -> str | None:
    """
    Look under:
      1. <script dir>/results/auto_antislop_runs
      2. /results/auto_antislop_runs
    Return the most recent `…/finetuned_model*/merged_16bit` directory or None.
    """
    candidate_bases = [
        Path(__file__).resolve().parent / "results" / "auto_antislop_runs",
        Path("/results/auto_antislop_runs"),
    ]

    latest: tuple[float, Path] | None = None
    for base in candidate_bases:
        if not base.is_dir():
            continue

        # run_*/finetuned_model*/merged_16bit
        for merged_dir in base.glob("run_*/finetuned_model*/merged_16bit"):
            if not merged_dir.is_dir():
                continue
            mtime = merged_dir.parent.stat().st_mtime  # use finetuned_model* dir mtime
            if latest is None or mtime > latest[0]:
                latest = (mtime, merged_dir.resolve())

    return str(latest[1]) if latest else None


model_path = find_latest_finetuned_model() or "."
print(f"Loading model from: {os.path.abspath(model_path)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )

    messages = [
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a short, engaging story about a princess Elara in summertime."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\nApplied chat template:\n", prompt)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    print("\n--- Generated Story ---\n", response)
    print("\nToken count (approximate):", len(generated_ids[0]) - len(input_ids[0]))

except Exception as e:
    print(f"Error: {e}")
