import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_dpo_dataset(
    iter0_jsonl: Path,
    final_iter_jsonl: Path,
    output_jsonl: Path,
) -> None:
    logger.info(f"Creating DPO dataset from {iter0_jsonl.name} and {final_iter_jsonl.name} -> {output_jsonl.name}")

    KEY_PROMPT = "prompt"
    KEY_GENERATION = "generation"
    KEY_PROMPT_ID = "prompt_id" # Assuming antislop-vllm output includes this

    def _strip_wrapping(text: str) -> str:
        # This needs to match how prompts are formatted by antislop-vllm/main.py
        # If main.py's HF dataset loading adds "Writing prompt: ... Your response:\n", strip it.
        # For now, assume prompts in the JSONL are the "actual" prompts.
        # If antislop-vllm's output `prompt` field is already clean, this might not be needed.
        # The example in the notebook was:
        # prefix = "Writing prompt: "
        # if text.startswith(prefix): text = text[len(prefix):]
        # return text.strip()
        return text # Assuming prompt field in JSONL is already the core prompt

    def _load_file(path: Path) -> dict[str, dict[str, str]]:
        out_data: dict[str, dict[str, str]] = {}
        if not path.exists():
            logger.warning(f"DPO source file not found: {path}")
            return out_data
            
        with path.open(encoding="utf-8") as fh:
            for i, line_raw in enumerate(fh):
                try:
                    row = json.loads(line_raw)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line {i+1} in {path}")
                    continue
                
                if not isinstance(row, dict):
                    logger.warning(f"Skipping non-dict row {i+1} in {path}")
                    continue

                prompt_raw = row.get(KEY_PROMPT)
                gen = row.get(KEY_GENERATION)
                prompt_id = row.get(KEY_PROMPT_ID) # Use prompt_id as the primary key

                if not isinstance(prompt_raw, str) or not isinstance(gen, str) or prompt_id is None:
                    # logger.debug(f"Skipping row {i+1} in {path} due to missing/invalid prompt, generation, or prompt_id.")
                    continue
                
                prompt_clean = _strip_wrapping(prompt_raw)
                key = str(prompt_id) # Use prompt_id as the key

                if not key:
                    logger.warning(f"Skipping row {i+1} in {path} due to empty key (prompt_id).")
                    continue
                out_data[key] = {"prompt": prompt_clean, "generation": gen}
        return out_data

    data_iter0 = _load_file(iter0_jsonl)
    data_final = _load_file(final_iter_jsonl)
    
    if not data_iter0 or not data_final:
        logger.error("DPO dataset not created: one or both input files were empty or could not be loaded.")
        return

    common_keys = data_iter0.keys() & data_final.keys()

    if not common_keys:
        logger.warning("No overlapping prompt_ids between iteration-0 and final iteration; DPO dataset not written.")
        logger.warning(f"Iter0 keys: {len(data_iter0)}, Final keys: {len(data_final)}")
        return

    count_written = 0
    with output_jsonl.open("w", encoding="utf-8") as out_fh:
        for key in common_keys:
            # Use the prompt from iter0 as canonical, assuming prompt_id ensures they are fundamentally the same.
            prompt_for_dpo = data_iter0[key]["prompt"]
            
            # Sanity check: if prompts differ significantly despite same ID, log it.
            if prompt_for_dpo != data_final[key]["prompt"]:
                 logger.debug(f"Prompt text mismatch for prompt_id '{key}'. Using iter0 prompt for DPO pair.")

            rec = {
                "prompt": prompt_for_dpo,
                "chosen": data_final[key]["generation"],
                "rejected": data_iter0[key]["generation"],
            }
            # Ensure chosen and rejected are not identical
            if rec["chosen"] == rec["rejected"]:
                logger.debug(f"Skipping DPO pair for prompt_id '{key}' as chosen and rejected generations are identical.")
                continue

            json.dump(rec, out_fh, ensure_ascii=False)
            out_fh.write("\n")
            count_written +=1

    if count_written > 0:
        logger.info(f"📁 DPO dataset written -> {output_jsonl} ({count_written} prompt pairs from {len(common_keys)} common prompt_ids)")
    else:
        logger.warning(f"No DPO pairs written. Common prompt_ids found: {len(common_keys)}, but all might have had identical chosen/rejected texts.")