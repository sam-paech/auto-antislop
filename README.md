# Auto-Antislop

Auto-Antislop is an automated pipeline designed to iteratively refine text generation datasets and finetune language models to reduce "slop" – common, repetitive, or undesirable phrases and patterns often found in LLM outputs. It combines generation, analysis, and DPO/FTPO (Direct Preference Optimization / Final Token Preference Optimization) finetuning into a configurable workflow.

The core idea is to:
1.  Generate text using a base model.
2.  Analyze this text against a human baseline and its own patterns to identify "slop."
3.  Create ban lists (n-grams, phrases, regex).
4.  Re-generate text, this time actively penalizing or preventing the generation of banned content.
5.  Repeat analysis and ban list updates for several iterations.
6.  Create a preference dataset where generations from the final, "anti-slopped" iteration are "chosen" and baseline generations are "rejected."
7.  Finetune the base model on this preference dataset using DPO or FTPO.

## 🌟 Key Features

*   **Iterative Dataset Refinement:** Automatically runs multiple cycles of generation and analysis.
*   **Comprehensive Slop Detection:**
    *   N-gram frequency analysis (compared to a human baseline).
    *   Over-represented word detection.
    *   Slop phrase extraction (using the `slop-forensics` submodule).
    *   Custom phrase and regex ban lists.
*   **Dynamic Banning during Generation:** Leverages the `antislop-vllm` submodule to apply bans in real-time.
*   **Automated Preference Dataset Creation:** Generates `chosen`/`rejected` pairs for DPO/FTPO.
*   **Integrated DPO/FTPO Finetuning:**
    *   Supports standard DPO and a custom "final-token DPO" (FTPO) for fine-grained preference on single token choices.
    *   Optional Unsloth integration for faster training and reduced memory usage.
    *   Automatic learning rate scaling.
    *   Early stopping callbacks.
*   **vLLM Management:** Can optionally start and stop a local vLLM server for generation.
*   **Resumability:** Pipeline can be resumed from a specific iteration if interrupted.
*   **Highly Configurable:** Most aspects are controlled via a central YAML configuration file.
*   **Detailed Output & Metrics:** Saves analysis results, ban lists, and performance metrics for each iteration.

## ⚙️ How It Works (Pipeline Flow)

1.  **(Optional) vLLM Server Management:** If `manage_vllm` is true, the script starts a vLLM server. If a server is already running on the configured port, or if `manage_vllm` is false, the script assumes an external vLLM server.
2.  **Iteration Loop (`num_iterations` times):**
    *   **Generation (Iter 0 - Baseline):**
        *   The `antislop-vllm` script generates text from a source dataset (e.g., writing prompts) *without* any ban lists active. This forms the "rejected" data for DPO.
    *   **Analysis (All Iterations):**
        *   The generated text is analyzed using tools from `slop-forensics` and custom analysis scripts:
            *   N-gram frequencies are compared against a human writing profile.
            *   Over-represented words are identified.
            *   Common "slop phrases" are extracted.
        *   Metrics like lexical diversity (TTR, RTTR) and repetition scores are calculated.
    *   **Ban List Update (All Iterations):**
        *   Based on the analysis, n-gram and slop phrase ban lists are created or updated. User-supplied `extra_ngrams_to_ban`, `extra_slop_phrases_to_ban`, and `extra_regex_patterns` from the config are merged.
    *   **Generation (Iter 1+ - Anti-Slop):**
        *   `antislop-vllm` generates text again, but this time it uses the accumulated ban lists (n-grams, slop phrases, regex) to guide generation away from undesirable content.
        *   The output of the final iteration serves as the "chosen" data for DPO.
3.  **DPO/FTPO Dataset Creation:**
    *   Pairs are created:
        *   `prompt`: The original input prompt.
        *   `chosen`: Generation from the *final* anti-slop iteration.
        *   `rejected`: Generation from the *initial* baseline iteration (Iter 0).
    *   For FTPO mode, a specialized dataset is created focusing on single-token choices where the model was guided away from a "bad" token.
4.  **(Optional) DPO/FTPO Finetuning:**
    *   If `finetune_enabled` is true, the script runs DPO or FTPO finetuning using the preference dataset.
    *   Supports LoRA and optional 4-bit quantization (QLoRA via Unsloth or bitsandbytes).
    *   Saves the LoRA adapters and optionally a merged 16-bit model.
5.  **(Optional) vLLM Server Shutdown:** If the script started vLLM, it stops it.

## 🚀 Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   NVIDIA GPU with CUDA installed (for vLLM and finetuning).
    *   Git.

2.  **Clone the Repository (with submodules):**
    ```bash
    git clone --recurse-submodules https://github.com/your-username/auto-antislop.git
    cd auto-antislop
    ```
    If you've already cloned without submodules, run:
    ```bash
    git submodule update --init --recursive
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on vLLM & Torch:** `vllm` and `torch` are listed in `requirements.txt` but commented out. It's often best to install versions compatible with your specific CUDA setup.
        *   If you plan to use the `--manage-vllm` feature, install `vllm` (e.g., `pip install vllm`).
        *   Ensure PyTorch is installed with CUDA support (see [pytorch.org](https://pytorch.org/)).
        *   Unsloth (for finetuning) will install its required torch version if not present or incompatible.

4.  **NLTK Data:**
    The script will attempt to download necessary NLTK resources (`punkt`, `punkt_tab`, `stopwords`) on first run. If this fails due to network issues, you might need to download them manually:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab') # For NLTK 3.9+
    nltk.download('stopwords')
    ```

## ⚙️ Configuration

The primary configuration is done through a YAML file. Refer to the example provided: `auto_antislop_config.yaml`.

Key configuration sections:

*   **`experiment_base_dir`**: Base directory for all output runs.
*   **`human_profile_path`**: Path to a JSON file containing statistics about human writing (used as a baseline for n-gram analysis).
*   **`log_level`**: Logging verbosity for the script.
*   **vLLM Management (`manage_vllm`, `vllm_model_id`, `vllm_port`, etc.):** Settings for the vLLM server if managed by the script.
*   **Iterative Pipeline (`num_iterations`):** Number of generation/analysis cycles.
*   **Generation Script Parameters (`generation_step_enabled`, `generation_api_base_url`, `generation_model_id`, `generation_max_prompts`, etc.):** Parameters passed to `antislop-vllm` for text generation.
    *   `generation_prompt_template`: How to wrap prompts from the dataset.
    *   `generation_force_backtrack`, `generation_invert_probs`: Advanced sampling controls in `antislop-vllm`.
*   **N-Gram Analysis & Banning (`enable_ngram_ban`, `top_k_bigrams`, quotas, `extra_ngrams_to_ban`):** Controls for n-gram based slop detection.
*   **Over-Represented Word Analysis (`compute_overrep_words`, quotas):** Controls for identifying and banning overused words.
*   **Slop Phrase Banning (`enable_slop_phrase_ban`, quotas, `extra_slop_phrases_to_ban`):** Controls for phrase-based slop detection.
*   **Regex Banning (`extra_regex_patterns`):** User-supplied regex patterns to ban.
*   **Metrics (`min_word_len_for_analysis`, etc.):** Parameters for analysis.
*   **Finetuning (`finetune_enabled`, `finetune_mode`, `finetune_base_model_id`, LoRA params, etc.):**
    *   `finetune_mode`: "dpo" or "ftpo".
    *   `finetune_use_unsloth`: Set to `true` to use Unsloth for finetuning.
    *   `finetune_ftpo_dataset`: Optionally specify an existing FTPO dataset.
    *   `finetune_cuda_visible_devices`: GPU mask specifically for the finetuning stage.

## 🛠️ Usage

1.  **Prepare Configuration:**
    *   Copy `auto_antislop_config.yaml` (if it's an example) or create your own.
    *   Modify it to suit your needs (model IDs, paths, desired number of iterations, etc.).
    *   Ensure your `human_profile_path` points to a valid JSON file.

2.  **Run the Pipeline:**
    ```bash
    python main.py --config-file path/to/your_config.yaml
    ```

3.  **Key Command-Line Arguments (override config settings):**
    *   `--config-file`: Path to the main YAML configuration file (default: `auto_antislop_config.yaml`).
    *   `--resume-from-dir`: Path to an existing experiment run directory to resume.
    *   `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    *   `--manage-vllm [true/false]`: Override vLLM management.
    *   `--vllm-port <port>`: Override vLLM port.
    *   `--vllm-model-id <model_id>`: Override vLLM model.
    *   `--num-iterations <N>`: Override number of iterations.
    *   `--generation-max-prompts <N>`: Override max prompts for generation.
    *   `--run-finetune [true/false]`: Override finetuning step.
    *   `--finetune-base-model-id <model_id>`: Override base model for DPO.
    *   `--finetune-mode [dpo/ftpo]`: Override finetuning mode.
    *   `--finetune-cuda-visible-devices "0,1"`: Set specific GPUs for finetuning.

4.  **Resuming Runs:**
    If the pipeline is interrupted, you can resume it by providing the path to the experiment directory:
    ```bash
    python main.py --config-file path/to/your_config.yaml --resume-from-dir results/auto_antislop_runs/run_YYYYMMDD_HHMMSS
    ```
    The script will attempt to pick up from the last successfully completed part of the iteration. Generation for an iteration is considered complete if the output JSONL file exists and contains the expected number of `prompt_id`s.

## 📂 Output Structure

Outputs are saved in `experiment_base_dir` (e.g., `results/auto_antislop_runs/`), under a timestamped directory for each run (e.g., `run_YYYYMMDD_HHMMSS/`):

*   `run_config_YYYYMMDD_HHMMSS.yaml`: The exact configuration used for this run.
*   `iter_N_creative_writing_generations.jsonl`: Raw generated text for iteration `N`.
*   `iter_N_ftpo_pairs.jsonl`: (If FTPO mode is active) Fine-grained preference pairs for iteration `N`.
*   `iter_N_analysis_results/`: Directory containing:
    *   `bigrams__dictionary_sorted.csv`, `trigrams__non_dictionary_sorted.csv`, etc.: N-gram analysis results.
    *   `overrepresented_words.csv`: Analysis of over-represented words.
    *   `slop_list_phrases.jsonl` (inside `phrase_tmp/`): Candidate slop phrases from `slop-forensics`.
    *   `banned_ngrams_used.json`, `banned_slop_phrases_used.json`: Copies of ban lists *used* for this iteration's generation (for iter > 0).
    *   `banned_ngrams_new_this_iter.json`, `banned_slop_phrases_new_this_iter.json`: Ban list entries *added* after this iteration's analysis.
    *   `orchestration.log`: Log specific to analysis and ban list updates for this iteration.
*   `banned_ngrams.json`: Aggregated list of banned n-grams across iterations.
*   `banned_slop_phrases.json` (or custom name): Aggregated list of banned slop phrases.
*   `user_defined_regex_blocklist.json`: Copy of user-defined regex patterns.
*   `dpo_pairs_dataset.jsonl`: The final preference dataset for DPO/FTPO.
*   `final_iteration_statistics.csv`: Summary metrics for each iteration.
*   `finetuned_model_SUFFIX/`: (If finetuning is run)
    *   `lora_adapters/`: Saved LoRA adapter weights and tokenizer config.
    *   `merged_16bit/`: (If `finetune_save_merged_16bit: true`) Full model with LoRA weights merged, in 16-bit precision.
    *   `gguf_q8_0.gguf`: (If `finetune_save_gguf_q8_0: true`) GGUF quantized model.
    *   `logprob_gap_analysis/`: (If FTPO mode) JSONL files with pre/post training logprob gap statistics.

## 🧪 Post-Finetuning: Testing the Model

A simple script `test_inference.py` is provided to load the latest finetuned model and run a test generation.

```bash
python test_inference.py
```
This script automatically searches for the most recent `merged_16bit` model in the standard output directories. You can modify the prompt within the script.

##🧩 Submodules

*   **`antislop-vllm`**: (Path: `antislop-vllm/`)
    *   Handles the core text generation using vLLM.
    *   Crucially, it implements the logic for dynamic banning of n-grams, phrases, and regex patterns during the generation process.
*   **`slop-forensics`**: (Path: `slop-forensics/`)
    *   Provides tools and algorithms for analyzing text to identify various types of "slop," including over-represented n-grams and common undesirable phrases.

## 💡 Notes & Troubleshooting

*   **GPU Memory:** Running vLLM and finetuning (especially with larger models) requires significant GPU VRAM. Adjust `vllm_gpu_memory_utilization` and finetuning batch sizes/quantization accordingly. If running both on the same GPU, the script attempts to stop vLLM before finetuning to free up VRAM.
*   **Submodule Issues:** If you encounter errors related to `antislop-vllm` or `slop-forensics`, ensure the submodules are correctly initialized (`git submodule update --init --recursive`).
*   **NLTK Data:** If `ensure_core_nltk_resources()` fails, download the resources manually as described in Installation.
*   **Unsloth Cache:** Unsloth might create a `unsloth_compiled_cache` directory. This is ignored by git.
*   **Gemma-3 Checkpoints:** The `utils/model_helpers.py` contains a `fix_gemma3_checkpoint` function to handle potential inconsistencies in Gemma-3 model key naming, and `detie_lm_head` to ensure proper saving of merged models.
*   **FTPO Mode:** The "ftpo" (Final Token Preference Optimization) mode uses `FTPOTrainer` which focuses on the preference for a single next token, given a context. This is useful for correcting specific token choices rather than entire continuations.
