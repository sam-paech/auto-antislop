# Auto-Antislop

Auto-Antislop is an automated pipeline which takes a model and does the following:

1. Generates a baseline dataset on a set of prompts that you specify
2. Identifies the model's slop (over-represented words, phrases & n-grams)
3. Using [antislop-vllm](https://github.com/sam-paech/antislop-vllm) generates a preference dataset for fine-tuning
4. Fine tunes the model on the generated preference dataset using a novel trainer (FTPO: final token preference optimisation)


## ⚙️ How It Works (Pipeline Flow)

1.  **vLLM Server Management:** If `manage_vllm` is true, the script starts a vLLM server. If a server is already running on the configured port, or if `manage_vllm` is false, the script assumes an external vLLM server.
2.  **Antislop Iteration Loop (`num_iterations` times):**
    *   **Generation (Iter 0 - Baseline):**
        *   The `antislop-vllm` script generates text from a source dataset (e.g., writing prompts) *without* any ban lists active. This forms the baseline dataset.
    *   **Analysis (All Iterations):**
        *   The generated text is analyzed using tools from `slop-forensics` and custom analysis scripts:
            *   N-gram frequencies are compared against a human writing profile.
            *   Over-represented words are identified.
            *   Common "slop phrases" are extracted.
    *   **Ban List Update (All Iterations):**
        *   Based on the analysis, n-gram and slop phrase ban lists are created or updated. User-supplied `extra_ngrams_to_ban`, `extra_slop_phrases_to_ban`, and `extra_regex_patterns` from the config are merged.
    *   **Generation (Iter 1+ - Anti-Slop):**
        *   `antislop-vllm` generates text again on the same prompts, but this time it uses the accumulated ban lists (n-grams, slop phrases, regex) to avoid slop.
        *   While generating, preference pairs are produced each time a ban occurs, containing a *rejected* token and a number of coherent alternative *chosen* tokens, as well as the preceding context.
3.  **FTPO Dataset Creation:**
    *   Pairs are created:
        *   `prompt`: The original input prompt + generated context so far.
        *   `rejected`: The first token in a banned sequence, at the time a ban occurred during antislop generation.
        *   `chosen`: A number of coherent alternative tokens at that position, constrained by min_p.
4.  **FTPO Finetuning:**
    *   If `finetune_enabled` is true, the script runs FTPO finetuning using the preference dataset.
    *   Supports LoRA and optional 4-bit quantization
    *   Supports unsloth or transformers/trl training paths (though some models may not work with both)
    *   Saves the LoRA adapters and optionally a merged 16-bit model.


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

The primary configuration is done through a YAML file. Refer to the examples provided in `configs/`.

While the pipeline is ostensibly automatic end to end, there are a lot of options you can tweak in the config file, some of which are important for a successful result. Let's walk through some of them:

#### VLLM & Antislop Generation Parameters

*   **`model_id`**: The model you want to unslop. Can be a huggingface id or a local dir.
*   **`vllm management`:**: The pipeline can launch vllm automatically with the settings you provide; alternatively you can provide an openai compatible endpoint.
*   **`generation_threads`:** How many parallel threads to generate with (set according to your system specs; if in doubt try 30).
*   **`generation_max_prompts`:** The number of prompts to generate a response for. Set to 50 for a quick test. Recommmended is 1000-2000 for good coverage.
*   **`generation_hf_dataset_name`:** The huggingface dataset used to source prompts (expects sharegpt format). The kinds of prompts you use determines the slop that will be removed from the model in fine tuning.
*   **`extra_slop_phrases_to_ban`:** Set your own slop list to ban. The strings you add here will be trained out of the model.

#### Fine-Tuning Parameters

*   **`finetune_use_unsloth`:** Supports unsloth or transformers/trl. Unsloth has lower vram usage but it doesn't work with all modesl in this pipeline.
*   **`finetune_mode`:** Set to "ftpo" or "dpo". FTPO is our trainer implemented specifically for the preference dataset we generate in this pipeline. It's more surgical in training out the slop words without impacting the weights otherwise, compared to DPO.
*   **`finetune_early_stopping_wins`:** This stops training when "chosen" tokens are preferred more than "rejected" tokens by this fraction. Early stopping is important to avoid overtraining. We find a good number is 0.8-0.85. Some models will degrade more easily than others, in which case you can try a lower stopping threshold.
*   **`finetune_lora_r`:** FTPO works best with a high lora rank (128-256), likely much higher than you are used to. This is because we are trying to do surgical updates of weights, and a high rank means less collateral damage on unrelated weights. Feel free to experiment; ymmv.
*   **`finetune_target_modules`:** The models we target in fine tuning seems to be model dependent in terms of what works best. Check out the training recipes in `configs/`, or try just `["lm_head"]` for a minimally invasive fine-tune.
*   **`finetune_learning_rate`:** Set the learning rate manually.
*   **`finetune_auto_learning_rate`:** OR use an automatic learning rate that adjusts to dataset size, batch size & lora rank. Adjustable via `finetune_auto_learning_rate_adjustment_scaling`.
*   **`finetune_max_train_examples`:** The number of training examples. Suggest 8000-12000.

#### FTPO Parameters

These params control the final token preference optimisation trainer. The defaults are probably fine for a first pass. See below for a breakdown of FTPO and the configurable parameters.

## 🛠️ Usage

### Quickstart:

Once your .yaml is set up:

```
python main.py --config myconfig.yaml
```

### Commandline Args

**Key Command-Line Arguments (override config settings):**
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

**Resuming Runs:**
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


### 📖 FTPO Explained

FTPO (Final-Token Preference Optimisation) is a surgical preference optimisation training algorithm that constrains gradient updates to just a rejected/chosen *continuation token*, and avoids training on the preceding context. The intent is to push probability mass **away from the first token of a banned phrase (the *rejected* token)** and **toward one or more viable alternatives (the *chosen* tokens)** while leaving the rest of the model distribution largely intact.

---

#### 1. How a training example is created in the Auto-Antislop pipeline

1. **Generation runs with antislop active.**
   In the auto-antislop pipeline, the FTPO dataset is generated in iterations > 0, when antislop is actively banning slop that it surfaced during the first iteration. Whenever the sampler encounters a banned n-gram / phrase / regex, it halts and constructs a training example before resuming inference with a non-banned continuation.

2. **Rejected token.**
   The would-have-been next token (the first token of the banned phrase) is stored as the `rejected` continuation token.

3. **Chosen tokens.**
   The sampler then draws further candidates for that same position, applying a *min-p* filter ¹ to keep only continuations whose tail probability mass is above a given threshold, to ensure they are coherent. These candidates are further filtered per the banned phrases list. The remaining tokens are stored as the `chosen` continuation tokens in the sample.

   * If no alternative passes the filter the event is discarded and no FTPO sample is written.

4. **Context.**
   The full prompt (and any chat template markers) up to but **not including** the banned token is stored.
   This means the model receives identical context for both the rejected and chosen tokens.

Result: a single JSONL line contains the shared context plus one rejected token and a small set of chosen tokens — exactly the information FTPO needs.

---

#### 2. Loss formulation

* **Preference term.**
  For each example the trainer computes
  `Δ = logit(chosen) − logit(rejected)` (or the equivalent in probability space).
  The loss is `−log σ(β Δ)` in “free” mode or a clipped variant in “clip” mode. Or in plain English, the amount that all the chosen logits are beating the rejected, averaged across chosen tokens for that sample.
  This encourages the model to rank *every* chosen token above the rejected one.
  Once a chosen logit is beating rejected by a given margin, it no longer contributes to the loss. This helps to avoid unnecessarily moving the weights when chosen is already winning.

* **Two-part KL regulariser.**
  A key part of the loss function is the KL loss, which is split into two terms:

  ```
  λ_kl · KL_non-target + λ_kl_target · KL_target
  ```

  Where "target" refers to the chosen & rejected logits.

   * **KL\_target**: We need the target logits to move significantly, in order to suppress the top "slop" logit and let the other candidates win. So they cannot be naively constrained to the reference, like they would be with ordinary KL loss. Instead, we define this `kl_target` term as the *mean* divergence from reference of the chosen + rejected set. This allows these logits more freedom to move relative to each other, while still being overall anchored to the reference.
   The target logits are also allowed some grace to move before kl loss kicks in, per the `tau_kl_target` parameter. So small logit shifts in the target tokens are free but large shifts are penalised quadratically.

   * **KL\_non-target**: This loss term applies exclusively to the rest of the vocabulary (all tokens not in {chosen ∪ rejected}). This is the traditional kl loss idea from DPO which constrains those logits to the reference.

  This split lets the model reorder the chosen/rejected logits relative to each other while avoiding unwanted drift elsewhere.


#### 3. FTPO Tunable hyper-parameters

* **`loss_mode`**
  `"clip"` (default) or `"free"`.

  * **clip** – applies the epsilon thresholds below so that, once a chosen token is comfortably ahead of the rejected one, its contribution to the loss is capped.
  * **free** – uses the raw `−log σ(β Δ)` everywhere; stronger gradients but higher risk of overshooting.

* **`beta`** *(float, default 0.1)*
  Scales the preference term. Smaller values dampen gradients; larger values make the model push logit differences more aggressively.

* **`LOSS_CALC_MODE`**
  `"logits"` (default) or `"probs"`.

  * **logits** – works directly in logit space; only the local logits move, leaving the softmax normalisation unchanged for the rest of the vocab.
  * **probs** – converts to probabilities first; gentler, but every token’s prob is indirectly affected by the softmax.

* **`clip_epsilon_logits`** *(float, default 2)*
  When in `"logits"` mode and `loss_mode="clip"`, the loss stops growing once `logit(chosen) − logit(rejected)` exceeds this value.

* **`clip_epsilon_probs`** *(float, default 0.2)*
  Analogous threshold for `"probs"` mode, defined on probability differences.

* **`lambda_kl`** *(float, default 0.4)*
  Weight of **KL\_non-target** – keeps all *other* vocabulary logits near the reference model.

* **`lambda_kl_target`** *(float, default 0.1)*
  Weight of **KL\_target** – steadies the *mean* of the chosen + rejected set.

* **`tau_kl_target`** *(float, default 2.0)*
  Grace band for **KL\_target**. Shifts of ≤ `tau_kl_target` are free; beyond that the quadratic penalty kicks in.

 

---

¹ *min-p sampler: see* *Nguyen et al., “Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs” (arXiv:2407.01082).*



## 💡 Notes & Troubleshooting

*   **GPU Memory:** Running vLLM and finetuning (especially with larger models) requires significant GPU VRAM. Adjust `vllm_gpu_memory_utilization` and finetuning batch sizes/quantization accordingly. If running both on the same GPU, the script attempts to stop vLLM before finetuning to free up VRAM.
*   **Submodule Issues:** If you encounter errors related to `antislop-vllm` or `slop-forensics`, ensure the submodules are correctly initialized (`git submodule update --init --recursive`).
*   **NLTK Data:** If `ensure_core_nltk_resources()` fails, download the resources manually as described in Installation.
*   **Unsloth Cache:** Unsloth might create a `unsloth_compiled_cache` directory. This is ignored by git.
*   **Gemma-3 Checkpoints:** The `utils/model_helpers.py` contains a `fix_gemma3_checkpoint` function to handle potential inconsistencies in Gemma-3 model key naming, and `detie_lm_head` to ensure proper saving of merged models.
*   **FTPO Mode:** The "ftpo" (Final Token Preference Optimization) mode uses `FTPOTrainer` which focuses on the preference for a single next token, given a context. This is useful for correcting specific token choices rather than entire continuations.
