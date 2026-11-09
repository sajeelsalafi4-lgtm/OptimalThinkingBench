# OptimalThinkingBench: Adaptive Reasoning Model (VS Code)

**Research:** [OptimalThinkingBench (arXiv:2508.13141)](https://arxiv.org/abs/2508.13141)
**Dataset:** [facebook/optimal_thinking_bench](https://huggingface.co/datasets/facebook/optimal_thinking_bench)
**Original project:** [facebookresearch/RAM projects/otb](https://github.com/facebookresearch/RAM/tree/main/projects/otb)

## What this project does

1. Downloads the official public **subset** (1,877 rows; NOT the complete benchmark).
2. Converts benchmark subset names into *easy* vs *hard* **proxy difficulty labels**.
3. Removes exact duplicate questions, then makes held-out train/validation/test splits.
4. Trains a TF-IDF + Logistic Regression difficulty router.
5. Uses **Qwen/Qwen3-0.6B** in `enable_thinking=False` or `True` mode, determined by that router.
6. Compares adaptive routing with always-thinking and never-thinking strategies.
7. Records predictions, approximate thinking-token counts, response time, and **proxy** benchmark metrics. Makes a comparison chart.

**Important scientific limitations:** Benchmark subset is a difficulty label, not an observed classification of overthinking/underthinking. The text-matching scorer is **not** the paper's LLM semantic evaluator or Reasoning Gym programmatic task verifier. The available public dataset omits OvT-Math and UT-Math. Numbers from this project **must not be presented as official benchmark reproductions**. Exact-question deduplication does not necessarily prevent template-level leakage.

## Recommended setup

- Windows with VS Code, Python **3.11 or 3.12**.
- A working Internet connection for first-run dataset and model downloads.
- CPU works but LLM inference may be very slow; an NVIDIA GPU is strongly recommended.
- Qwen3-0.6B is about 0.6B parameters; downloading weights requires available disk/RAM.

Open the folder in VS Code, then open **Terminal → New Terminal**.

### Step 1: Set up a virtual environment (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, open a **Command Prompt** terminal in VS Code and use:

```cmd
.venv\Scripts\activate.bat
```

If Python 3.12 is not installed, install it, or use a currently installed 3.11 version. In VS Code select **Python: Select Interpreter** and choose `.venv`.

### Step 2: Download/split the real dataset

```powershell
python prepare_data.py
```

Outputs: `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl`.

If you already have the dataset `otl_bench.jsonl`, copy it into this project directory and run:

```powershell
python prepare_data.py --local-jsonl otl_bench.jsonl
```

The metadata stays with examples. We split by distinct question to prevent *exact* duplicated questions from leaking between train and test. Very similar templates may still appear in different splits.

### Step 3: Train the difficulty-routing model

```powershell
python train_router.py
```

Outputs: `artifacts/difficulty_router.joblib` and `artifacts/router_metrics.json`.

### Step 4: Install the optional LLM runtime

```powershell
python -m pip install -r requirements-llm.txt
```

For a GPU, follow the official PyTorch installation instructions at https://pytorch.org/get-started/locally/ and ensure `torch.cuda.is_available()` returns `True`. The generic requirements may install a CPU-only wheel depending on platform.

### Step 5: Ask an easy question

```powershell
python ask.py --question "What is 5 + 5?"
```

### Step 6: Ask a difficult question

```powershell
python ask.py --question "Find all integer solutions of x squared minus 5 times y squared equals 1 with x and y positive and x less than 200."
```

Or force one mode for comparison:

```powershell
python ask.py --question "What is 5 + 5?" --force no_think
python ask.py --question "What is 5 + 5?" --force think --max-new-tokens 2048
```

The router can make mistakes; classifying a prompt as hard does **not** guarantee the LLM will solve it. If a thinking response uses up all output tokens before producing a final answer, the script reports it as incomplete (rather than pretending it's correct).

### Step 7: Evaluate on held-out questions

```powershell
python evaluate.py --per-subset 3 --strategy all --max-new-tokens 1024
```

Runs the **same question sample** with `adaptive`, `always_think` and `never_think`; caches generated responses and creates:

- `results/comparison.csv`
- `results/detailed_predictions.csv`
- `results/comparison.png`
- `results/inference_cache.jsonl` for reuse

For CPU or low-memory machines, start with:

```powershell
python evaluate.py --per-subset 1 --strategy adaptive --max-new-tokens 512
```

For more informative experiments, use more questions and allow longer thinking:

```powershell
python evaluate.py --per-subset 30 --strategy all --max-new-tokens 4096
```

### Step 8: Run unit tests

```powershell
python -m unittest test_project -v
```

## How the research metrics are approximated

On *easy* benchmark samples, overthinking-adjusted accuracy at threshold `t` is:

`OAA(t) = (1/N) * sum[ correct_i * I(thinking_tokens_i < t) ]`

The normalized continuous area under this curve (with `t_max=1000`) is:

`AUC_OAA = mean[correct_i * max(0, 1 - thinking_tokens_i / 1000)]`

On *hard* benchmark samples, `Acc_under = mean(correct_i)`.

`F1_OTB = 2 * AUC_OAA * Acc_under / (AUC_OAA + Acc_under)`.

In this educational implementation, `correct_i` is determined by a deliberately simple conservative **string/numeric answer matcher**, not the official judges. The results are therefore called `*_proxy`. The paper used a semantic LLM judge and task-specific verifiers; to reproduce paper numbers, use the authors' official evaluation stack and full dataset.

## Folder guide

```text
OptimalThinkingBench_VSCode/
├── README.md
├── requirements.txt
├── requirements-llm.txt
├── prepare_data.py      # Load, normalize, deduplicate, split
├── train_router.py      # Train real classifier
├── llm_backend.py       # Run real Qwen3 model in 2 modes
├── ask.py               # Interactive inference
├── metrics.py           # Proxy accuracy, OAA, F1
├── evaluate.py          # Compare 3 strategies and plot
├── test_project.py      # Runs offline without LLM
├── data/                # Prepared data after execution
├── artifacts/           # Learned router and metrics
└── results/             # Scores, predictions and charts
```

## Expected outcomes (not fabricated results)

After training, the router prints its measured validation/test accuracy. After inference, the LLM prints its final answer, thinking token count, and runtime. After evaluation, it saves actual CSV scores from your own run. Scores depend on your downloaded model, random sampling, generation length and the approximate grader.

## Citation

Aggarwal et al. (2025), *OptimalThinkingBench: Evaluating Over and Underthinking in LLMs*, arXiv:2508.13141. Public benchmark subset under the dataset's Research License; check its terms before using it beyond coursework.
