<div align="center">

# From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://husky-morocco-f72.notion.site/From-f-x-and-g-x-to-f-g-x-LLMs-Learn-New-Skills-in-RL-by-Composing-Old-Ones-2499aba4486f802c8108e76a12af3020)  [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/PRIME-RL/RL-Compositionality) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/lifan__yuan/status/1963662222602723673)
</div>

📘 This repository contains the code accompanying the paper **"FROM $f(x)$ AND $g(x)$ TO $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones"**. The repo is built upon [veRL](https://github.com/volcengine/verl), and we provide the synthetic data generators and training pipelines required to reproduce results reported in the paper.

⚡ The repository is organized around reproducible bash entrypoints located in [`bash/`](bash), grouped by the paper sections they support.

---

## 🗂️ Table of Contents
- [⚙️ Environment Setup](#️-environment-setup)
- [📁 Repository Layout](#-repository-layout)
- [🧑‍🏫 Stage 1: Atomic Skill Acquisition](#-stage-1-atomic-skill-acquisition)
- [🧩 Stage 2: Learning Compositional Skills](#-stage-2-learning-compositional-skills)
  - [🤖 RL Variants (Sections 4.1 & 4.2)](#-rl-variants-sections-41--42)
  - [📊 RFT Baseline (Section 4.2)](#-rft-baseline-section-42)
- [🔀 Cross-Task Transfer to Countdown (Section 4.3)](#-cross-task-transfer-to-countdown-section-43)
- [📈 Pass@k Analysis (Sections 4.4)](#-passk-analysis-sections-44)
- [📚 Citing](#-citing)

---

## ⚙️ Environment Setup

1. **Clone the repository** 📥
   ```bash
   git clone https://github.com/PRIME-RL/RL-Compositionality.git
   cd RL-Compositionality
   ```

2. **Create a Python environment** 🧑‍💻 (Python ≥ 3.10 recommended, tested with 3.12)

   ```bash
   virtualenv rl_comp
   source rl_comp/bin/activate
   pip install -e .
   pip install flash-attn --no-build-isolation
   ```

3. **Login to Wandb** 🔑

   ```bash
   export WANDB_API_KEY="..."
   ```

4. **Model checkpoints** 📂
   Update `MODEL_PATH` variables inside the bash scripts if your checkpoint is stored elsewhere.

---

## 📁 Repository Layout

* [`bash/`](bash): One-click pipelines grouped by paper section numbers.

  * [`section41_42/`](bash/section41_42): Data generation, Stage 1 RFT, Stage 2 RL/RFT experiments.
  * [`section43/`](bash/section43): Countdown data generation and transfer experiments.
  * [`section44/`](bash/section44): Pass@k evaluation utilities.
* [`examples/`](examples): Python entrypoints used by the bash scripts.

---

## 🧑‍🏫 Stage 1: Atomic Skill Acquisition

Stage 1 corresponds to rejection fine-tuning (RFT) on tasks where the function definitions are visible. The process produces a Stage 1 checkpoint that serves as the initialization for all Stage 2 experiments.

1. **Generate synthetic training data**

   ```bash
   bash bash/section41_42/stage1_create_problems.sh
   ```

   This script populates `data/string_task/stage1_level1/` with train and test Parquet files containing atomic string transformations (the test Parquet is actually not used in the experiment).

2. **Collect rollouts & convert into RFT datasets**

   ```bash
   bash bash/section41_42/stage1_create_train_data.sh
   ```

   The script samples `N_SAMPLES` responses per prompt and stores them in `data/string_task/stage1_level1/rollout.parquet`. And then filters the rollouts based on the accuracy and emits `train.parquet` / `test.parquet` splits under `data/string_task/stage1_level1/rft_data/`.


3. **Train the Stage 1 RFT model**

   ```bash
   bash bash/section41_42/stage1_rft.sh
   ```

   The resulting checkpoint is saved to `checkpoints/string-task/stage1-rft/` and initializes every Stage 2 run.

---

## 🧩 Stage 2: Learning Compositional Skills

Stage 2 removes access to function implementations and focuses on compositional reasoning. **All scripts assume that `bash/section41_42/stage1_rft.sh` has been executed successfully.**


### 🤖 RL Variants (Sections 4.1 & 4.2)

1. **Create compositional problem sets**

   ```bash
   bash bash/section41_42/stage2_create_problems.sh
   ```

   ➡️ Generates Level-1, Level-2, Level-1-to-2 training splits and a Level-1-to-8 evaluation split.

2. **Launch RL training with your desired setting**:

   ```bash
   bash bash/section41_42/stage2_rl_level1.sh      # Level-1 only
   bash bash/section41_42/stage2_rl_level2.sh      # Level-2 only
   bash bash/section41_42/stage2_rl_level1to2.sh   # Mixed Level-1+2
   ```

### 📊 RFT Baseline (Section 4.2)

To compare RL with supervised rejection fine-tuning:

1. **Partition Level-2 data into iterative RFT chunks**

   ```bash
   bash bash/section41_42/stage2_rft_create_problems.sh
   ```

2. **Collect rollouts & convert them into RFT datasets**

   ```bash
   bash bash/section41_42/stage2_rft_create_train_data.sh
   ```

3. **Fine-tune with Stage 1 checkpoint**

   ```bash
   bash bash/section41_42/stage2_rft_train.sh
   ```

4. **Iterative RFT** 🔁
   Repeat steps 2–3 with updated model/data paths.

---

## 🔀 Cross-Task Transfer to Countdown (Section 4.3)

1. **Generate Countdown arithmetic datasets** 

   ```bash
   bash bash/section43/create_countdown_data.sh
   ```

2. **Collect rollouts & convert to RFT data**

   ```bash
   bash bash/section43/stage1_collect_train_data.sh
   ```

   The script will collect model's rollout on Countdown Level 2 problems, filter according to the accuracy, merge them with string task RFT data, and save to `data/string_countdown_task/stage1_rft_data/`

3. **Train the model** 🏋️
   You can reuse the script from `bash/section41_42/stage1_rft.sh`, and change the data path to `data/string_countdown_task/stage1_rft_data`.

The Stage 2 training is the same as Section 4.1/4.2. You can reuse the scripts and change the model path.

---

## 📈 Pass@k Analysis (Sections 4.4)

1. **1000 Rollout Collection**

   ```bash
   bash bash/section44/passk.sh
   ```

   This command samples completions from a trained policy, saves them to `results/stage2_rl_level1/all.parquet`, and enables downstream computation of pass@k metrics. Change the model path to obtain the 1000 responses from other models.

---

## 📚 Citing

If you build upon this work, please cite the accompanying paper:

```bibtex
@misc{yuan2025llms,
  author       = {Lifan Yuan and Weize Chen and Yuchen Zhang and Ganqu Cui and Hanbin Wang and Ziming You and Ning Ding and Zhiyuan Liu and Maosong Sun and Hao Peng},
  title        = {From $f(x)$ and $g(x)$ to $f(g(x))$: {LLMs} Learn New Skills in {RL} by Composing Old Ones},
  year         = {2025},
  note         = {Notion blog post, available online},
  howpublished = {\url{https://husky-morocco-f72.notion.site/From-f-x-and-g-x-to-f-g-x-LLMs-Learn-New-Skills-in-RL-by-Composing-Old-Ones-2499aba4486f802c8108e76a12af3020}},
}
```

