
# 🎮 Video2Text: Real-Time Game Video Commentary
### Pause-Aware Decoding Approaches with Multimodal LLMs

[![Paper](https://img.shields.io/badge/arXiv-2603.02655-B31B1B.svg)](https://arxiv.org/pdf/2603.02655)
[![Conference](https://img.shields.io/badge/LREC-2026-blue)](https://lrec-coling-2026.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](#)

Official Repository for the paper: **"Video2Text: Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches"**, accepted at **LREC 2026**.

This repository supports both inference (using In-Context Learning) and fine-tuning for video-to-text tasks, specifically optimized for long-form content like race commentary.

---

## Setup

### 1. Prerequisites
* **Python:** 3.8+
* **System Tools:** [FFmpeg](https://ffmpeg.org/) (Required for video frame extraction)

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/anum94/Video2Text.git](https://github.com/anum94/Video2Text.git)
cd Video2Text
````
Install dependencies
```bash
pip install -r requirements.txt
````
---

## Usage

### 1. Inference & Processing
The `main.py` script is used to process video directories. You can run it with default settings or provide detailed arguments for context and sampling.

**Basic Command:**
```bash
python main.py "path/to/data"
```

**Advanced Example (with ICL and custom window):**
```bash
python main.py --dir "/groups/gac50547/RaceCommentary" --n 1 --icl True --step 20 --k 4 --frames 1 --context_window 4096 --wb False
```

#### Arguments:
* `--dir`: Path to the video directory.
* `--n`: Number of samples/iterations.
* `--icl`: Enable/Disable In-Context Learning (True/False).
* `--step`: Step size for frame/sequence sampling.
* `--k`: Number of examples for few-shot prompting.
* `--frames`: Number of frames to extract per segment.
* `--context_window`: Token limit for the model's context window.
* `--wb`: Enable/Disable Weights & Biases (W&B) logging.

---

### 2. Fine-tuning
Use `finetune.py` to train the model on your specific dataset.

**Example:**
```bash
python finetune.py --dir "/groups/gac50547/RaceCommentary/" --frames 2 --step 2 --n 2000
```

#### Arguments:
* `--dir`: Path to the training dataset.
* `--frames`: Number of frames per training instance.
* `--step`: Sampling step size.
* `--n`: Total number of training steps or samples.

---

## 📝 Methodology Overview
The core of this work focuses on **Pause-Aware Decoding**. Unlike traditional video-to-text models that generate continuous captions, our approach leverages Multimodal LLMs to detect and respect the natural pauses inherent in live sports and gaming commentary, resulting in more realistic and contextually grounded output.

---

## ✍️ Citation

If you use this code or our paper's findings in your research, please cite:

```bibtex
@inproceedings{afzal2026video2text,
  title={Video2Text: Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches},
  author={Afzal, Anum and Saito, Yuki and Takamura, Hiroya and Sudoh, Katsuhito and Takamichi, Shinnosuke and Neubig, Graham and Matthes, Florian and Ishigaki, Tatsuya},
  booktitle={Proceedings of the Language Resources and Evaluation Conference (LREC)},
  year={2026}
}
```
