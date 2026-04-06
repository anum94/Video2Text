
# Video2Text: Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches

Official Repository for the paper: https://arxiv.org/pdf/2603.02655 accepted/published at LREC 2026.

This repository supports both inference (using In-Context Learning) and fine-tuning for video-to-text tasks, specifically optimized for long-form content like race commentary.


## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/anum94/Video2Text.git](https://github.com/anum94/Video2Text.git)
   cd Video2Text
### Install dependencies:
(Ensure you have Python 3.8+ and FFmpeg installed)

```bash
pip install -r requirements.txt
```
## Usage
# Inference & Processing
The main.py script is used to process video directories. You can run it with default settings or provide detailed arguments for context and sampling.

Basic Command:

```bash
python main.py "path/to/data"
```
## Advanced Example (with ICL and custom window):

```bash
python main.py --dir "/groups/gac50547/RaceCommentary" --n 1 --icl True --step 20 --k 4 --frames 1 --context_window 4096 --wb False
```
## Arguments:

--dir: Path to the video directory.

--n: Number of samples/iterations.

--icl: Enable/Disable In-Context Learning (True/False).

--step: Step size for frame/sequence sampling.

--k: Number of examples for few-shot prompting.

--frames: Number of frames to extract per segment.

--context_window: Token limit for the model's context window.

--wb: Enable/Disable Weights & Biases logging.

## Fine-tuning
Use finetune.py to train the model on your specific dataset.

Example:

```bash
python finetune.py --dir "/groups/gac50547/RaceCommentary/" --frames 2 --step 2 --n 2000
```
Arguments:

--dir: Path to the training dataset.

--frames: Number of frames per training instance.

--step: Sampling step size.

--n: Total number of training steps or samples.
