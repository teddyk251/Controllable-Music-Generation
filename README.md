# Compositional Music Generation using Textual Inversion

This repository explores Textual Inversion (TI) for guiding music generation using MusicGen. We inject learned genre-specific tokens into MusicGen’s language model (LM) embedding space and analyze their effect on controllability and alignment using CLAP similarity metrics.


### Project Overview

We propose a compositional textual inversion framework that learns new token embeddings (<lofi>, <jazz>, etc.) from short audio clips. These tokens can be inserted into music prompts to guide MusicGen to generate music in specific genres. We compare the results with baseline generations that do not use TI.


### Setup

1. Clone and Install

```
git clone https://github.com/teddyk251/Controllable-Music-Generation.git
cd Controllable-Music-Generation
conda create -n musicgen python=3.10
conda activate musicgen
pip install -r requirements.txt
```

2. Install FFMPEG (if not already)

```sudo apt-get install ffmpeg  # or use homebrew on Mac```

### Directory Structure

<pre><code>
├── scripts/
│   ├── train.py           # Train TI tokens per genre
│   ├── inference.py       # Generate music with learned tokens
│   ├── evaluate.py        # CLAP similarity evaluation
│   ├── app.py             # Gradio interface for generation
├── trained_tokens/        # Learned embeddings per genre
├── preprocessed_data_2/   # Audio clips for training (150s)
├── gradio_generated/      # Generated audio clips via Gradio
├── wandb/                 # Experiment logs (if W&B used)
├── results/
│   ├── clap_scores.json   # Evaluation results
│   └── similarity_plot.png
├── requirements.txt
├── README.md
</code></pre>



### Training

#### Train TI Tokens for All Genres

```python scripts/train.py```

	•	Learns one embedding per codebook (4 total) per genre.
	•	Stores them in trained_tokens/{genre}/{genre}_codebook{i}.pt
	•	Logs training loss and learning rate via Weights & Biases

#### Configuration

Modify these in train.py:
```
DATA_DIR = "preprocessed_data_2"
NUM_EPOCHS = 600
LEARNING_RATE = 1e-4
WANDB_PROJECT = "musicgen-textual-inversion_150s"
```





### Inference

Generate music using a trained TI token:

```python scripts/inference.py```

Edit these in inference.py:

```GENRE_NAME = "afrobeat"
LEARNED_TOKEN = "<afrobeat>"
PROMPT = f"A soulful {LEARNED_TOKEN} beat with vibrant percussion"
```

Output is saved to generated_{GENRE_NAME}.wav.



### Evaluation

Evaluate CLAP similarity between generated audio and text prompts:

```python scripts/evaluate.py```

	•	Generates both baseline (natural prompt) and TI (injected token) music.
	•	Computes CLAP similarity against the text prompt.
	•	Outputs scores and bar chart to results/similarity_plot.png.



### Gradio App (Interactive Demo)

Launch an interactive app:

```python scripts/multiselect_gradio.py```

	•	Input your prompt
	•	Select whether to use TI
	•	Choose a token if TI is enabled (You can choose multiply styles to compose a mix of different styles)
	•	Audio preview with playback



### Experimental Highlights

| Genre     | TI Prompt Example                                | Key Observations                            |
|-----------|--------------------------------------------------|---------------------------------------------|
| lofi      | A mellow `<lofi>` beat with nostalgic textures   | TI slightly improves mood alignment         |
| jazz      | A soulful `<jazz>` track with saxophone          | TI significantly boosts CLAP similarity     |
| afrobeat  | An energetic `<afrobeat>` groove with drums      | TI outperforms baseline in genre fidelity   |




How It Works
	•	Text prompts are passed through T5 encoder (frozen)
	•	Prompt embeddings go into the MusicGen LM
	•	The LM outputs codebook token IDs representing audio
	•	Encodec decoder reconstructs the waveform
	•	Our learned TI tokens are injected into the LM embedding space, not into the text encoder



Notes
	•	Uses MusicGen-Large (4 codebooks, 2048-dim embeddings)
	•	Encodec is used only during training and waveform decoding
	•	TI tokens are not recognized by the text encoder



### Citation

If this project helps your work, please consider citing MusicGen:

@article{copet2023simple,
  title={Simple and Controllable Music Generation},
  author={Copet, Jade and Défossez, Alexandre and Plantinga, Peter and others},
  journal={arXiv preprint arXiv:2306.05284},
  year={2023}
}
