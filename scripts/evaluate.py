import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from audiocraft.models import MusicGen
from transformers import ClapProcessor, ClapModel
from scipy.spatial.distance import cosine

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = "facebook/musicgen-large"
SAMPLE_RATE = 32000
DURATION = 10
OUTPUT_DIR = "evaluation_outputs"
TI_TOKENS_DIR = "trained_tokens"
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

GENRES = {
    "lofi": {
        "ti_token": "<lofi>",
        "paraphrased_prompt": "A relaxing mellow lo-fi beat with nostalgic melodies",
        "ti_prompt": "A relaxing mellow <lofi> beat with nostalgic melodies",
    },
    "hiphop": {
        "ti_token": "<hiphop>",
        "paraphrased_prompt": "A high-energy old school hip hop instrumental",
        "ti_prompt": "A high-energy old school <hiphop> instrumental",
    },
    "pop": {
        "ti_token": "<pop>",
        "paraphrased_prompt": "An upbeat pop dance track with catchy vocals",
        "ti_prompt": "An upbeat <pop> dance track with catchy vocals",
    },
    "reggae": {
        "ti_token": "<reggae>",
        "paraphrased_prompt": "A 1970s-style reggae groove with soulful influence",
        "ti_prompt": "A 1970s-style <reggae> groove with soulful influence",
    },
    "country": {
        "ti_token": "<country>",
        "paraphrased_prompt": "A heartfelt country acoustic guitar ballad",
        "ti_prompt": "A heartfelt <country> acoustic guitar ballad",
    },
    "jazz": {
        "ti_token": "<jazz>",
        "paraphrased_prompt": "A smooth jazz instrumental with soulful saxophone",
        "ti_prompt": "A smooth <jazz> instrumental with soulful saxophone",
    },
    "afrobeat": {
        "ti_token": "<afrobeat>",
        "paraphrased_prompt": "An energetic afrobeat song with vibrant drums",
        "ti_prompt": "An energetic <afrobeat> song with vibrant drums",
    },
    "ambient": {
        "ti_token": "<ambient>",
        "paraphrased_prompt": "A calming ambient music with soft textures",
        "ti_prompt": "A calming <ambient> music with soft textures",
    },
}

# -------------------------
# LOAD MODELS
# -------------------------
print(f"\U0001F3B5 Loading MusicGen: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=DURATION)
model.lm = model.lm.to(DEVICE)
model.lm.eval()

print("\U0001F3BC Loading CLAP model...")
clap_model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(DEVICE)
clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)

# -------------------------
# MAKE OUTPUT DIRS
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/wavs", exist_ok=True)

# -------------------------
# INJECT TOKENS + GENERATE + EVALUATE
# -------------------------
results = []

for genre_name, info in GENRES.items():
    ti_token = info["ti_token"]
    paraphrased_prompt = info["paraphrased_prompt"]
    ti_prompt = info["ti_prompt"]

    print(f"\n\U0001F680 Processing {genre_name}...")

    # Inject tokens
    for i in range(4):
        trained_emb_path = os.path.join(TI_TOKENS_DIR, genre_name, f"{genre_name}_codebook{i}.pt")
        new_token = torch.load(trained_emb_path).to(DEVICE)

        emb_layer = model.lm.emb[i]
        old_emb = emb_layer.weight.data
        new_weight = torch.cat([old_emb, new_token], dim=0)
        emb_layer.weight = torch.nn.Parameter(new_weight)

    # Generate baseline and TI audio
    prompts = {"baseline": paraphrased_prompt, "ti": ti_prompt}

    genre_results = {"genre": genre_name}

    for mode, prompt in prompts.items():
        print(f"\U0001F3B6 Generating {mode} audio...")
        wav = model.generate([prompt], progress=False)[0].cpu()
        wav_path = f"{OUTPUT_DIR}/wavs/{genre_name}_{mode}.wav"
        torchaudio.save(wav_path, wav, SAMPLE_RATE)

        # CLAP similarity
        # Resample generated audio to 48kHz for CLAP
        if SAMPLE_RATE != 48000:
            wav_resampled = torchaudio.functional.resample(wav, orig_freq=SAMPLE_RATE, new_freq=48000)
        else:
            wav_resampled = wav

        # Now pass the resampled wav to CLAP
        audio_inputs = clap_processor(audios=wav_resampled.squeeze(), sampling_rate=48000, return_tensors="pt").to(DEVICE)
        text_inputs = clap_processor(text=[paraphrased_prompt], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio_emb = clap_model.get_audio_features(**audio_inputs)
            text_emb = clap_model.get_text_features(**text_inputs)

        similarity = 1 - cosine(audio_emb.cpu().numpy()[0], text_emb.cpu().numpy()[0])
        print(f"\U0001F9EA {mode} similarity: {similarity:.4f}")

        genre_results[f"{mode}_similarity"] = similarity

    results.append(genre_results)

# -------------------------
# SAVE RESULTS
# -------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/similarity_results.csv", index=False)
print(f"✅ Results saved to {OUTPUT_DIR}/similarity_results.csv")

# -------------------------
# PLOT
# -------------------------
fig, ax = plt.subplots(figsize=(12,6))

x = range(len(results))
ax.bar([i-0.2 for i in x], results_df['baseline_similarity'], width=0.4, label="Baseline", align='center')
ax.bar([i+0.2 for i in x], results_df['ti_similarity'], width=0.4, label="TI", align='center')

ax.set_xticks(x)
ax.set_xticklabels(results_df['genre'], rotation=45)
ax.set_ylabel("CLAP Similarity")
ax.set_title("Baseline vs TI CLAP Similarity")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/clap_similarity_barchart.png")
print(f"✅ Barchart saved to {OUTPUT_DIR}/clap_similarity_barchart.png")
plt.close()
