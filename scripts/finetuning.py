# fine_tune_embeddings.py

import os
import torch
import torchaudio
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from audiocraft.models import MusicGen

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "preprocessed_data_2"  # your updated 150s clips folder
MODEL_SIZE = "facebook/musicgen-large"
NUM_EPOCHS = 500
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 32000
PATIENCE = 20
FACTOR = 0.5
MIN_LR = 1e-6

PROJECT_NAME = "musicgen_finetune_embeddings"

# -------------------------
# LOAD MODEL
# -------------------------
print(f"\U0001F3B5 Loading MusicGen: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=10)
model.lm = model.lm.to(DEVICE)
model.lm.eval()

# Freeze transformer layers (VERY important!)
for param in model.lm.transformer.parameters():
    param.requires_grad = False

# -------------------------
# LOAD DATASET
# -------------------------
print("\U0001F4DA Loading training dataset...")
genre_to_paths = {}

for genre_folder in os.listdir(DATA_DIR):
    genre_path = os.path.join(DATA_DIR, genre_folder)
    if not os.path.isdir(genre_path):
        continue
    paths = []
    for audio_file in os.listdir(genre_path):
        if audio_file.endswith(".wav"):
            paths.append(os.path.join(genre_path, audio_file))
    if paths:
        genre_to_paths[genre_folder] = paths

print(f"✅ Found {sum(len(v) for v in genre_to_paths.values())} samples across {len(genre_to_paths)} genres.")

# -------------------------
# TRAIN PER GENRE
# -------------------------
save_dir = "finetuned_models"
os.makedirs(save_dir, exist_ok=True)

for genre_name, audio_paths in genre_to_paths.items():
    print(f"\n\U0001F525 Finetuning embeddings for genre: {genre_name}")

    wandb.init(project=PROJECT_NAME, name=f"finetune_{genre_name}", reinit=True)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.lm.emb.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=FACTOR, patience=PATIENCE, min_lr=MIN_LR, verbose=True
    )

    # TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for audio_path in tqdm(audio_paths, desc=f"{genre_name} Epoch {epoch+1}/{NUM_EPOCHS}"):
            wav, sr = torchaudio.load(audio_path)

            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            wav = wav.to(DEVICE)

            with torch.no_grad():
                encode_result = model.compression_model.encode(wav.unsqueeze(0))
                audio_codes = encode_result[0].squeeze(0)  # (4, T)

            loss = 0.0
            for code_idx in range(audio_codes.shape[0]):
                target_tokens = audio_codes[code_idx]
                true_embeds = model.lm.emb[code_idx](target_tokens)

                pred_embeds = model.lm.emb[code_idx](target_tokens)

                assert pred_embeds.shape == true_embeds.shape, f"Mismatch: pred {pred_embeds.shape}, true {true_embeds.shape}"
                loss += F.mse_loss(pred_embeds, true_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(audio_paths)
        scheduler.step(avg_loss)

        print(f"🎯 {genre_name} Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss, "lr": optimizer.param_groups[0]['lr']})

    # SAVE MODEL STATE
    genre_model_path = os.path.join(save_dir, f"{genre_name}_emb_finetuned.pth")
    torch.save(model.lm.emb.state_dict(), genre_model_path)
    print(f"💾 Saved finetuned embeddings for {genre_name} to {genre_model_path}")

    wandb.finish()

print("\n✅✅✅ All genres finetuning complete!")
