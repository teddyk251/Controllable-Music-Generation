import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from audiocraft.models import MusicGen
import wandb

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "preprocessed_data_2"
MODEL_SIZE = "facebook/musicgen-large"
NUM_EPOCHS = 600
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 32000

PATIENCE = 20
FACTOR = 0.5
MIN_LR = 1e-6

WANDB_PROJECT = "musicgen-textual-inversion_150s"  # Your project name
USE_WANDB = True  # set False if you don't want to log

wandb.login(key="")  # Login to Weights & Biases

# -------------------------
# LOAD MODEL
# -------------------------
print(f"Loading MusicGen: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=10)
model.lm = model.lm.to(DEVICE)
model.lm.eval()

# -------------------------
# LOAD DATASET
# -------------------------
print("Loading training dataset...")
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

print(f"Found {sum(len(v) for v in genre_to_paths.values())} training samples across {len(genre_to_paths)} genres.")

# -------------------------
# TRAIN TOKENS PER GENRE
# -------------------------
save_dir = "trained_tokens"
os.makedirs(save_dir, exist_ok=True)

for genre_name, audio_paths in genre_to_paths.items():
    print(f"\nTraining token for genre: {genre_name}")
    print(f"Found {len(audio_paths)} clips for {genre_name}")

    # W&B initialization for each genre
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"textual_inversion_{genre_name}",
            config={
                "epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "patience": PATIENCE,
                "factor": FACTOR,
                "min_lr": MIN_LR,
                "genre": genre_name,
                "model_size": MODEL_SIZE
            },
            reinit=True
        )

    # -------------------------
    # PREPARE TRAINABLE TOKENS
    # -------------------------
    new_token_embeddings = []

    for i, emb_layer in enumerate(model.lm.emb):
        vocab_size, emb_dim = emb_layer.weight.shape
        print(f"Codebook {i}: vocab_size={vocab_size}, emb_dim={emb_dim}")

        new_token = torch.nn.Parameter(torch.randn(1, emb_dim, device=DEVICE) * 0.02)
        new_token_embeddings.append(new_token)

    optimizer = torch.optim.Adam(new_token_embeddings, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=FACTOR, patience=PATIENCE, min_lr=MIN_LR, verbose=True
    )

    # -------------------------
    # TRAINING LOOP
    # -------------------------
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
                target_tokens = audio_codes[code_idx]  # (T,)

                pred_embeds = new_token_embeddings[code_idx].expand(target_tokens.shape[0], -1)
                true_embeds = model.lm.emb[code_idx](target_tokens)

                assert pred_embeds.shape == true_embeds.shape, f"Mismatch: pred {pred_embeds.shape}, true {true_embeds.shape}"

                loss += F.mse_loss(pred_embeds, true_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(audio_paths)
        scheduler.step(avg_loss)

        print(f"{genre_name} Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")

        if USE_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"]
            })

    # -------------------------
    # SAVE TOKEN
    # -------------------------
    genre_token_dir = os.path.join(save_dir, genre_name)
    os.makedirs(genre_token_dir, exist_ok=True)

    for i, token in enumerate(new_token_embeddings):
        token_path = os.path.join(genre_token_dir, f"{genre_name}_codebook{i}.pt")
        torch.save(token.detach().cpu(), token_path)

    print(f"Saved trained embeddings for {genre_name} to {genre_token_dir}")

    if USE_WANDB:
        wandb.finish()

print("\nAll genres training complete!")