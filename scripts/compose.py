import torch
from audiocraft.models import MusicGen
import torchaudio
import os

# -------------------------
# CONFIG
# -------------------------
MODEL_SIZE = "facebook/musicgen-large"
SAMPLE_RATE = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENRE_1 = "reggae"     # First genre
GENRE_2 = "pop"      # Second genre
TOKEN_1 = "<reggae>"
TOKEN_2 = "<pop>"
DURATION = 30

trained_token_dir1 = f"trained_tokens/{GENRE_1}"
trained_token_dir2 = f"trained_tokens/{GENRE_2}"

# -------------------------
# LOAD MODEL
# -------------------------
print(f"🎵 Loading MusicGen: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=DURATION)

model.lm = model.lm.to(DEVICE)
model.lm.eval()

# -------------------------
# INJECT BOTH TOKENS
# -------------------------
print(f"🛠️ Injecting tokens {TOKEN_1} and {TOKEN_2} into the model...")

for i in range(4):  # 4 codebooks
    # Load embeddings separately
    token1 = torch.load(os.path.join(trained_token_dir1, f"{GENRE_1}_codebook{i}.pt")).to(DEVICE)
    token2 = torch.load(os.path.join(trained_token_dir2, f"{GENRE_2}_codebook{i}.pt")).to(DEVICE)

    # Expand vocabulary twice: add token1, then token2
    emb_layer = model.lm.emb[i]
    old_emb = emb_layer.weight.data

    new_weight = torch.cat([old_emb, token1, token2], dim=0)  # Add two tokens
    emb_layer.weight = torch.nn.Parameter(new_weight)

print(f"✅ Both tokens injected!")

# -------------------------
# GENERATE MUSIC
# -------------------------
PROMPT = f"A vibrant fusion of {TOKEN_1} and {TOKEN_2} styles with rhythmic guitars"

print(f"📝 Prompt: {PROMPT}")

output_wav = model.generate([PROMPT], progress=True)[0].cpu()

save_path = f"generated_{GENRE_1}_{GENRE_2}_fusion.wav"
torchaudio.save(save_path, output_wav, SAMPLE_RATE)

print(f"✅ Generated composed audio saved at: {save_path}")