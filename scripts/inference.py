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
GENRE_NAME = "afrobeat"     # The genre you trained
LEARNED_TOKEN = "<afrobeat>"  # The token you were optimizing
DURATION = 10           # seconds

trained_token_dir = f"trained_tokens/{GENRE_NAME}"

# -------------------------
# LOAD MODEL
# -------------------------
print(f"🎵 Loading MusicGen: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=DURATION)

# Move the internal transformer model to the correct device
model.lm = model.lm.to(DEVICE)
model.lm.eval()

# -------------------------
# INJECT TRAINED TOKENS
# -------------------------
print(f"🛠️ Injecting learned embeddings for {GENRE_NAME}...")

for i in range(4):  # 4 codebooks
    trained_emb_path = os.path.join(trained_token_dir, f"{GENRE_NAME}_codebook{i}.pt")
    new_token = torch.load(trained_emb_path).to(DEVICE)  # Shape (1, emb_dim)

    # Expand the vocabulary to add 1 more slot
    emb_layer = model.lm.emb[i]
    old_emb = emb_layer.weight.data

    # Concatenate the new token
    new_weight = torch.cat([old_emb, new_token], dim=0)  # (old_vocab + 1, emb_dim)

    # Replace the embedding layer's weight
    emb_layer.weight = torch.nn.Parameter(new_weight)

print(f"✅ Learned tokens injected into model!")

# -------------------------
# GENERATE MUSIC
# -------------------------
# PROMPT = f"an energetic yet calm <lofi> groove"
# PROMPT = f"an {LEARNED_TOKEN} track"
PROMPT = f"a happy {LEARNED_TOKEN} beat"
# PROMPT = f"a relaxing instrumental study track with {LEARNED_TOKEN}"
# PROMPT = f"Street {LEARNED_TOKEN} track with a catchy melody"
# PROMPT = f"A soulful {LEARNED_TOKEN} track with a catchy guitar riffs"
# PROMPT = f"An upbeat {LEARNED_TOKEN} track "

print(f"📝 Prompt: {PROMPT}")

output_wav = model.generate([PROMPT], progress=True)[0].cpu()  # (Channels, Samples)

# Save output
save_path = f"generated_{GENRE_NAME}.wav"
torchaudio.save(save_path, output_wav, SAMPLE_RATE)

print(f"✅ Generated audio saved at: {save_path}")