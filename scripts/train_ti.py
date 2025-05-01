import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from audiocraft.models import MusicGen

# -------------------------
# CONFIG
# -------------------------
GENRE = "jazz"
NEW_TOKEN = f"<{GENRE}>"
DATA_DIR = "preprocessed_data_2"
SAMPLE_RATE = 32000
NUM_EPOCHS = 300
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD MUSICGEN + T5 SETUP
# -------------------------
model = MusicGen.get_pretrained("facebook/musicgen-large")
model = model.lm.eval()  # optional: set to eval mode
# model.lm = model.lm.to(DEVICE)
model.lm.cfg_dropout.p = 0.0  # Disable classifier-free dropout

t5_cond = model.lm.condition_provider.conditioners["description"]
tokenizer = t5_cond.t5_tokenizer
encoder = t5_cond.t5

# Add token and resize
tokenizer.add_tokens([NEW_TOKEN])
encoder.resize_token_embeddings(len(tokenizer))
token_id = tokenizer.convert_tokens_to_ids(NEW_TOKEN)

# Make the new embedding learnable
embedding_matrix = encoder.get_input_embeddings().weight
embedding = torch.nn.Parameter(
    torch.randn(embedding_matrix.shape[1], device=DEVICE, dtype=embedding_matrix.dtype) * 0.02
)
optimizer = torch.optim.Adam([embedding], lr=LEARNING_RATE)

# -------------------------
# DATA
# -------------------------
genre_dir = os.path.join(DATA_DIR, GENRE)
audio_paths = [os.path.join(genre_dir, f) for f in os.listdir(genre_dir) if f.endswith(".wav")]
print(f"Found {len(audio_paths)} clips for genre '{GENRE}'.")

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for path in tqdm(audio_paths, desc=f"Epoch {epoch+1}"):
        # Load & preprocess audio
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(dim=0, keepdim=True).to(DEVICE)

        # Target audio embedding (proxy)
        with torch.no_grad():
            encoded = model.compression_model.encode(wav.unsqueeze(0))[0].squeeze(0)
            target_vec = encoded.mean(dim=1).mean(dim=0)  # shape: (D,)

        # Tokenize prompt
        prompt = f"A {NEW_TOKEN} track"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Replace token embedding manually during forward pass
        def patch_embedding(input_ids):
            original_embed = encoder.get_input_embeddings()
            embeddings = original_embed(input_ids)
            # Mask and replace
            mask = input_ids == token_id  # (B, T)
            embeddings[mask] = embedding  # Injected learnable token
            return embeddings

        # Forward pass manually with patched embedding
        embeddings = patch_embedding(input_ids)
        encoder_outputs = encoder(inputs_embeds=embeddings).last_hidden_state  # shape: (1, T, D)

        # Extract NEW_TOKEN representation
        token_pos = (input_ids == token_id).nonzero(as_tuple=True)
        if len(token_pos[0]) == 0:
            print("Token not found in input sequence.")
            continue
        t5_repr = encoder_outputs[token_pos]  # shape: (D,)

        
        loss = F.mse_loss(t5_repr, target_vec.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(audio_paths)
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")

# -------------------------
# SAVE EMBEDDING
# -------------------------
os.makedirs("learned_tokens_text_encoder", exist_ok=True)
torch.save(embedding.detach().cpu(), f"learned_tokens_text_encoder/{GENRE}.pt")
print(f"Saved embedding to learned_tokens_text_encoder/{GENRE}.pt")