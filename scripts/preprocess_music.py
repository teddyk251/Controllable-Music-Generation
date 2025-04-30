import os
import torchaudio
import torch
from tqdm import tqdm

# -------------------------
# CONFIGURATION
# -------------------------
RAW_DATA_DIR = "training_data"          # Your current folder with genre subfolders
OUTPUT_DIR = "preprocessed_data_2"         # Folder to save preprocessed 30s 32kHz clips
TARGET_SAMPLE_RATE = 32000
TARGET_DURATION_SEC = 150  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# FUNCTION DEFINITIONS
# -------------------------

def preprocess_audio(file_path, target_sr=32000, target_duration=150, start_sec=10):
    # Load audio
    waveform, sr = torchaudio.load(file_path)

    # Resample if necessary
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Crop to target duration starting from start_sec
    start_sample = int(start_sec * target_sr)
    end_sample = start_sample + int(target_duration * target_sr)

    num_samples = waveform.shape[-1]

    if end_sample <= num_samples:
        # Clip from 10s to 40s
        waveform = waveform[:, start_sample:end_sample]
    else:
        # If the clip is too short, pad at the end
        waveform = waveform[:, start_sample:]
        pad_size = (target_sr * target_duration) - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    return waveform

# -------------------------
# MAIN SCRIPT
# -------------------------

print(f"🚀 Starting preprocessing from {RAW_DATA_DIR}...")

for genre_folder in tqdm(os.listdir(RAW_DATA_DIR)):
    genre_path = os.path.join(RAW_DATA_DIR, genre_folder)
    if not os.path.isdir(genre_path):
        continue

    output_genre_dir = os.path.join(OUTPUT_DIR, genre_folder)
    os.makedirs(output_genre_dir, exist_ok=True)

    for filename in os.listdir(genre_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            input_path = os.path.join(genre_path, filename)
            output_path = os.path.join(output_genre_dir, os.path.splitext(filename)[0] + ".wav")

            try:
                processed_waveform = preprocess_audio(input_path)
                torchaudio.save(output_path, processed_waveform, TARGET_SAMPLE_RATE)
            except Exception as e:
                print(f"❌ Error processing {input_path}: {e}")

print("✅ Preprocessing complete! Files saved under:", OUTPUT_DIR)