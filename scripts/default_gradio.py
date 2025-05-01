import os
import torch
import torchaudio
import gradio as gr
from audiocraft.models import MusicGen

# -------------------------
# CONFIG
# -------------------------
MODEL_SIZE = "facebook/musicgen-large"
SAMPLE_RATE = 32000
DURATION = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TI_GENRES = ["lofi", "hiphop", "pop", "reggae", "country", "jazz", "afrobeat", "ambient"]
TI_TOKEN_DIR = "trained_tokens"  # where your learned tokens are stored

# -------------------------
# LOAD MODEL
# -------------------------
print(f"Loading MusicGen model: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=DURATION)
model.lm = model.lm.to(DEVICE)
model.lm.eval()

# -------------------------
# INJECT LEARNED TOKENS
# -------------------------
def inject_tokens(genres):
    for genre in genres:
        token_dir = os.path.join(TI_TOKEN_DIR, genre)
        if not os.path.exists(token_dir):
            print(f"Warning: No tokens found for {genre}")
            continue
        print(f"🛠️ Injecting {genre} tokens...")
        for i in range(4):  # 4 codebooks
            token_path = os.path.join(token_dir, f"{genre}_codebook{i}.pt")
            if not os.path.exists(token_path):
                print(f"Warning: Missing token {i} for {genre}")
                continue
            new_token = torch.load(token_path).to(DEVICE)
            emb_layer = model.lm.emb[i]
            old_emb = emb_layer.weight.data
            new_weight = torch.cat([old_emb, new_token], dim=0)
            emb_layer.weight = torch.nn.Parameter(new_weight)

inject_tokens(TI_GENRES)
print("All tokens injected!")

# -------------------------
# GENERATE MUSIC FUNCTION
# -------------------------
def generate_music(prompt, use_ti, selected_token):
    if use_ti:
        ti_prompt = prompt + f" in the style of <{selected_token}>"
    else:
        ti_prompt = prompt

    print(f"Final prompt: {ti_prompt}")

    # Generate music
    output_wav = model.generate([ti_prompt], progress=True)[0].cpu()

    # Save to file
    save_dir = "gradio_generated"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "output.wav")
    torchaudio.save(save_path, output_wav, SAMPLE_RATE)

    return save_path

# -------------------------
# BUILD GRADIO APP
# -------------------------
with gr.Blocks() as app:
    gr.Markdown("# MusicGen + Textual Inversion (TI) Demo")
    gr.Markdown("Generate music from text prompts. Optionally guide generation using learned TI tokens!")

    # Prompt
    prompt_input = gr.Textbox(
        label="Enter your music prompt",
        placeholder="Example: A relaxing piano melody with a soft beat"
    )

    # TI toggle
    use_ti = gr.Checkbox(label="Use Textual Inversion (TI)?", value=False)

    # Replace dropdown with Buttons
    ti_selector = gr.Radio(
    choices=TI_GENRES,
    label="Select TI Token",
    type="value",
    visible=False
    )

    # When use_ti flips, show/hide the buttons
    use_ti.change(
    fn=lambda flag: gr.update(visible=flag),
    inputs=use_ti,
    outputs=ti_selector
)



    # Generate button and output
    generate_btn = gr.Button("🎼 Generate Music")
    audio_output = gr.Audio(label="Generated Music", type="filepath", autoplay=True)
         
    generate_btn.click(
    fn=generate_music,
    inputs=[prompt_input, use_ti, ti_selector],
    outputs=[audio_output]
    )

app.launch(share=True)