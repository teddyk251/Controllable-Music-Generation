import os
import torch
import re
import torchaudio
import gradio as gr
from audiocraft.models import MusicGen

# -------------------------
# CONFIG
# -------------------------
MODEL_SIZE    = "facebook/musicgen-large"
SAMPLE_RATE   = 32000
DURATION      = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TI_GENRES     = ["lofi","hiphop","pop","reggae","country","jazz","afrobeat","ambient"]
TI_TOKEN_DIR  = "trained_tokens"

# -------------------------
# LOAD & INJECT TOKENS
# -------------------------
print(f"Loading MusicGen model: {MODEL_SIZE}")
model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(duration=DURATION)
model.lm = model.lm.to(DEVICE)
model.lm.eval()

def inject_tokens(genres):
    for g in genres:
        token_dir = os.path.join(TI_TOKEN_DIR, g)
        if not os.path.isdir(token_dir):
            print(f"No tokens for {g}")
            continue
        for i in range(4):
            path = os.path.join(token_dir, f"{g}_codebook{i}.pt")
            if not os.path.isfile(path):
                print(f"Missing {g}_codebook{i}")
                continue
            new_t = torch.load(path).to(DEVICE)
            emb  = model.lm.emb[i]
            emb.weight = torch.nn.Parameter(torch.cat([emb.weight.data, new_t], dim=0))
inject_tokens(TI_GENRES)
print("All tokens injected!")

# -------------------------
# PROMPT-UPDATE & GENERATION
# -------------------------

def update_prompt(raw_prompt, tokens):
    """
    Wrap each selected token in <> wherever it appears,
    but skip ones that are already wrapped.
    """
    out = raw_prompt
    for t in tokens:
        esc = re.escape(t)
        # (?<!<) ensures the match isn’t already preceded by '<'
        # (?!>)  ensures it isn’t already followed by '>'
        pattern = re.compile(rf'(?<!<)\b{esc}\b(?!>)', flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<{m.group(0)}>", out)
    return out

def generate_music(prompt, use_ti, tokens):
    # apply TI to prompt
    final_prompt = prompt
    if use_ti and tokens:
        final_prompt = update_prompt(prompt, tokens)
    print("▶Using prompt:", final_prompt)

    wav = model.generate([final_prompt], progress=True)[0].cpu()
    os.makedirs("gradio_generated", exist_ok=True)
    path = os.path.join("gradio_generated","output.wav")
    torchaudio.save(path, wav, SAMPLE_RATE)
    return path

# -------------------------
# BUILD THE UI
# -------------------------
with gr.Blocks() as app:
    gr.Markdown("## MusicGen + Textual Inversion Demo")
    prompt_input = gr.Textbox(label="Prompt", placeholder="e.g. a danceable afrobeat track")

    use_ti = gr.Checkbox(label="Use Textual Inversion?", value=False)

    # multi-select
    ti_selector = gr.CheckboxGroup(
        choices=TI_GENRES,
        label="Select TI token(s)",
        visible=False
    )

    # whenever TI flips, show/hide the selector and clear previous picks
    use_ti.change(
        fn=lambda flag: gr.update(visible=flag, value=[]),
        inputs=use_ti,
        outputs=ti_selector
    )

    # live-update the prompt box as soon as tokens change
    ti_selector.change(
        fn=update_prompt,
        inputs=[prompt_input, ti_selector],
        outputs=prompt_input
    )

    btn = gr.Button("Generate 🎵")
    audio = gr.Audio(type="filepath", autoplay=True)

    btn.click(
        fn=generate_music,
        inputs=[prompt_input, use_ti, ti_selector],
        outputs=audio
    )

app.launch(share=True)