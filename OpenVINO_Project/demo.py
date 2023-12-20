import numpy as np
import gradio as gr
from bark import SAMPLE_RATE
from bark.generation import SUPPORTED_LANGS

from main_tts import *

AVAILABLE_PROMPTS = ["Unconditional", "Announcer"]
PROMPT_LOOKUP = {}
for _, lang in SUPPORTED_LANGS:
    for n in range(10):
        label = f"Speaker {n} ({lang})"
        AVAILABLE_PROMPTS.append(label)
        PROMPT_LOOKUP[label] = f"{lang}_speaker_{n}"
PROMPT_LOOKUP["Unconditional"] = None
PROMPT_LOOKUP["Announcer"] = "announcer"

default_text = "Today's diary..."

title = "# 🐶 DYNAMIC DIARY using OpenVINO</div>"

def gen_tts(text, history_prompt):
    history_prompt = PROMPT_LOOKUP[history_prompt]
    audio_arr = generate_audio(text, history_prompt=history_prompt)
    audio_arr = (audio_arr * 32767).astype(np.int16)
    return (SAMPLE_RATE, audio_arr)


with gr.Blocks() as block:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Today's Diary", lines=2, value=default_text)
            run_button = gr.Button()
        with gr.Column():
            dynamic_out = gr.Audio(label="Dynamic Reading", type="numpy")
    inputs = [input_text]
    outputs = [dynamic_out]
    run_button.click(fn=gen_tts, inputs=inputs, outputs=outputs, queue=True)
try:
    block.queue().launch(debug=True)
except Exception:
    block.queue().launch(share=True, debug=True)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/