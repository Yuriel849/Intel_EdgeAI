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

default_text = "Hello, my name is Suno. And, uh — and I like pizza. [laughs]\nBut I also have other interests such as playing tic tac toe."

title = "# 🐶 Bark: Text-to-Speech using OpenVINO</div>"

description = """
Bark is a universal text-to-audio model created by [Suno](http://suno.ai). \
Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. \
The model output is not censored and the authors do not endorse the opinions in the generated content. \
Use at your own risk.
"""

article = """

## 🌎 Foreign Language

Bark supports various languages out-of-the-box and automatically determines language from input text. \
When prompted with code-switched text, Bark will even attempt to employ the native accent for the respective languages in the same voice.

Try the prompt:

```
Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. But I suppose your english isn't terrible.
```

## 🤭 Non-Speech Sounds

Below is a list of some known non-speech sounds, but we are finding more every day. \
Please let us know if you find patterns that work particularly well on Discord!

* [laughter]
* [laughs]
* [sighs]
* [music]
* [gasps]
* [clears throat]
* — or ... for hesitations
* ♪ for song lyrics
* capitalization for emphasis of a word
* MAN/WOMAN: for bias towards speaker

Try the prompt:

```
" [clears throat] Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as... ♪ singing ♪."
```

## 🎶 Music
Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. \
Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.

Try the prompt:

```
♪ In the jungle, the mighty jungle, the lion barks tonight ♪
```

## 🧬 Voice Cloning

Bark has the capability to fully clone voices - including tone, pitch, emotion and prosody. \
The model also attempts to preserve music, ambient noise, etc. from input audio. \
However, to mitigate misuse of this technology, we limit the audio history prompts to a limited set of Suno-provided, fully synthetic options to choose from.

## 👥 Speaker Prompts

You can provide certain speaker prompts such as NARRATOR, MAN, WOMAN, etc. \
Please note that these are not always respected, especially if a conflicting audio history prompt is given.

Try the prompt:

```
WOMAN: I would like an oatmilk latte please.
MAN: Wow, that's expensive!
```

"""

examples = [
    [
        "Please surprise me and speak in whatever voice you enjoy. Vielen Dank und Gesundheit!",
        "Unconditional",
    ],
    [
        "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.",
        "Speaker 1 (en)",
    ],
    [
        "Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. But I suppose your english isn't terrible.",
        "Speaker 0 (es)",
    ],
]


def gen_tts(text, history_prompt):
    history_prompt = PROMPT_LOOKUP[history_prompt]
    audio_arr = generate_audio(text, history_prompt=history_prompt)
    audio_arr = (audio_arr * 32767).astype(np.int16)
    return (SAMPLE_RATE, audio_arr)


with gr.Blocks() as block:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", lines=2, value=default_text)
            options = gr.Dropdown(
                AVAILABLE_PROMPTS, value="Speaker 1 (en)", label="Acoustic Prompt"
            )
            run_button = gr.Button()
        with gr.Column():
            audio_out = gr.Audio(label="Generated Audio", type="numpy")
    inputs = [input_text, options]
    outputs = [audio_out]
    gr.Examples(examples=examples, fn=gen_tts, inputs=inputs, outputs=outputs)
    gr.Markdown(article)
    run_button.click(fn=gen_tts, inputs=inputs, outputs=outputs, queue=True)
try:
    block.queue().launch(debug=True)
except Exception:
    block.queue().launch(share=True, debug=True)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/