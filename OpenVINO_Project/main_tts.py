from initialize_tts import *
from pipeline_tts import *
import time
from bark import SAMPLE_RATE
import scipy.io.wavfile

# << Prerequisites installation >>
# if sys.platform == "linux":
#     %pip install -q "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1" --index-url https://download.pytorch.org/whl/cpu
# else:
#     %pip install -q "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"
# %pip install -q "openvino>=2023.1.0" gradio
# %pip install -q "git+https://github.com/suno-ai/bark.git"

tokenizer, text_encoder_path0, text_encoder_path1, coarse_encoder_path, fine_model_dir = initialize()

core = ov.Core()

ov_text_model = OVBarkTextEncoder(
    core, "AUTO", text_encoder_path0, text_encoder_path1
)
ov_coarse_model = OVBarkEncoder(core, "AUTO", coarse_encoder_path)
ov_fine_model = OVBarkFineEncoder(core, "AUTO", fine_model_dir)

torch.manual_seed(42)
t0 = time.time()
text = "WOMAN: Title, Snowy Night. ...... \
        Today it snowed for the first time this winter. \
        It was a light snowfall and the air too warm for the snow to pile up on the ground. \
        Still, the snow continued throughout the night. \
        This snowy night gives me hope for a white Christmas."
audio_array = generate_audio(tokenizer, ov_text_model, ov_coarse_model, ov_fine_model, text)
generation_duration_s = time.time() - t0
audio_duration_s = audio_array.shape[0] / SAMPLE_RATE

print(f"took {generation_duration_s:.0f}s to generate {audio_duration_s:.0f}s of audio")

scipy.io.wavfile.write("snowy_night.wav", rate=SAMPLE_RATE, data=audio_array)

# from IPython.display import Audio
# Audio(audio_array, rate=SAMPLE_RATE)
