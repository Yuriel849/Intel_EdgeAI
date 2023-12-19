from initialize_tts import *
from pipeline import *
import time
from bark import SAMPLE_RATE
import scipy.io.wavfile

# if sys.platform == "linux":
#     %pip install -q "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1" --index-url https://download.pytorch.org/whl/cpu
# else:
#     %pip install -q "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"
# %pip install -q "openvino>=2023.1.0" gradio
# %pip install -q "git+https://github.com/suno-ai/bark.git"

from pathlib import Path
from bark.generation import load_model, codec_decode, _flatten_codebooks
import torch
import openvino as ov

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

text_use_small = True

text_encoder = load_model(
    model_type="text", use_gpu=False, use_small=text_use_small, force_reload=False
)

text_encoder_model = text_encoder["model"]
tokenizer = text_encoder["tokenizer"]

text_model_suffix = "_small" if text_use_small else ""
text_model_dir = models_dir / f"text_encoder{text_model_suffix}"
text_model_dir.mkdir(exist_ok=True)
text_encoder_path1 = text_model_dir / "bark_text_encoder_1.xml"
text_encoder_path0 = text_model_dir / "bark_text_encoder_0.xml"

if not text_encoder_path0.exists() or not text_encoder_path1.exists():
    text_encoder_exportable = TextEncoderModel(text_encoder_model)
    ov_model = ov.convert_model(
        text_encoder_exportable, example_input=torch.ones((1, 513), dtype=torch.int64)
    )
    ov.save_model(ov_model, text_encoder_path0)
    logits, kv_cache = text_encoder_exportable(torch.ones((1, 513), dtype=torch.int64))
    ov_model = ov.convert_model(
        text_encoder_exportable,
        example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
    )
    ov.save_model(ov_model, text_encoder_path1)
    del ov_model
    del text_encoder_exportable
del text_encoder_model, text_encoder

coarse_use_small = True

coarse_model = load_model(
    model_type="coarse", use_gpu=False, use_small=coarse_use_small, force_reload=False, 
)

coarse_model_suffix = "_small" if coarse_use_small else ""
coarse_model_dir = models_dir / f"coarse{coarse_model_suffix}"
coarse_model_dir.mkdir(exist_ok=True)
coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder.xml"

if not coarse_encoder_path.exists():
    coarse_encoder_exportable = CoarseEncoderModel(coarse_model)
    logits, kv_cache = coarse_encoder_exportable(
        torch.ones((1, 886), dtype=torch.int64)
    )
    ov_model = ov.convert_model(
        coarse_encoder_exportable,
        example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
    )
    ov.save_model(ov_model, coarse_encoder_path)
    del ov_model
    del coarse_encoder_exportable
del coarse_model

fine_use_small = False

fine_model = load_model(model_type="fine", use_gpu=False, use_small=fine_use_small, force_reload=False)

fine_model_suffix = "_small" if fine_use_small else ""
fine_model_dir = models_dir / f"fine_model{fine_model_suffix}"
fine_model_dir.mkdir(exist_ok=True)

fine_feature_extractor_path = fine_model_dir / "bark_fine_feature_extractor.xml"

if not fine_feature_extractor_path.exists():
    lm_heads = fine_model.lm_heads
    fine_feature_extractor = FineModel(fine_model)
    feature_extractor_out = fine_feature_extractor(
        3, torch.zeros((1, 1024, 8), dtype=torch.int32)
    )
    ov_model = ov.convert_model(
        fine_feature_extractor,
        example_input=(
            torch.ones(1, dtype=torch.long),
            torch.zeros((1, 1024, 8), dtype=torch.long),
        ),
    )
    ov.save_model(ov_model, fine_feature_extractor_path)
    for i, lm_head in enumerate(lm_heads):
        ov.save_model(
            ov.convert_model(lm_head, example_input=feature_extractor_out),
            fine_model_dir / f"bark_fine_lm_{i}.xml",
        )

core = ov.Core()

ov_text_model = OVBarkTextEncoder(
    core, "AUTO", text_encoder_path0, text_encoder_path1
)
ov_coarse_model = OVBarkEncoder(core, "AUTO", coarse_encoder_path)
ov_fine_model = OVBarkFineEncoder(core, "AUTO", fine_model_dir)

torch.manual_seed(42)
t0 = time.time()
text = "눈 내리는 밤하늘 아래. 오늘은 이번 겨울에 처음으로 눈이 내린 날이다. 하지만 나는 집에 가지도 못하고 사무실에 잡혀있어, 창 밖으로 눈 내리는 도시 풍경을 볼 수 밖에 없었다."
audio_array = generate_audio(tokenizer, ov_text_model, ov_coarse_model, ov_fine_model, text)
generation_duration_s = time.time() - t0
audio_duration_s = audio_array.shape[0] / SAMPLE_RATE

print(f"took {generation_duration_s:.0f}s to generate {audio_duration_s:.0f}s of audio")

scipy.io.wavfile.write("snowy_night.wav", rate=SAMPLE_RATE, data=audio_array)

# from IPython.display import Audio
# Audio(audio_array, rate=SAMPLE_RATE)
