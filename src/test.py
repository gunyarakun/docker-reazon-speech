import soundfile
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

device = "cuda" if torch.cuda.is_available() else "cpu"

speech2text = Speech2Text.from_pretrained(
    "reazon-research/reazonspeech-espnet-v1",
    beam_size=5,
    batch_size=0,
    device=device
)

speech, _ = soundfile.read("samplesound.wav")
nbests = speech2text(speech)
text, *_ = nbests[0]
print(text)

