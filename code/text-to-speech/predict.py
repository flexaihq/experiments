# This is a basic inference script you can easily modify for your needs
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

checkpoint_path = "YOUR CHECKPOINT PATH"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint_path = (
    "/Users/gcheron/code/fcs-experiments/tts-output/checkpoint-10800-epoch-99"
)

model = ParlerTTSForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

prompt = "Salut, comment vas-tu aujourd'hui ?"
description = "A man speaking at a moderate speed with moderate pitch, very clear audio recording that has no background noise."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
attention_mask = input_ids.clone().fill_(1)

generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids,
    attention_mask=attention_mask,
)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("audio.wav", audio_arr, model.config.sampling_rate)
