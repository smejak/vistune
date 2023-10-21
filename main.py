import requests
from PIL import Image
import scipy
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio

import streamlit as st

def image_to_music(path_to_image):
    img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    img_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(path_to_image).convert('RGB')

    inputs = img_processor(raw_image, return_tensors="pt")

    out = img_model.generate(**inputs)
    print(img_processor.decode(out[0], skip_special_tokens=True))
    txt = img_processor.decode(out[0], skip_special_tokens=True)
    audio_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    audio_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = audio_processor(
        text=[txt],
        padding=True,
        return_tensors="pt",
    )

    audio_values = audio_model.generate(**inputs, max_new_tokens=256)
    sampling_rate = audio_model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("music.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

st.header("VisTune: an AI Image-to-Music generator")
st.image('./forest.png')

if st.button("Generate Music"):
    image_to_music('./forest.png')
    st.audio("music.wav")