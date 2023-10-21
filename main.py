import requests
from PIL import Image
import scipy
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import streamlit as st

def image_to_music(raw_image):
    img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    img_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    inputs = img_processor(raw_image, return_tensors="pt")

    out = img_model.generate(**inputs)
    txt = img_processor.decode(out[0], skip_special_tokens=True)

    audio_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    audio_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = audio_processor(
        text=[txt],
        padding=True,
        return_tensors="pt",
    )

    audio_values = audio_model.generate(**inputs, max_new_tokens=1000)
    sampling_rate = audio_model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("music.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

st.header("VisTune: an AI Image-to-Music generator")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

if st.button("Generate Music") and uploaded_image:
    raw_image = Image.open(uploaded_image).convert('RGB')
    image_to_music(raw_image)
    st.audio("music.wav")
