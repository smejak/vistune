# VisTune - AI Image-to-Music Generator

Demo: https://huggingface.co/spaces/smejak/vistune

VisTune is an artificial intelligence (AI) application that leverages state-of-the-art machine learning models to convert images into music. The application utilizes the image captioning abilities of Salesforce's BLIP model to generate textual descriptions of images, which are then used as input for the Musicgen model from Facebook to generate music. This document provides a brief overview of how to use VisTune and the underlying technologies.

---

## Table of Contents
1. [Features](#features)
2. [Installation & Dependencies](#installation--dependencies)
3. [Usage](#usage)
4. [Notes](#notes)

---

## Features
- **Image Captioning**: Uses the BLIP model from Salesforce to generate textual captions of images.
- **Music Generation**: Uses the generated captions as input to the Musicgen model from Facebook to produce music.
- **Interactive GUI**: A streamlined interface built using Streamlit for real-time image-to-music generation.

---

## Installation & Dependencies

Before you can use VisTune, you'll need to ensure you have the necessary libraries and models installed. Here are the primary dependencies:

- `requests`
- `PIL`
- `transformers`
- `IPython`
- `streamlit`

You can install these using `pip`:

```bash
pip install requests Pillow transformers IPython streamlit
```

You'll also need to download the required models:

- Salesforce's BLIP model for image captioning: `"Salesforce/blip-image-captioning-large"`
- Facebook's Musicgen model for music generation: `"facebook/musicgen-small"`

---

## Usage

To run VisTune, execute the script. It will launch a Streamlit app in your browser. 

1. Once the app loads, you'll see an image (in this case, `forest.png`).
2. Click on the "Generate Music" button.
3. The application will generate a caption for the image and then use this caption to produce music.
4. The generated music will be played in the Streamlit app.

Note: The provided code demonstrates the conversion for a specific image (`forest.png`). To use other images, you'll need to modify the `st.image` and `image_to_music` function calls accordingly.

---

## Notes

- The music generated is conditional on the caption produced by the BLIP model. Different images or slight changes in captions might produce different musical outputs.
- The provided code demonstrates the process for a specific image, but you can easily adapt it to other images or even allow users to upload their own images for processing.
- The code includes both conditional and unconditional image captioning approaches. The unconditional approach, which doesn't use a seed text ("an image of"), is the one used for music generation in this example.

---

Enjoy the magical experience of converting visuals into auditory wonders with VisTune! 🎶🖼️