import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import gc

@st.cache_resource
def load_pipeline():
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    return pipeline.to("cpu")

# Load the pipeline
pipeline = load_pipeline()

# Streamlit app
st.title("Image Generation with Stable Diffusion")
st.write("Enter a prompt to generate an image:")

# Input from the user
prompt = st.text_input("Prompt", "A fantasy landscape", max_chars=100)

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        with torch.no_grad():
            image = pipeline(prompt).images[0]

        # Resize if necessary
        image = image.resize((512, 512))  # Resize to reduce memory

        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Free memory
        del image
        gc.collect()

st.write("This application uses the Stable Diffusion model for image generation.")
