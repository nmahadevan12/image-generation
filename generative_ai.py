import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import transformers
except ImportError:
    install("transformers")

try:
    import accelerate
except ImportError:
    install("accelerate")

import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the model with float16 precision for reduced memory usage
@st.cache_resource
def load_pipeline():
    st.write("Loading model...")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline = pipeline.to("mps")  # Set to MPS for Apple Silicon
    st.write("Model loaded successfully!")
    return pipeline

# Load the pipeline
pipeline = load_pipeline()

# Reduce the number of inference steps for faster image generation
pipeline.scheduler.set_timesteps(25)  # Adjusted for faster generation

# Define a function to generate images at specified resolution
def generate_image(prompt, height=512, width=512):
    # Generate an image with the specified resolution
    with torch.no_grad():
        image = pipeline(prompt, height=height, width=width).images[0]
    return image

# Streamlit app layout
st.title("AI Image Generator")
st.write("Generate images from descriptive text using AI.")

# Input prompt from the user
prompt = st.text_input("Enter a descriptive text prompt:", )

# Slider for selecting image resolution
st.write("Select the image resolution:")
height = st.slider("Height (in pixels)", 512, 1024, 768)  # Adjusted default to 768x768
width = st.slider("Width (in pixels)", 512, 1024, 768)

# Button to generate the image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt, height=height, width=width)
        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)
        # Save and provide download option
        filename = f"generated_image_{width}x{height}.png"
        image.save(filename)
        st.success(f"Image generated! You can download it below.")
        st.download_button(label="Download Image", data=open(filename, "rb").read(), file_name=filename)
