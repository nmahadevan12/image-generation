import streamlit as st
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import os

# Set MPS high watermark to disable memory limit if using macOS
if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Load the model with the desired scheduler
@st.cache_resource
def load_pipeline():
    st.write("Loading model...")
    precision = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=precision
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Replace scheduler with DDIMScheduler for faster generation
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    st.write("Model loaded successfully!")
    return pipeline

# Load the pipeline
pipeline = load_pipeline()

# Reduce the number of inference steps for faster image generation
pipeline.scheduler.set_timesteps(10)  # Lowered for faster generation

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
prompt = st.text_input("Enter a descriptive text prompt:")

# Slider for selecting image resolution
st.write("Select the image resolution:")
height = st.slider("Height (in pixels)", 256, 512, 384)  # Reduced max resolution
width = st.slider("Width (in pixels)", 256, 512, 384)    # Reduced max resolution

# Button to generate the image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt, height=height, width=width)
        st.image(image, caption="Generated Image", use_column_width=True)
        
        # Save and provide download option
        filename = f"generated_image_{width}x{height}.png"
        image.save(filename)
        st.success("Image generated! You can download it below.")
        
        # Download button
        with open(filename, "rb") as file:
            st.download_button(
                label="Download Image",
                data=file.read(),
                file_name=filename,
                mime="image/png"
            )
