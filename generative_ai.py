import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion pipeline
def load_pipeline():
    # Load the pre-trained model
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Move the pipeline to CPU
    pipeline = pipeline.to("cpu")
    
    return pipeline

# Initialize the pipeline
pipeline = load_pipeline()

# Streamlit app
st.title("Image Generation with Stable Diffusion")
st.write("Enter a prompt to generate an image:")

# Input from the user
prompt = st.text_input("Prompt", "A fantasy landscape")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        # Generate image
        with torch.no_grad():
            image = pipeline(prompt).images[0]
        
        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)

# Optional: Add a footer or additional information
st.write("This application uses the Stable Diffusion model for image generation.")
