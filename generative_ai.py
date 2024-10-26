import streamlit as st
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline
from accelerate import infer_auto_device_map

@st.cache_resource
def load_pipeline():
    model_name = "CompVis/stable-diffusion-v-1-4"  # Replace with your valid model name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = StableDiffusionPipeline.from_pretrained(model_name)
        return pipeline
    except EnvironmentError as e:
        st.error(f"Failed to load model: {e}")
        return None
        
def main():
    st.title("Image Generation ")
    pipeline = load_pipeline()

    # Add a text input box for the prompt
    prompt = st.text_input("Enter your prompt here:", "A beautiful landscape")

    # Button to generate the image
    if st.button("Image Generation"):
        if prompt:
            with st.spinner("Generating image..."):
                try:
                    # Generate image based on the prompt
                    image = pipeline(prompt).images[0]  
                    st.image(image, caption="Generated Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error generating image: {e}")
        else:
            st.warning("Please enter a valid prompt.")

if __name__ == "__main__":
    main()

