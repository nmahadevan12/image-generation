import streamlit as st
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline
from accelerate import infer_auto_device_map

@st.cache_resource
def load_pipeline():
    model_name = "your-model-name"  # Replace with your model name
    device_map = infer_auto_device_map(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, device_map=device_map)
    return pipeline

def main():
    st.title("Image Generation App")
    pipeline = load_pipeline()

    prompt = st.text_input("Enter your prompt here", "A beautiful landscape")

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                try:
                    image = pipeline(prompt).images[0]  # Adjust this line if necessary
                    st.image(image, caption="Generated Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error generating image: {e}")
        else:
            st.warning("Please enter a valid prompt.")

if __name__ == "__main__":
    main()
