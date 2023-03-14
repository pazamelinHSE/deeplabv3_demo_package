import demo
from demo import *

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide", page_title="Image Background Remover")
st.write("## Semantic image segmentation with mobile-deeplabv3-plus")
st.write(
    ":dog: Try uploading an image to watch the segmentation. ",
    "Full quality images can be downloaded from the sidebar. ",
    "This code is open source and available ",
    "[here](https://github.com/tyler-simons/BackgroundRemoval).",
    "Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:",
)
st.sidebar.write("## Upload and download :gear:")


def process_image(pil_image):
    col1.write("Original Image :camera:")
    col1.image(pil_image)

    # convert image, drop alpha channel
    input_image = np.array(pil_image)
    input_image = input_image[:, :, :3]
    # run inference
    output_image, _ = st.session_state.runner.process_image(input_image, save_result=False)

    col2.write("Segmented Image :wrench:")
    col2.image(output_image)
    st.sidebar.markdown("\n")


if __name__ == "__main__":
    # cache imported model
    if "runner" not in st.session_state:
        model = demo.DEMO_DEFAULT_MODEL
        st.session_state.runner = demo.DemoRunner(model, num_threads=1, silent=True)

    col1, col2 = st.columns(2)
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])
    if uploaded_file is None:
        process_image(Image.open(demo.DEMO_DEFAULT_IMG))
    else:
        process_image(Image.open(uploaded_file))
