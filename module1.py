import streamlit as st
from PIL import Image
import io

# spinner function
def spinner_(message,task):
    with st.spinner(message):
        task

# To view uploaded image
def display_image(uploaded_img):
    img_binary = uploaded_img.read()
    image = Image.open(io.BytesIO(img_binary))
    return image