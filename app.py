import streamlit as st
from predict_image import predict_image
from predict_webcam import run_webcam
from utils import get_dataloaders

_, _, class_names = get_dataloaders("D:/Coding/Facial_Expression_Recognition")


st.title("Facial Expression Recognition")

mode = st.radio("Choose input mode:", ["Image Upload", "Webcam Stream"])

if mode == "Image Upload":
    uploaded_file=st.file_uploader("Upload an image",type=["jpg","png"])
    if uploaded_file:
        st.image(uploaded_file)
        predict_image(uploaded_file,class_names)
elif mode == "Webcam Stream":
    st.write("Press 'Start' to begin webcam emotion detection.")
    if st.button("Start"):
        run_webcam(class_names)
