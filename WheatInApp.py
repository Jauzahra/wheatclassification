import streamlit as st
import torch
from PIL import Image
from model_utils.loader import load_model
from utils.preprocessing import preprocess_image
from utils.treatment_info import get_treatment_info

# Model paths
MODEL_PATHS = {
    "EfficientNetV2": "models/efficientnet.pth",
    "ConvNeXt": "models/convnext.pth",
    "ResNet50V2": "models/resnet.pth"
}

CLASS_NAMES = ['Black Rust', 'Brown Rust', 'Yellow Rust', 'Blast', 'Common Root Rot',
               'Fusarium Head Blight', 'Leaf Blight', 'Mildew', 'Septoria', 'Smut',
               'Tan Spot', 'Healthy']

st.set_page_config(page_title="Wheat Disease Classifier", layout="centered")
st.title("\U0001F33E Wheat Disease Classification and Advice")

# Sidebar options
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model:", ["EfficientNetV2", "ConvNeXt", "ResNet50V2"])

# Load selected model
@st.cache_resource
def load_selected_model():
    path = MODEL_PATHS[model_choice]
    return load_model(model_choice, path)

model = load_selected_model()

# Image upload
uploaded_file = st.file_uploader("Upload a wheat leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Wheat Leaf", use_column_width=True)

    if st.button("Classify Disease"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = CLASS_NAMES[pred.item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item() * 100

        st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")

        st.markdown(f"### \U0001F9EA Recommended Action for {predicted_class}")
        st.info(get_treatment_info(predicted_class))
