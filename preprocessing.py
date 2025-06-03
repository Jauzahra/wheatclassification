from PIL import Image
from model_utils.base import transform

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0)  # Add batch dimension
