import gdown
import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load your trained model
# https://drive.google.com/file/d/1mrlCErSUBZM66DItDVKtRDyuyX_o-J7e/view?usp=sharing
file_id = "1mrlCErSUBZM66DItDVKtRDyuyX_o-J7e"  # Replace with your file ID
url = f"https://drive.google.com/uc?id={file_id}"

# Output path to save the model
output = "Unite_ClassifierV2.h5"  # Replace with your model file name

# Download the file if it doesn't exist
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load your model
def load_model():
    # Replace with code to load your actual model
    loaded_model = tf.keras.models.load_model(output)
    return loaded_model

model = load_model()

# Print model summary to verify
model.summary()


# Define a function to preprocess the image and make predictions
def predict_image(image):
    # Define the target size (make sure it matches your model's input size)
    target_size = (224, 224)

    # Convert the PIL image to a NumPy array
    image = np.array(image)

    # Resize the image
    image_resized = cv2.resize(image, target_size)

    # Normalize the image
    image_normalized = image_resized / 255.0

    # Expand dimensions to match the model's expected input
    image_expanded = np.expand_dims(image_normalized, axis=0)

    # Make prediction
    predictions = model.predict(image_expanded)
    predicted_class_idx = np.argmax(predictions[0])

    # Define your class labels (replace this with your own labels)
    labels = [
        "Absol",
        "Aegislash",
        "Alolan Ninetales",
        "Azumarill",
        "Blastoise",
        "Blaziken",
        "Blissey",
        "Buzzwole",
        "Ceruledge",
        "Chandelure",
        "Charizard",
        "Cinderace",
        "Clefable",
        "Comfey",
        "Cramorant",
        "Crustle",
        "Decidueye",
        "Delphox",
        "Dodrio",
        "Dragapult",
        "Dragonite",
        "Duraludon",
        "Eldegoss",
        "Espeon",
        "Falinks",
        "Garchomp",
        "Gardevoir",
        "Gengar",
        "Glaceon",
        "Goodra",
        "Greedent",
        "Greninja",
        "Gyarados",
        "Ho-oh",
        "Hoopa",
        "Inteleon",
        "Lapras",
        "Leafeon",
        "Lucario",
        "Machamp",
        "Mamoswine",
        "Mega Mewtwo X",
        "Mega Mewtwo Y",
        "Meowscarada",
        "Metagross",
        "Mew",
        "Mimikyu",
        "Miraidon",
        "Mr. Mime",
        "Pikachu",
        "Sableye",
        "Scizor",
        "Slowbro",
        "Snorlax",
        "Sylveon",
        "Talonflame",
        "Trevenant",
        "Tsareena",
        "Tyranitar",
        "Umbreon",
        "Urshifu",
        "Venusaur",
        "Wigglytuff",
        "Zacian",
        "Zeraora",
        "Zoroark",
    ]

    # Get the predicted class label
    predicted_class = labels[predicted_class_idx]

    return predicted_class, image


# Streamlit app
st.title("Pokémon Identifier")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image of a single UNITE pokemon to identify...", type="jpg"
)

# URL input
url_input = st.text_input("Or enter the URL of an image")

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Process the image and get predictions
    with st.spinner("Processing..."):
        predicted_class, img_array = predict_image(image)

    # Plot the image with predicted class as the title
    st.image(
        img_array, caption=f"Predicted Class: {predicted_class}", use_column_width=True
    )
    st.write(f"Predicted Class: {predicted_class}")

elif url_input:
    try:
        # Download the image from the URL
        response = requests.get(url_input)
        image = Image.open(BytesIO(response.content))

        # Display the image from the URL
        st.image(image, caption="Image from URL.", use_column_width=True)

        # Process the image and get predictions
        with st.spinner("Processing..."):
            predicted_class, img_array = predict_image(image)

        # Plot the image with predicted class as the title
        st.image(
            img_array,
            caption=f"Predicted Class: {predicted_class}",
            use_column_width=True,
        )
        st.write(f"Predicted Class: {predicted_class}")

    except Exception as e:
        st.error(f"Error loading image: {e}")
