import gdown
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load your trained model
file_id = "1mrlCErSUBZM66DItDVKtRDyuyX_o-J7e"  # Replace with your file ID
url = f"https://drive.google.com/uc?id={file_id}"
output = "Unite_ClassifierV2.h5"  # Replace with your model file name

# Download the file if it doesn't exist
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load your model
def load_model():
    return tf.keras.models.load_model(output)

model = load_model()

# Print model summary to verify
model.summary()

# Define a function to preprocess the image and make predictions
def predict_image(img):
    target_size = (224, 224)
    
    # Convert the PIL image to RGB (to handle PNG with transparency)
    img = img.convert('RGB')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert the PIL image to a NumPy array
    img_array = np.array(img, dtype=np.float32)  # Ensure correct data type
    
    # Normalize the image array
    img_array /= 255.0
    
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # Define your class labels
    labels = [
        "Absol", "Aegislash", "Alolan Ninetales", "Azumarill", "Blastoise",
        "Blaziken", "Blissey", "Buzzwole", "Ceruledge", "Chandelure",
        "Charizard", "Cinderace", "Clefable", "Comfey", "Cramorant",
        "Crustle", "Decidueye", "Delphox", "Dodrio", "Dragapult",
        "Dragonite", "Duraludon", "Eldegoss", "Espeon", "Falinks",
        "Garchomp", "Gardevoir", "Gengar", "Glaceon", "Goodra",
        "Greedent", "Greninja", "Gyarados", "Ho-oh", "Hoopa",
        "Inteleon", "Lapras", "Leafeon", "Lucario", "Machamp",
        "Mamoswine", "Mega Mewtwo X", "Mega Mewtwo Y", "Meowscarada", "Metagross",
        "Mew", "Mimikyu", "Miraidon", "Mr. Mime", "Pikachu",
        "Sableye", "Scizor", "Slowbro", "Snorlax", "Sylveon",
        "Talonflame", "Trevenant", "Tsareena", "Tyranitar", "Umbreon",
        "Urshifu", "Venusaur", "Wigglytuff", "Zacian", "Zeraora",
        "Zoroark",
    ]
    
    # Get the predicted class label
    predicted_class = labels[predicted_class_index]

    return predicted_class, img

# Streamlit app
st.title("Pokémon Identifier")

# Option for the user to choose
option = st.radio(
    "What do you want to do?",
    ("Identify a single Pokémon", "Identify multiple Pokémon from a combined image")
)

if option == "Identify a single Pokémon":
    # Single Pokémon identification
    uploaded_file = st.file_uploader("Choose an image of a single UNITE Pokémon to identify...", type=["jpg", "png"])

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

        st.write(f"Predicted Pokemon: {predicted_class}")

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

            st.write(f"Predicted Pokemon: {predicted_class}")

        except Exception as e:
            st.error(f"Error loading image: {e}")

elif option == "Identify multiple Pokémon from a combined image":

# File uploader
    uploaded_file2 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file2 is not None:
    # Load the image
        image = Image.open(uploaded_file2)

    # Convert the image to RGB if it's not already in that format
        image_rgb = image.convert("RGB")

    # Display the original image
        st.image(image_rgb, caption="Original Image", use_column_width=True)

    # Dimensions of the original image
        width, height = image.size

    # Define the margins to be cut off from left and right
        margin_left = int(width * 0.17)
        margin_right = int(width * 0.17)

    # Crop the image to remove the empty parts on the left and right
        cropped_image = image_rgb.crop((margin_left, 0, width - margin_right, height))

    # Get the new dimensions after cropping
        new_width, new_height = cropped_image.size

    # Define the coordinates for each Pokémon's bounding box within the cropped image
        pokemon_coords = [
        # Team A (Top row)
            (0, 0, new_width // 5, new_height // 2),
            (new_width // 5, 0, 2 * new_width // 5, new_height // 2),
            (2 * new_width // 5, 0, 3 * new_width // 5, new_height // 2),
            (3 * new_width // 5, 0, 4 * new_width // 5, new_height // 2),
            (4 * new_width // 5, 0, new_width, new_height // 2),

        # Team B (Bottom row)
            (0, new_height // 2, new_width // 5, new_height),
            (new_width // 5, new_height // 2, 2 * new_width // 5, new_height),
            (2 * new_width // 5, new_height // 2, 3 * new_width // 5, new_height),
            (3 * new_width // 5, new_height // 2, 4 * new_width // 5, new_height),
            (4 * new_width // 5, new_height // 2, new_width, new_height)
        ]

    # Dictionary to store Pokémon images
        pokemon_images = {}
        predicted_class = {}

    # Crop and store each Pokémon image with additional cropping
        for i, (x1, y1, x2, y2) in enumerate(pokemon_coords):
        # Calculate the additional cropping margins
            add_margin_x = int((x2 - x1) * 0.07)  # 7% from left and right
            add_margin_y_top = int((y2 - y1) * 0.12)  # 12% from the top
            add_margin_y_bottom = int((y2 - y1) * 0.30)  # 30% from the bottom

        # Adjust the coordinates with the additional margins
            x1_new = max(x1 + add_margin_x, 0)
            y1_new = max(y1 + add_margin_y_top, 0)
            x2_new = min(x2 - add_margin_x, new_width)
            y2_new = min(y2 - add_margin_y_bottom, new_height)

        # Crop the image
            pokemon_image = cropped_image.crop((x1_new, y1_new, x2_new, y2_new))

        # Store the image in the dictionary
            pokemon_images[f'pokemon_{i + 1}'] = pokemon_image
            predicted_calss[f'pokemon_{i + 1}'] = predict_image(pokemon_image)

    # Plot the Pokémon images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.tight_layout(pad=3.0)

        for i in range(5):
            axes[0, i].imshow(pokemon_images[f'pokemon_{i + 1}'])
            axes[0, i].set_title(predicted_class[f'pokemon_{i + 1}'])
            axes[0, i].axis('off')

        for i in range(5):
            axes[1, i].imshow(pokemon_images[f'pokemon_{i + 6}'])
            axes[1, i].set_title(predicted_class[f'pokemon_{i + 6}'])
            axes[1, i].axis('off')

        st.pyplot(fig)

    else:
        st.write("Please upload an SS of UNITE Matchmaking to start the process.")
