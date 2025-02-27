import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)
# Load the labels
class_names = open("labels.txt", "r").readlines()
# Funzione per fare la previsione


# Imposta l'interfaccia utente
st.title('Riconoscimento Immagini con Keras e Streamlit')

# Carica un'immagine
uploaded_image = st.file_uploader("Carica un'immagine", type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    # Mostra l'immagine caricata
    st.image(uploaded_image, caption="Immagine caricata", use_column_width=True)
    
    # Apri l'immagine con Pillow
    img = Image.open(uploaded_image)
    
    # Fai la previsione
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
    image = Image.open(uploaded_image).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
    image_array = np.asarray(image)

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
    data[0] = normalized_image_array

# Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Mostra la previsione
    st.write("Previsioni:", class_name)
