import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model/your_model.h5')

# Funzione per fare la previsione
def predict_image(img):
    img = img.resize((224, 224))  # Ridimensiona l'immagine alle dimensioni richieste dal modello
    img_array = np.array(img)  # Converte l'immagine in un array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Aggiungi la dimensione del batch
    img_array = img_array / 255.0  # Normalizza l'immagine se il modello lo richiede
    
    # Fai la previsione
    predictions = model.predict(img_array)
    return predictions

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
    predictions = predict_image(img)
    
    # Mostra la previsione
    st.write("Previsioni:", predictions)
