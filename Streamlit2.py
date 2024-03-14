import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('models/LeNet5_vehicleclassifier.h5')

def predict_image_class(model, image):
    """Predict the class of the image using the provided model."""
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = 'Vehicle' if prediction[0][0] > 0.5 else 'Non-Vehicle'
    return predicted_class

# Set up the title of the app
st.title('Vehicle Classification App')

# Upload file widget
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Display the original uploaded image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the PIL image to the correct size for the model
    image = load_img(uploaded_file, target_size=(64, 64))
    
    # On a button click, predict the class
    if st.button('Predict'):
        predicted_class = predict_image_class(model, image)
        st.write(f"The image was predicted to be a {predicted_class}")

        
#run using command prompt python -m streamlit run Streamlit2.py
