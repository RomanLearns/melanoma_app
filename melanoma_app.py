import streamlit as st
import tensorflow as tf
from tensorflow import keras # Use keras directly
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input # Import Input explicitly
# Import the specific preprocess_input function and base model class
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
# from tensorflow.keras.applications import EfficientNetB0 # Not needed if loading full model
import numpy as np
from PIL import Image
import matplotlib as mpl # For colormap
import matplotlib.pyplot as plt # For colormap usage
import cv2 # For resizing heatmap
import os
import traceback

# --- Configuration --- (Keep as before)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
MODEL_PATH = 'melanoma_detection_finetuned_model.h5' # Or .keras
CLASS_NAMES = ['Benign', 'Malignant']

# --- Load Original Model (Cached) --- (Keep as before)
@st.cache_resource
def load_keras_model(model_path):
    st.write(f"Loading original model from: {model_path}")
    try:
        model = load_model(model_path, compile=False) # Load uncompiled is fine here
        st.success("Original model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading original model: {e}")
        print(f"Error loading original model: {e}")
        traceback.print_exc()
        return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# --- Display GradCAM Overlay --- (Keep as before)
def display_gradcam_overlay(img_pil, heatmap, alpha=0.5):
# ... (display logic) ...
    if heatmap is None: return img_pil
    # ... (rest of display logic) ...
    try:
        img = keras.utils.img_to_array(img_pil); heatmap_uint8 = np.uint8(255 * heatmap)
        jet = mpl.colormaps["jet"]; jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_uint8]; jet_heatmap_img = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap_resized = jet_heatmap_img.resize((img.shape[1], img.shape[0]))
        jet_heatmap_array = keras.utils.img_to_array(jet_heatmap_resized)
        superimposed_img_array = jet_heatmap_array * alpha + img * (1 - alpha)
        superimposed_img = keras.utils.array_to_img(superimposed_img_array)
        return superimposed_img
    except Exception as e:
        print(f"Error displaying heatmap overlay: {e}"); print(f"Error displaying heatmap overlay: {e}")
        

# --- Streamlit App ---
# ... (Setup, Load original_model, Get target_conv_layer_object) ...
st.set_page_config(layout="wide")
st.title("Melanoma Detection Assistant")
original_model = load_keras_model(MODEL_PATH)
photo_model = tf.keras.applications.EfficientNetB0(weights='imagenet', input_shape=IMG_SHAPE, include_top=True)
# ... (File uploader) ...
tab1, tab2 = st.tabs(["Upload Image", "Metrics"])
with tab1:
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        _, mid, _ = st.columns([2, 7, 2])
        try:
            with mid:
                image = Image.open(uploaded_file).convert('RGB')
                # st.image(image, caption='Uploaded Image', use_column_width=True)
                img_resized_display = image.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array_orig = tf.keras.utils.img_to_array(img_resized_display)
                img_array_expanded_orig = np.expand_dims(img_array_orig, axis=0)
                
            
                # --- Prediction (Original Model) ---
                st.header("Diagnosis Suggestion")
                # Use original [0, 255] image array
                prediction = original_model.predict(img_array_expanded_orig)
                score = prediction[0][0]; class_index = 1 if score > 0.5 else 0
                predicted_class = CLASS_NAMES[class_index]
                confidence = score if class_index == 1 else 1 - score
                text1 = f"Prediction: {predicted_class}"
                text2 = f"Confidence: {confidence:.2%}"
                new_title1 = f"<p style='font-family:sans-serif; color:Black; font-size: 18px;'><b>Prediction: </b>{predicted_class}</p>"
                new_title2 = f"<p style='font-family:sans-serif; color:Black; font-size: 18px;'><b>Confidence: </b>{confidence:.2%}</p>"
                st.write(new_title1, unsafe_allow_html=True)
                st.write(new_title2, unsafe_allow_html=True)
                
                heatmap = make_gradcam_heatmap(img_array_expanded_orig, photo_model, 'top_conv')
                image = display_gradcam_overlay(img_resized_display, heatmap, alpha=0.4)
                
                # image_output = image.resize((800, 800))

                st.image(image, caption='Uploaded Image With Heatmap', use_column_width=True)
            
        except Exception as e: st.error(f"Error processing image: {e}"); uploaded_file = None
    else:
        st.info("Please upload an image file.")

with tab2:
    # metrics tab
    # have 7 plots we need to show, want to do 3x3 perhaps? 4x2? 3x2 with a 1 below in the middle?
    rows = 