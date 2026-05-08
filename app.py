# ============================================================
# KIDNEY CT CLASSIFICATION WEB APP
# STREAMLIT + RESNET50 + GRAD-CAM
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import os
import gdown

# ============================================================
# GOOGLE DRIVE MODEL DOWNLOAD
# ============================================================

MODEL_PATH = "improved_kidney_model.h5"

FILE_ID = "1TTdJIkJrekzjYh668H00JRS1Qv6RHIrO"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Kidney Disease Detection AI",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🩺 Kidney CT Disease Detection AI")

st.markdown("""
Upload a kidney CT scan image to detect:

- Kidney Stone
- Kidney Tumor
- Kidney Cyst
- Normal Kidney

The system uses:
- ResNet50
- Grad-CAM Explainable AI
- Focused ROI Localization
""")

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 384

class_names = [
    "Cyst",
    "Normal",
    "Stone",
    "Tumor"
]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_ai_model():

    model = load_model(
        "improved_kidney_model.h5",
        compile=False
    )

    return model

model = load_ai_model()

# ============================================================
# LOAD CLEAN RESNET BACKBONE
# ============================================================

@st.cache_resource
def load_backbone():

    backbone = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    return backbone

backbone = load_backbone()

# ============================================================
# FEATURE EXTRACTOR
# ============================================================

last_conv_layer = backbone.get_layer(
    "conv5_block3_out"
)

feature_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=last_conv_layer.output
)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    pil_image = Image.open(uploaded_file).convert("RGB")

    original = np.array(pil_image)

    st.subheader("Uploaded CT Image")

    st.image(
        original,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    img = pil_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img_array = np.array(img).astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_tensor = tf.convert_to_tensor(
        img_array
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        img_array
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        np.max(predictions[0])
    ) * 100

    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # ========================================================
    # ONLY SHOW ROI IF ABNORMAL
    # ========================================================

    if predicted_class != "Normal":

        st.subheader("Explainable AI Visualization")

        # ====================================================
        # GRAD-CAM
        # ====================================================

        with tf.GradientTape() as tape:

            feature_maps = feature_model(
                img_tensor,
                training=False
            )

            tape.watch(feature_maps)

            predictions_tensor = tf.convert_to_tensor(
                predictions,
                dtype=tf.float32
            )

            score = predictions_tensor[
                :,
                predicted_index
            ]

        grads = tape.gradient(
            score,
            feature_maps
        )

        # SAFETY
        if grads is None:

            grads = tf.ones_like(
                feature_maps
            )

        # ====================================================
        # GLOBAL AVG POOLING
        # ====================================================

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0,1,2)
        )

        feature_maps = feature_maps[0]

        # ====================================================
        # CREATE HEATMAP
        # ====================================================

        heatmap = tf.reduce_sum(
            pooled_grads * feature_maps,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        heatmap /= (
            tf.reduce_max(heatmap) + 1e-10
        )

        heatmap = heatmap.numpy()

        # ====================================================
        # RESIZE HEATMAP
        # ====================================================

        heatmap = cv2.resize(
            heatmap,
            (
                original.shape[1],
                original.shape[0]
            )
        )

        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        heatmap_uint8 = np.uint8(
            heatmap * 255
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ====================================================
        # ADVANCED ROI
        # ====================================================

        heatmap_blur = cv2.GaussianBlur(
            heatmap,
            (15,15),
            0
        )

        heatmap_focus = np.uint8(
            heatmap_blur * 255
        )

        _, thresh = cv2.threshold(
            heatmap_focus,
            230,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((5,5), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        roi_image = original.copy()

        if len(contours) > 0:

            largest = max(
                contours,
                key=cv2.contourArea
            )

            if cv2.contourArea(largest) > 80:

                if len(largest) >= 5:

                    ellipse = cv2.fitEllipse(
                        largest
                    )

                    cv2.ellipse(
                        roi_image,
                        ellipse,
                        (255,0,0),
                        3
                    )

                M = cv2.moments(largest)

                if M["m00"] != 0:

                    cx = int(
                        M["m10"] / M["m00"]
                    )

                    cy = int(
                        M["m01"] / M["m00"]
                    )

                    cv2.circle(
                        roi_image,
                        (cx, cy),
                        6,
                        (0,255,0),
                        -1
                    )

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Grad-CAM Heatmap")

            st.image(
                overlay,
                use_container_width=True
            )

        with col2:

            st.subheader("Focused ROI Detection")

            st.image(
                roi_image,
                use_container_width=True
            )

    else:

        st.success(
            "No abnormal kidney region detected."
        )

        st.info(
            "Focused ROI visualization hidden for normal cases."
        )