import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import load_model
from gradcam_utils import generate_gradcam
from pytorch_grad_cam.utils.image import show_cam_on_image
import requests
from io import BytesIO

# Page config
st.set_page_config(page_title="Coral Health Assessment", layout="wide", initial_sidebar_state="expanded")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model load with spinner
with st.spinner("Loading model..."):
    model = load_model("coral_health_resnet50.pth", device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

labels = ["Bleached", "Healthy"]

def load_image_from_upload(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        return img
    except Exception:
        return None

def load_image_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None

def preprocess(img):
    return transform(img).unsqueeze(0).to(device)

def predict(img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probs).item()
    return pred, probs


st.title("🪸 Coral Health Assessment")
st.markdown(
    "This app classifies coral images as **Bleached** or **Healthy (Unbleached)** and can show Grad-CAM visualizations to explain the model's focus."
)

# Sidebar controls
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
image_url = st.sidebar.text_input("Or paste an image URL")
show_cam = st.sidebar.checkbox("Show Grad-CAM explanation", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Tips: upload clear photos of coral, preferably centered and well-lit.")

# Main layout
col1, col2 = st.columns([2, 1])

# Load image (upload or URL)
image = None
if uploaded_file is not None:
    image = load_image_from_upload(uploaded_file)
elif image_url:
    image = load_image_from_url(image_url)

if image is None:
    col1.info("Upload an image from the sidebar or paste an image URL to get started.")
    st.markdown("#### Example workflow")
    st.markdown("1. Upload or paste an image URL in the sidebar.\n2. Wait for the prediction.\n3. Toggle Grad-CAM to see model attention.")
else:
    # show the image
    col1.image(image, caption="Input image", use_container_width=True)

    # prediction panel
    img_tensor = preprocess(image)

    with st.spinner("Running inference..."):
        try:
            pred, probs = predict(img_tensor)
        except Exception as e:
            col2.error(f"Inference failed: {e}")
            pred = None

    if pred is not None:
        confidence = float(probs[pred].cpu().numpy())
        label = labels[pred]

        # nice summary
        if label == "Bleached":
            col2.error(f"Prediction: {label}")
        else:
            col2.success(f"Prediction: {label}")

        col2.metric("Confidence", f"{confidence*100:.2f}%")
        # progress bar for confidence
        col2.progress(int(confidence * 100))

        # Grad-CAM
        if show_cam:
            try:
                cam = generate_gradcam(model, img_tensor)

                img_np = np.array(image.resize((224, 224))) / 255.0
                cam_image = show_cam_on_image(img_np, cam, use_rgb=True)

                col1.markdown("**Grad-CAM**")
                col1.image(cam_image, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                col1.warning(f"Could not generate Grad-CAM: {e}")

        # small probabilities table
        prob_text = "\n".join([f"{labels[i]}: {float(p.cpu().numpy())*100:.2f}%" for i, p in enumerate(probs)])
        col2.write("**Class probabilities**")
        col2.write(prob_text)

    # End if pred

