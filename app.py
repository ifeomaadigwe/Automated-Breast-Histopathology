# app.py

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load trained model
model_path = Path("artifacts/resnet18_breast_histology.pth")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Sidebar
st.sidebar.title(" Breast Histopathology Classifier")
st.sidebar.write("Upload a histopathology image to predict whether it's benign or malignant.")

st.sidebar.markdown("### 🧬 What Do These Terms Mean?")
st.sidebar.markdown("""
- **Benign**: Non-cancerous tissue. 
- **Malignant**: Cancerous tissue.
""")


# Main UI
st.title("🔬 Histopathology Image Classification Dashboard")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Malignant" if pred == 1 else "Benign"
    st.subheader(f"🩺 Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Triage Recommendation
    if label == "Benign":
        st.success("✅ Likely benign — Non-cancerous tissue detected.")
    else:
        st.warning("⚠️ Requires pathologist review — Cancerous tissue detected.")
    st.info("ℹ️ This is a research prototype and not for clinical use.")

# Collapsible Confusion Matrix
with st.expander("🔍 Show Confusion Matrix"):
    cm_path = Path("artifacts/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix", use_column_width=True)
    else:
        st.warning("Confusion matrix not found.")

# Footer
st.markdown("---")
st.caption("Created by Ifeoma Adigwe • Powered by Streamlit and PyTorch")
