# app.py

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ✅ Streamlit UI configuration
st.set_page_config(page_title="Histopathology Slide Triage Dashboard", layout="wide")

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
st.sidebar.title("🧬 Breast Histopathology Classifier")
st.sidebar.write("Upload up to 10 histopathology images to predict benign or malignant tumors.")

st.sidebar.markdown("### 🔍 What Do These Terms Mean?")
st.sidebar.markdown("""
- **Benign**: Non-cancerous tissue. These cells do not invade nearby tissues or spread to other parts of the body.
- **Malignant**: Cancerous tissue. These cells can grow aggressively, invade surrounding areas, and may spread to other organs.
""")

# Main UI
st.title("🔬 Histopathology Slide Triage Dashboard")
uploaded_files = st.file_uploader("Upload up to 10 images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    benign_count = 0
    malignant_count = 0

    for uploaded_file in uploaded_files[:10]:  # Limit to 10
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "Malignant" if pred == 1 else "Benign"
        triage = (
            "✅ Likely benign — Non-cancerous tissue detected."
            if label == "Benign"
            else "⚠️ Requires pathologist review — Cancerous tissue detected."
        )

        if label == "Benign":
            benign_count += 1
            st.success(f"🩺 Prediction: **{label}** — Confidence: **{confidence:.2%}**")
            st.write(triage)
        else:
            malignant_count += 1
            st.warning(f"🩺 Prediction: **{label}** — Confidence: **{confidence:.2%}**")
            st.write(triage)

        st.markdown("---")

    # Summary
    st.subheader("📊 Batch Summary")
    st.write(f"Total slides reviewed: **{len(uploaded_files[:10])}**")
    st.write(f"✅ Benign cases: **{benign_count}**")
    st.write(f"⚠️ Malignant cases: **{malignant_count}**")
    st.info("ℹ️ This is a research prototype and not for clinical use.")

# Footer
st.markdown("---")
st.caption("Created by Ifeoma Adigwe • Powered by Streamlit and PyTorch")
