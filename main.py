# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Breast Histopathology Classifier",
    description="Upload up to 10 histopathology images to predict benign or malignant tumors.",
    version="1.0.0"
)

# Enable CORS (for frontend integration if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model_path = Path("artifacts/resnet18_breast_histology.pth")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Root endpoint
@app.get("/", tags=["Welcome"])
def read_root():
    return {"message": "Welcome to the Breast Histopathology Classifier API. Use /docs to test the prediction endpoint."}

# Batch prediction endpoint
@app.post("/predict", tags=["Classification"])
async def predict(images: List[UploadFile] = File(...)):
    """
    Predict whether each histopathology image is benign or malignant.
    - **images**: Upload up to 10 PNG or JPG images
    """
    results = []

    for image_file in images[:10]:  # Limit to 10
        contents = await image_file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "malignant" if pred == 1 else "benign"
        triage = (
            "Likely benign — Non-cancerous tissue detected."
            if label == "benign"
            else "Requires pathologist review — Cancerous tissue detected."
        )

        results.append({
            "filename": image_file.filename,
            "prediction": label,
            "confidence": round(confidence, 4),
            "triage_recommendation": triage
        })

    return JSONResponse(content={"results": results})
