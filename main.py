# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Breast Histopathology Classifier",
    description="Upload histopathology images to predict benign or malignant tumors.",
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

# Prediction endpoint
@app.post("/predict", tags=["Classification"])
async def predict(image: UploadFile = File(...)):
    """
    Predict whether a histopathology image is benign or malignant.
    - **image**: Upload a PNG or JPG image
    """
    contents = await image.read()
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

    return JSONResponse(content={
        "prediction": label,
        "confidence": round(confidence, 4),
        "triage_recommendation": triage
    })
