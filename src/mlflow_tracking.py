# mlflow_tracking.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Config imports
from config import ARTIFACTS_DIR, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_STATE

import mlflow
import mlflow.pytorch

# Metrics from model_training (you can import or pass them in)
metrics = {
    "accuracy": 0.9741,
    "precision": 0.9888,
    "recall": 0.9733,
    "f1_score": 0.9810,
    "roc_auc": 0.9953
}

# Start MLflow run
mlflow.set_tracking_uri("file:///" + (project_root / "mlruns").as_posix())
mlflow.set_experiment("Breast Histopathology Classification")

with mlflow.start_run(run_name="ResNet18"):
    # Log parameters
    mlflow.log_param("image_size", IMAGE_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("epochs", EPOCHS)

    # Log metrics
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Log model
    model_path = ARTIFACTS_DIR / "resnet18_breast_histology.pth"
    mlflow.log_artifact(model_path)

    # Log plots
    mlflow.log_artifact(ARTIFACTS_DIR / "confusion_matrix.png")
    mlflow.log_artifact(ARTIFACTS_DIR / "evaluation_metrics.png")

print("✅ MLflow tracking complete. Open the UI with:")
print("mlflow ui --backend-store-uri mlruns")
