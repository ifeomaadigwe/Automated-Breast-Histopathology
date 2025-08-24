from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "BreakHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)  

# Image settings
IMAGE_SIZE = (224, 224)
CHANNELS = 3

# Training settings
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
RANDOM_STATE = 42

# Test settings
TEST_SIZE = 0.2
