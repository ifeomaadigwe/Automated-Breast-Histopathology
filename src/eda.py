# eda.py

import sys
from pathlib import Path

# Dynamically add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import centralized config
from config import DATA_DIR, ARTIFACTS_DIR, RANDOM_STATE

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Load and index image metadata
records = []
for label in ['benign', 'malignant']:
    sob_dir = DATA_DIR / label / 'SOB'
    for subtype_dir in sob_dir.iterdir():
        if not subtype_dir.is_dir():
            continue
        for slide_dir in subtype_dir.iterdir():
            if not slide_dir.is_dir():
                continue
            for mag_dir in slide_dir.iterdir():
                if not mag_dir.is_dir():
                    continue
                for img_path in mag_dir.glob('*.[pj][pn]g'):
                    records.append({
                        'path': str(img_path),
                        'label': label,
                        'subtype': subtype_dir.name,
                        'slide_id': slide_dir.name,
                        'magnification': mag_dir.name
                    })

df = pd.DataFrame(records)
print(f"✅ Found {len(df)} images")

# Visualize 5 random samples
sample_df = df.sample(5, random_state=RANDOM_STATE)

# Ensure artifacts folder exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

for i, row in sample_df.iterrows():
    img = mpimg.imread(row['path'])
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"{row['label']} - {row['subtype']} - {row['magnification']}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / f"sample_{i}.png")  # Save to artifacts folder
    plt.show()
print("✅ Sample images saved to artifacts folder")