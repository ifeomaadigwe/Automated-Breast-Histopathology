# split_data.py

import sys
from pathlib import Path

# Dynamically add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import centralized config
from config import DATA_DIR, RANDOM_STATE, TEST_SIZE

import pandas as pd
from sklearn.model_selection import train_test_split

# Rebuild image metadata
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
print(f"✅ Total images found: {len(df)}")

# Split into train/test sets
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df['label'],
    random_state=RANDOM_STATE
)

print(f"✅ Train set size: {len(train_df)}")
print(f"✅ Test set size: {len(test_df)}")
