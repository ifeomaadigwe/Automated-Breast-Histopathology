# imbalance.py

import sys
from pathlib import Path

# Dynamically add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import centralized config
from config import DATA_DIR, RANDOM_STATE, TEST_SIZE

import pandas as pd
from torch.utils.data import WeightedRandomSampler

# Rebuild image metadata (same logic as in split_data.py)
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

# Split into train/test sets
from sklearn.model_selection import train_test_split
train_df, _ = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df['label'],
    random_state=RANDOM_STATE
)

# Count class frequencies
class_counts = train_df['label'].value_counts().to_dict()
class_weights = {label: 1.0 / count for label, count in class_counts.items()}
sample_weights = train_df['label'].map(class_weights).values

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print("✅ WeightedRandomSampler created to handle class imbalance")

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Visualize class distribution BEFORE balancing
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=train_df, palette='Set2')
plt.title("Class Distribution Before Balancing")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(project_root / "artifacts" / "class_distribution_before.png")
plt.show()

# Simulate sampling to visualize AFTER balancing
sample_indices = list(sampler)
balanced_labels = train_df.iloc[sample_indices]['label']

plt.figure(figsize=(6, 4))
sns.countplot(x=balanced_labels, palette='Set1')
plt.title("Class Distribution After Balancing (Simulated)")
plt.xlabel("Label")
plt.ylabel("Sampled Count")
plt.tight_layout()
plt.savefig(project_root / "artifacts" / "class_distribution_after.png")
plt.show()
print("✅ Class distribution plots saved to artifacts folder")