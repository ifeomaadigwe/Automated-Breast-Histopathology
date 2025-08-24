import os
from pathlib import Path
import pandas as pd

# Auto-detect project root
project_root = Path(__file__).resolve().parents[1]
base_dir = project_root / "data" / "BreakHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"

records = []
for label in ['benign', 'malignant']:
    sob_dir = base_dir / label / 'SOB'
    if not sob_dir.exists():
        print(f"❌ Path not found: {sob_dir}")
        continue
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

# Create DataFrame
df = pd.DataFrame(records)
print(f"✅ Found {len(df)} images")
print(df.head())
