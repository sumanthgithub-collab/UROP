import os
import ast
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
BALANCED_DIR = BASE_DIR / "dataset" / "balanced"
IMAGE_DIR = BALANCED_DIR / "images"
SPLITS_DIR = BASE_DIR / "dataset" / "splits"
TOKENIZED_DIR = BASE_DIR / "dataset" / "tokenized"
FEATURES_DIR = BASE_DIR / "dataset" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def verify_features():
    text_pt_path = FEATURES_DIR / "text_features.pt"
    image_pt_path = FEATURES_DIR / "image_features.pt"
    if text_pt_path.exists() and image_pt_path.exists():
        saved_text = torch.load(text_pt_path)
        saved_image = torch.load(image_pt_path)
        print(f"Pre-extracted text features shape: {saved_text['features'].shape}")
        print(f"Pre-extracted image features shape: {saved_image['features'].shape}")
        return True
    return False

if __name__ == "__main__":
    if not verify_features():
        print("Feature files missing, run extraction logic.")
