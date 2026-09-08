import os
import re
import sys
import io
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
import emoji
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPVisionModelWithProjection

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
BALANCED_DIR = BASE_DIR / "dataset" / "balanced"
IMAGE_DIR = BALANCED_DIR / "images"
TEXT_DIR = BALANCED_DIR / "texts"
METADATA_FILE = BALANCED_DIR / "metadata.csv"
FEATURES_DIR = BASE_DIR / "dataset" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_hashtag(tag):
    # Split camelCase or concatenated hashtags
    tag = tag.lstrip("#")
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+', tag)
    if words:
        return " ".join(words)
    return tag

def preprocess_social_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Demojize emojis
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")
    
    # 2. Expand hashtags
    text = re.sub(r'#(\w+)', lambda m: clean_hashtag(m.group(1)), text)
    
    # 3. Clean URLs & Mentions
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
    text = re.sub(r'@\w+', '@user', text)
    
    # 4. Clean extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"Loading metadata from {METADATA_FILE}...", flush=True)
    df = pd.read_csv(METADATA_FILE)
    print(f"Total samples: {len(df)}", flush=True)
    
    sample_ids = []
    raw_texts = []
    cleaned_texts = []
    image_paths = []
    labels = []
    
    label_map = {"negative": 0, "positive": 1}
    
    for _, row in df.iterrows():
        sid = str(row["sample_id"]).strip()
        if sid.endswith(".0"):
            sid = sid[:-2]
        
        t_path = TEXT_DIR / f"{sid}.txt"
        i_path = IMAGE_DIR / f"{sid}.jpg"
        
        if not t_path.exists() or not i_path.exists():
            continue
            
        with open(t_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read().strip()
            
        cleaned_t = preprocess_social_text(text_content)
        
        sample_ids.append(sid)
        raw_texts.append(text_content)
        cleaned_texts.append(cleaned_t)
        image_paths.append(i_path)
        labels.append(label_map[row["label"].strip().lower()])

    print(f"Valid samples found: {len(sample_ids)}", flush=True)
    
    # -------------------------------------------------------------
    # Extract Twitter-RoBERTa Features
    # -------------------------------------------------------------
    roberta_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"\n[1/3] Extracting Twitter-RoBERTa features from {roberta_name}...", flush=True)
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_name)
    roberta_model = AutoModel.from_pretrained(roberta_name).to(device)
    roberta_model.eval()
    
    roberta_feats = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            encoded = roberta_tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = roberta_model(**encoded)
            # Use mean pooled or CLS token representation
            cls_repr = out.last_hidden_state[:, 0, :].cpu()
            roberta_feats.append(cls_repr)
            if (i // batch_size) % 10 == 0:
                print(f"Processed {min(i+batch_size, len(cleaned_texts))}/{len(cleaned_texts)} text samples...", flush=True)
                
    roberta_feats = torch.cat(roberta_feats, dim=0)
    print(f"Twitter-RoBERTa features shape: {roberta_feats.shape}", flush=True)
    
    # -------------------------------------------------------------
    # Extract DeBERTa-v3 Features
    # -------------------------------------------------------------
    deberta_name = "microsoft/deberta-v3-base"
    print(f"\n[2/3] Extracting DeBERTa-v3 features from {deberta_name}...", flush=True)
    deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_name)
    deberta_model = AutoModel.from_pretrained(deberta_name).to(device)
    deberta_model.eval()
    
    deberta_feats = []
    with torch.no_grad():
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            encoded = deberta_tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = deberta_model(**encoded)
            cls_repr = out.last_hidden_state[:, 0, :].cpu()
            deberta_feats.append(cls_repr)
            if (i // batch_size) % 10 == 0:
                print(f"Processed {min(i+batch_size, len(cleaned_texts))}/{len(cleaned_texts)} text samples...", flush=True)
                
    deberta_feats = torch.cat(deberta_feats, dim=0)
    print(f"DeBERTa-v3 features shape: {deberta_feats.shape}", flush=True)
    
    # -------------------------------------------------------------
    # Extract CLIP-ViT-Large Visual Features
    # -------------------------------------------------------------
    clip_name = "openai/clip-vit-large-patch14"
    print(f"\n[3/3] Extracting CLIP-ViT-Large visual features from {clip_name}...", flush=True)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_name).to(device)
    clip_model.eval()
    
    clip_feats = []
    img_batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(image_paths), img_batch_size):
            batch_paths = image_paths[i:i+img_batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(img)
                
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            out = clip_model(**inputs)
            img_embeds = out.image_embeds.cpu() # 768-dim projected vision vector
            clip_feats.append(img_embeds)
            if (i // img_batch_size) % 10 == 0:
                print(f"Processed {min(i+img_batch_size, len(image_paths))}/{len(image_paths)} image samples...", flush=True)
                
    clip_feats = torch.cat(clip_feats, dim=0)
    print(f"CLIP-ViT-Large features shape: {clip_feats.shape}", flush=True)
    
    # -------------------------------------------------------------
    # Save SOTA Features
    # -------------------------------------------------------------
    sota_text_path = FEATURES_DIR / "sota_text_features.pt"
    sota_image_path = FEATURES_DIR / "sota_image_features.pt"
    
    torch.save({
        "sample_ids": sample_ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "twitter_roberta": roberta_feats,
        "deberta": deberta_feats,
        "cleaned_texts": cleaned_texts
    }, sota_text_path)
    
    torch.save({
        "sample_ids": sample_ids,
        "clip_large": clip_feats
    }, sota_image_path)
    
    print(f"\nSuccessfully saved SOTA feature files to:")
    print(f"  - Text:  {sota_text_path}")
    print(f"  - Image: {sota_image_path}")

if __name__ == "__main__":
    main()
