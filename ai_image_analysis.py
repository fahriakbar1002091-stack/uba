# ai_image_analysis.py - REAL AI (BLIP + Keyword)
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from datetime import datetime
import os

caption_model = None
processor = None

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))

def load_models():
    global caption_model, processor
    if caption_model is None:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return caption_model, processor

def analyze_image_real(image_path: str) -> dict:
    try:
        model, proc = load_models()
        image = Image.open(image_path).convert("RGB")
        
        inputs = proc(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        ai_caption = proc.decode(out[0], skip_special_tokens=True)

        # Real keyword extraction from caption
        words = ai_caption.lower().split()
        objects = list(set([w for w in words if w in ["orang", "gunung", "langit", "tanaman", "hewan", "mobil", "bangunan"]]))
        topics = ["nature", "travel"] if "gunung" in ai_caption.lower() else ["lifestyle"]
        mood = ["calm", "fresh"] if "indah" in ai_caption.lower() else ["energetic"]

        return {
            "ai_caption": ai_caption,
            "ai_objects": objects or ["object"],
            "ai_topics": topics,
            "ai_mood": mood,
            "ai_summary": f"{ai_caption}. Gambar menunjukkan {', '.join(topics)}.",
            "analyzed_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "ai_caption": "Gambar tidak dapat dianalisis",
            "ai_objects": [],
            "ai_topics": [],
            "ai_mood": [],
            "ai_summary": str(e),
            "analyzed_at": datetime.now().isoformat()
        }