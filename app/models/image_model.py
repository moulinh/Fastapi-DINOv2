import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import io

MODEL_NAME = "best_dino_finetuned.pth"
class_names = ["00-normal", "01-minor", "02-moderate", "03-severe"]
device = "cpu"

def preprocess_image(image_bytes: bytes):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    return img_tensor

# 1. L'architecture (doit être identique à l'entraînement)
class DinoClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.classifier = nn.Linear(384, num_classes)
    def forward(self, x):
        return self.classifier(self.backbone(x))
    
def load_image_model():
    model = DinoClassifier(num_classes=4).to(device)
    project_root = Path.cwd()
    print("project_root : " + str(project_root))
    MODEL_PATH = "." + str(project_root) + "/" + MODEL_NAME
    print("MODEL_PATH : ", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print("*** Aucun modèle image trouvé !!!")
        model = None
    else: 
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Modèle image chargé.")
    return model

def predict_image(model, image_bytes: bytes) -> dict:
    model.eval()

    with torch.no_grad():
        input_tensor = preprocess_image(image_bytes).to(device)
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
    
    probs = probs.cpu()
    digit = int(torch.argmax(probs))
    class_name = class_names[digit] 
    confidence = round(float(probs[digit]), 4)
    probabilities = {
        class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
    }
    
    return {
        "class_name": class_name, 
        "confidence": confidence, 
        "probabilities": probabilities
    }