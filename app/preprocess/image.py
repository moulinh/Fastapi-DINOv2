import numpy as np
from PIL import Image
import io
from PIL import Image
from torchvision import transforms
device = "cpu"

def preprocess_image(image_bytes: bytes):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    return img_tensor
