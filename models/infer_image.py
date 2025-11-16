import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# Path to your trained model
MODEL_PATH = r"models\image_model\best_model.pth"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define same preprocessing as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
def load_model(path):
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Predict function
def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

    label_map = {0: "fake", 1: "real"}
    label = label_map[preds.item()]
    confidence = conf.item()

    print(f"✅ Prediction: {label.upper()} ({confidence*100:.2f}% confidence)")
    return label, confidence

if __name__ == "__main__":
    model = load_model(MODEL_PATH)

    # 🔹 Example: change this path to any image you want to test
    test_image_path = r"C:\Users\ashta\visualguard\dataset\Celeb-DF Preprocessed\test\fake\id2_id9_0006_frame0_face1.jpg"

    if not os.path.exists(test_image_path):
        print("⚠️ Please update 'test_image_path' with a valid image file path.")
    else:
        predict_image(model, test_image_path)
