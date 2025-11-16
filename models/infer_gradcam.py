import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Path to your saved model
MODEL_PATH = r"models\image_model\best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------
# Grad-CAM Helper Class
# -----------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = torch.mean(grads, dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * acts, dim=0)
        cam = torch.relu(cam).cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= cam.max() if cam.max() != 0 else 1
        return cam

# -----------------------------------------------------------------
# Model loader
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# Grad-CAM Prediction
# -----------------------------------------------------------------
def gradcam_predict(model, image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

    label_map = {0: "fake", 1: "real"}
    label = label_map[preds.item()]
    confidence = conf.item()

    print(f" Prediction: {label.upper()} ({confidence*100:.2f}% confidence)")

    # Grad-CAM visualization
    target_layer = model.layer4[-1]
    gc = GradCAM(model, target_layer)
    cam = gc.generate(input_tensor, preds.item())

    # Convert to overlay
    orig = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * heatmap + 0.4 * orig).astype(np.uint8)

    # Show & save
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(orig)
    plt.title(f"Original ({label})")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Explanation")
    plt.axis("off")

    out_path = os.path.join("results", f"gradcam_{os.path.basename(image_path)}")
    os.makedirs("results", exist_ok=True)
    Image.fromarray(overlay).save(out_path)
    print(f" Grad-CAM saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    test_image_path = r"C:\Users\ashta\OneDrive\Attachments\Pictures\Camera Roll\WIN_20250217_21_24_14_Pro.jpg"
    if not os.path.exists(test_image_path):
        print(" Please update 'test_image_path' to a valid image path.")
    else:
        gradcam_predict(model, test_image_path)
