from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import os

from model import get_model

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_model_path() -> str:
    """Resolve checkpoint path relative to project root, not the shell cwd."""
    candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints_dryrun", "best_model.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def validate_chest_xray(img: Image.Image) -> tuple[bool, str]:
    error_msg = "The file isn't an X-ray image. Please upload a different image"
    try:
        # 1. Format/Dimensions check
        width, height = img.size
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, error_msg

        img_np = np.array(img)
        if len(img_np.shape) >= 3 and img_np.shape[2] >= 3:
            # 2. Color check: Check if there are colored pixels
            max_c = img_np[:, :, :3].max(axis=-1).astype(float)
            min_c = img_np[:, :, :3].min(axis=-1).astype(float)
            diff = max_c - min_c
            colored_pixels_ratio = np.mean(diff > 10)

            if colored_pixels_ratio > 0.005:  # more than 0.5% pixels have distinct color
                return False, error_msg

        # 3. Contrast check: X-rays have rib/spine structures which create high contrast
        gray_img = img.convert('L')
        gray_np = np.array(gray_img)
        std_dev = gray_np.std()

        if std_dev < 15.0:
            return False, error_msg

        # 4. Brightness distribution check (borders vs center)
        h, w = gray_np.shape
        center_y_start = int(h * 0.25)
        center_y_end = int(h * 0.75)
        center_x_start = int(w * 0.25)
        center_x_end = int(w * 0.75)
        center_pixels = gray_np[center_y_start:center_y_end, center_x_start:center_x_end]
        mean_center = np.mean(center_pixels)

        if mean_center < 35.0:
            return False, error_msg

        if mean_center > 240.0:
            return False, error_msg

        return True, ""
    except Exception:
        return False, error_msg


CLASS_NAMES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Guard for platforms without MPS backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def process_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomEqualize(p=1.0),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image).unsqueeze(0)


def load_model(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"

    try:
        # Fix for SSL certificate errors on some systems
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

        model = get_model(num_classes=len(CLASS_NAMES), pretrained=False)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


app = Flask(__name__)
# Allow requests from Vite dev server during development
CORS(app)

# Optimize PyTorch CPU resources for low-RAM hosts (like Render Free Tier)
torch.set_num_threads(1)

device = get_device()

model_path = resolve_model_path()
model, model_error = load_model(model_path, device)

if model is None:
    print(f"WARNING: {model_error}")
    print("Run 'python save_dummy.py' from the project root, then restart this server.")
else:
    print(f"Model loaded from {model_path}")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "error": model_error,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            jsonify(
                {
                    "error": (
                        f"Model not loaded: {model_error}. "
                        "Run 'python save_dummy.py' from the project root, "
                        "then restart the API server (python src/server.py)."
                    )
                }
            ),
            500,
        )

    if "file" not in request.files:
        return (
            jsonify({"error": "No file part in the request (expected key 'file')"}),
            400,
        )

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Validate image is a chest X-ray
    is_valid, validation_msg = validate_chest_xray(image)
    if not is_valid:
        return jsonify({"error": validation_msg}), 400

    try:
        img_tensor = process_image(image).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        top_prob, top_idx = torch.max(probs, 0)
        predicted_class = CLASS_NAMES[top_idx.item()]
        confidence = float(top_prob)

        probabilities = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        return jsonify(
            {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For development only
    app.run(host="0.0.0.0", port=5002, debug=True)

