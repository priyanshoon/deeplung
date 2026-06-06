from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import os

from model import get_model


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

        model = get_model(num_classes=len(CLASS_NAMES))
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

device = get_device()

# Determine model path automatically (same logic as streamlit app)
default_path = "checkpoints/best_model.pth"
if not os.path.exists(default_path) and os.path.exists(
    "checkpoints_dryrun/best_model.pth"
):
    default_path = "checkpoints_dryrun/best_model.pth"

model, model_error = load_model(default_path, device)


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
        return jsonify({"error": f"Model not loaded: {model_error}"}), 500

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
    app.run(host="0.0.0.0", port=5000, debug=True)

