import os
import torch
import faiss
import numpy as np
import open_clip

from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import zipfile
import io

torch.set_default_dtype(torch.float16)  # Use lower precision to save memory

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B/32", device=device)

index_path = "faiss_index.idx"
dim = 512

# Load FAISS index if available
index = faiss.read_index(index_path) if os.path.exists(index_path) else faiss.IndexFlatL2(dim)

labels_path = "labels.npy"
label_map = np.load(labels_path, allow_pickle=True).item() if os.path.exists(labels_path) else {}

app = Flask(__name__)
CORS(app)

def process_images(images, label):
    global index, label_map

    for img in images:
        image = Image.open(img).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy()

        index.add(image_features)
        label_map[index.ntotal - 1] = label

        torch.cuda.empty_cache()  # Free GPU memory

    faiss.write_index(index, index_path)
    np.save(labels_path, label_map)

@app.route("/")
def home():
    return jsonify({"message": "API is running!"})

@app.route("/api/train-multiple", methods=["POST"])
def train_multiple():
    if "label" not in request.form or "images" not in request.files:
        return jsonify({"error": "Label and images are required"}), 400

    label = request.form["label"]
    images = request.files.getlist("images")

    if not images:
        return jsonify({"error": "No images provided"}), 400

    process_images(images, label)
    return jsonify({"message": "Training successful", "label": label, "images_trained": len(images)})

@app.route("/api/train-folder", methods=["POST"])
def train_folder():
    if "zipfile" not in request.files:
        return jsonify({"error": "Zip file is required"}), 400

    zip_file = request.files["zipfile"]
    folder_name = os.path.splitext(zip_file.filename)[0]
    images = []

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                with zip_ref.open(file_name) as img_file:
                    images.append(io.BytesIO(img_file.read()))

    if not images:
        return jsonify({"error": "No valid images in folder"}), 400

    process_images(images, folder_name)
    return jsonify({"message": "Training successful", "label": folder_name, "images_trained": len(images)})

@app.route("/api/predict", methods=["POST"])
def predict():
    if "test_image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(request.files["test_image"]).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()

    if index.ntotal == 0:
        return jsonify({"error": "No trained data available"}), 400

    _, closest = index.search(image_features, 1)
    predicted_label = label_map.get(int(closest[0][0]), "Unknown")

    torch.cuda.empty_cache()
    return jsonify({"predicted_label": predicted_label})

@app.route("/api/unlearn", methods=["POST"])
def unlearn():
    global index, label_map

    if "label" not in request.json:
        return jsonify({"error": "Label is required"}), 400

    label_to_remove = request.json["label"]
    indices_to_remove = [i for i, lbl in label_map.items() if lbl == label_to_remove]

    if not indices_to_remove:
        return jsonify({"error": "Label not found"}), 404

    indices_to_remove = np.array(indices_to_remove, dtype=np.int64)
    index.remove_ids(indices_to_remove)

    label_map = {i: lbl for i, lbl in label_map.items() if lbl != label_to_remove}
    faiss.write_index(index, index_path)
    np.save(labels_path, label_map)

    return jsonify({"message": f"Successfully removed label: {label_to_remove}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
