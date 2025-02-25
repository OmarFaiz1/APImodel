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

app = Flask(__name__)
CORS(app)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B/32", device=device)

# FAISS index (512-dimensional vectors)
index_path = "faiss_index.idx"
dim = 512

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatL2(dim)

# Load label mapping
labels_path = "labels.npy"
if os.path.exists(labels_path):
    label_map = np.load(labels_path, allow_pickle=True).item()
else:
    label_map = {}

# 🔹 Function to process images and update FAISS index
def process_images(images, label):
    global index, label_map

    feature_list = []
    for img in images:
        image = Image.open(img).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy()
            feature_list.append(image_features)

    features = np.vstack(feature_list)
    index.add(features)

    # Update label map
    for i in range(len(feature_list)):
        label_map[index.ntotal - len(feature_list) + i] = label

    # Save FAISS index & labels
    faiss.write_index(index, index_path)
    np.save(labels_path, label_map)

@app.route("/")
def home():
    return jsonify({"message": "API is running!"})

# 🔹 API to train model with multiple images
@app.route("/api/train-multiple", methods=["POST"])
def train_multiple():
    if "label" not in request.form or "images" not in request.files:
        return jsonify({"error": "Label and images are required"}), 400

    label = request.form["label"]
    images = request.files.getlist("images")

    if len(images) == 0:
        return jsonify({"error": "No images provided"}), 400

    process_images(images, label)

    return jsonify({"message": "Training successful", "label": label, "images_trained": len(images)})

# 🔹 API to train using a folder (folder name = label)
@app.route("/api/train-folder", methods=["POST"])
def train_folder():
    if "zipfile" not in request.files:
        return jsonify({"error": "Zip file is required"}), 400

    zip_file = request.files["zipfile"]
    folder_name = os.path.splitext(zip_file.filename)[0]  # Folder name as label

    images = []
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                with zip_ref.open(file_name) as img_file:
                    images.append(io.BytesIO(img_file.read()))

    if len(images) == 0:
        return jsonify({"error": "No valid images in the folder"}), 400

    process_images(images, folder_name)

    return jsonify({"message": "Training successful", "label": folder_name, "images_trained": len(images)})

# 🔹 Bulk training API
@app.route("/api/train-bulk", methods=["POST"])
def train_bulk():
    if "zipfile" not in request.files:
        return jsonify({"error": "Zip file is required"}), 400

    zip_file = request.files["zipfile"]
    images_by_label = {}

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                label = os.path.dirname(file_name)  # Extract the folder name as label
                if label:
                    if label not in images_by_label:
                        images_by_label[label] = []
                    with zip_ref.open(file_name) as img_file:
                        images_by_label[label].append(io.BytesIO(img_file.read()))

    if not images_by_label:
        return jsonify({"error": "No valid images found"}), 400

    # Process images for each label
    for label, images in images_by_label.items():
        process_images(images, label)

    return jsonify({"message": "Bulk training successful", "labels_trained": list(images_by_label.keys())})

    if "zipfile" not in request.files:
        return jsonify({"error": "Zip file is required"}), 400

    zip_file = request.files["zipfile"]
    images_by_label = {}

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            parts = file_name.split("/")
            if len(parts) > 1 and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                label = parts[0]  # Folder name as label
                if label not in images_by_label:
                    images_by_label[label] = []
                with zip_ref.open(file_name) as img_file:
                    images_by_label[label].append(io.BytesIO(img_file.read()))

    if not images_by_label:
        return jsonify({"error": "No valid images found"}), 400

    for label, images in images_by_label.items():
        process_images(images, label)

    return jsonify({"message": "Bulk training successful", "labels_trained": list(images_by_label.keys())})

# 🔹 Prediction API
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

    return jsonify({"predicted_label": predicted_label})

# 🔹 Unlearn API - Remove label from FAISS index
@app.route("/api/unlearn", methods=["POST"])
def unlearn():
    global index, label_map

    if "label" not in request.json:
        return jsonify({"error": "Label is required"}), 400

    label_to_remove = request.json["label"]

    # Find indices associated with the label
    indices_to_remove = [i for i, lbl in label_map.items() if lbl == label_to_remove]

    if not indices_to_remove:
        return jsonify({"error": "Label not found"}), 404

    # Convert indices to FAISS array format
    indices_to_remove = np.array(indices_to_remove, dtype=np.int64)

    # Remove vectors from FAISS index
    index.remove_ids(indices_to_remove)

    # Remove label from label_map
    label_map = {i: lbl for i, lbl in label_map.items() if lbl != label_to_remove}

    # Save updated index and labels
    faiss.write_index(index, index_path)
    np.save(labels_path, label_map)

    return jsonify({"message": f"Successfully removed label: {label_to_remove}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
