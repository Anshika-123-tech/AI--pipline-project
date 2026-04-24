from flask import Flask, request, jsonify, render_template_string, send_from_directory
import os
import uuid
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import clip
from PIL import Image

# =========================
# 🔥 Load Models (ONCE)
# =========================
model = YOLO("yolov8s.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

app = Flask(__name__)

# Create folders
os.makedirs("temp", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================
# ✅ Detection Function (Improved)
# =========================
def detect_objects(image_path):
    results = model(image_path, conf=0.15)

    detections = []

    for r in results:
        names = r.names

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "bbox": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)],
                "confidence": round(conf, 3),
                "class_id": cls,
                "class_name": names[cls]
            })

    return detections


# =========================
# ✅ CLIP Feature Extraction
# =========================
def extract_features(image_path, detections):
    img = cv2.imread(image_path)
    features = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            features.append(np.zeros(512))
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = clip_model.encode_image(input_tensor)

        embedding = embedding.cpu().numpy().flatten()

        # normalize
        if np.linalg.norm(embedding) != 0:
            embedding = embedding / np.linalg.norm(embedding)

        features.append(embedding)

    return np.array(features)


# =========================
# ✅ CLIP-based Grouping (REPLACED)
# =========================
def group_products(image_path, detections):
    if len(detections) == 0:
        return detections

    features = extract_features(image_path, detections)

    groups = []
    group_ids = []

    threshold = 0.85  # cosine similarity

    for f in features:
        assigned = False

        for i, g in enumerate(groups):
            similarity = np.dot(f, g)

            if similarity > threshold:
                group_ids.append(i)
                assigned = True
                break

        if not assigned:
            groups.append(f)
            group_ids.append(len(groups) - 1)

    for i, det in enumerate(detections):
        det["group_id"] = int(group_ids[i])

    return detections


# =========================
# ✅ Draw Boxes (Minor upgrade)
# =========================
def draw_boxes(image_path, detections, output_path):
    img = cv2.imread(image_path)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255)
    ]

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        gid = det.get("group_id", 0)

        color = colors[gid % len(colors)]

        label = f"G{gid} | {det.get('class_name', '')}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, img)


# =========================
# ✅ Routes (UNCHANGED)
# =========================
@app.route('/')
def home():
    return "AI Pipeline Running 🚀"


@app.route('/upload')
def upload_page():
    return render_template_string("""
        <h2>Upload Image</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <br><br>
            <button type="submit">Upload</button>
        </form>
    """)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']

    # Save image
    image_id = str(uuid.uuid4())
    path = f"temp/{image_id}.jpg"
    file.save(path)

    # 🔥 Pipeline
    detections = detect_objects(path)
    detections = group_products(path, detections)

    output_path = f"outputs/{image_id}.jpg"
    draw_boxes(path, detections, output_path)

    # 🔥 Group summary
    groups = {}
    for det in detections:
        gid = det["group_id"]
        if gid not in groups:
            groups[gid] = {"group_id": gid, "count": 0}
        groups[gid]["count"] += 1

    group_list = list(groups.values())

    # =========================
    # Browser UI
    # =========================
    if request.content_type and "multipart/form-data" in request.content_type:
        return render_template_string("""
            <h3>Pipeline Completed ✅</h3>

            <p><b>Original Image:</b></p>
            <img src="/{{input_path}}" width="300"><br><br>

            <p><b>Output Image (Grouped):</b></p>
            <img src="/{{output_path}}" width="300"><br><br>

            <p><b>Group Summary:</b></p>
            <pre>{{groups}}</pre>

            <p><b>Detections:</b></p>
            <pre>{{detections}}</pre>

            <br>
            <a href="/upload">Upload Another</a>
        """, input_path=path, output_path=output_path,
           detections=detections, groups=group_list)

    # =========================
    # JSON API
    # =========================
    return jsonify({
        "image_id": image_id,
        "total_objects": len(detections),
        "groups": group_list,
        "detections": detections,
        "output_image": output_path
    })


# =========================
# ✅ Serve Images
# =========================
@app.route('/temp/<filename>')
def get_temp_image(filename):
    return send_from_directory('temp', filename)


@app.route('/outputs/<filename>')
def get_output_image(filename):
    return send_from_directory('outputs', filename)


# =========================
# ✅ Run Server
# =========================
if __name__ == '__main__':
    app.run(debug=True)