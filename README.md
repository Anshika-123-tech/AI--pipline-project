🚀 AI Product Detection & Grouping Pipeline
📌 Project Overview

This project is an end-to-end AI pipeline system built using Flask that performs:

Product/Object detection using YOLOv8
Feature extraction using CLIP embeddings
Product grouping using cosine similarity
Visualization of grouped detections
JSON API response for integration

The system is designed as a modular AI microservice pipeline focusing on:

Low latency inference
Scalable architecture design
Clean API-based communication
🧠 System Architecture
Client (Image Upload)
        ↓
Flask API Server
        ↓
Detection Service (YOLOv8)
        ↓
Feature Extraction (CLIP)
        ↓
Grouping Service (Cosine Similarity Clustering)
        ↓
Visualization Service (OpenCV)
        ↓
Response (JSON + Output Image)
⚙️ Technologies Used
Python
Flask (Backend API)
Ultralytics YOLOv8 (Object Detection)
OpenAI CLIP (Feature Embeddings)
OpenCV (Image Processing & Visualization)
PyTorch (Deep Learning Backend)
NumPy
📦 Project Structure
infelict-ai/
│
├── app.py                  # Main Flask API + pipeline
├── temp/                  # Uploaded input images
├── outputs/               # Output visualized images
├── requirements.txt       # Dependencies
└── README.md
🔥 AI Pipeline Breakdown
1. Object Detection (YOLOv8)
Model: yolov8s.pt
Detects objects in the uploaded image
Outputs bounding boxes, class labels, confidence scores

Output Example:

{
  "bbox": [x1, y1, x2, y2],
  "class_name": "book",
  "confidence": 0.92
}
2. Feature Extraction (CLIP Model)
Uses ViT-B/32 CLIP model
Extracts semantic embeddings for each detected object crop
Converts each object into a high-dimensional feature vector
3. Product Grouping (Cosine Similarity Clustering)
Groups visually/semantically similar objects
Uses cosine similarity threshold (0.85)
Assigns unique group_id to each cluster

Note:

In absence of a labeled FMCG dataset, CLIP-based semantic grouping is used to simulate brand-level clustering.

4. Visualization
Bounding boxes drawn using OpenCV
Each group is color-coded
Labels include:
Group ID
Class name
🌐 API Endpoints
🔹 Home
GET /

Returns:

AI Pipeline Running 🚀
🔹 Upload Page
GET /upload

Simple HTML form to upload images.

🔹 Prediction API
POST /predict
Input:
Multipart form-data
Key: image
Output (JSON):
{
  "status": "success",
  "image_id": "unique-id",
  "total_objects": 3,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "class_name": "book",
      "confidence": 0.88,
      "group_id": 0
    }
  ],
  "groups": [
    {
      "group_id": 0,
      "count": 2,
      "items": ["book", "book"]
    }
  ],
  "output_image": "outputs/image_id.jpg"
}
🔹 Static Image Access
Input images:
/temp/<filename>
Output images:
/outputs/<filename>
🧪 How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Run Flask server
python app.py
3. Open in browser
http://127.0.0.1:5000/upload
4. API usage (Postman/cURL)

Send POST request to:

http://127.0.0.1:5000/predict

with:

key: image
type: file
🧩 Key Design Decisions
✔ Why YOLOv8?
Lightweight
Fast inference
Good general object detection performance
✔ Why CLIP?
Captures semantic similarity between objects
Works without labeled training data
Useful for grouping similar products
✔ Why Cosine Similarity?
Efficient for embedding comparison
Suitable for clustering visual features
⚡ Performance Considerations
Models loaded once at startup (reduces latency)
No repeated initialization
Lightweight Flask server
Local file-based caching for outputs
📌 Limitations
YOLOv8 COCO model is not trained on FMCG datasets
Grouping is semantic, not brand-trained classification
No distributed microservice deployment (single-node demo)
🚀 Future Improvements
Fine-tune YOLO on SKU-110K / retail datasets
Replace cosine grouping with trained clustering model
Add Redis/Celery for async processing
Deploy using Docker + Kubernetes
Add real-time streaming inference API
👨‍💻 Author Notes

This system demonstrates a complete AI pipeline architecture including:

Detection
Feature extraction
Clustering
Visualization
API deployment

Designed to be modular, scalable, and extendable for production-grade AI systems.
