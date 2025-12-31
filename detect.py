from transformers import AutoImageProcessor, AutoModelForImageClassification
from insightface.app import FaceAnalysis
import cv2
import torch
import numpy as np
from PIL import Image


# --------------------------------------------------
# Device (CUDA → CPU fallback)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Torch device: {device}")

# --------------------------------------------------
# Emotion Model (ViT)
# --------------------------------------------------
MODEL_NAME = "trpakov/vit-face-expression"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
emotion_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
emotion_model.to(device)
emotion_model.eval()

# --------------------------------------------------
# Face Detector (SCRFD via InsightFace)
# --------------------------------------------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
    if torch.cuda.is_available() else ["CPUExecutionProvider"]

face_detector = FaceAnalysis(
    name="buffalo_l",   # SCRFD + landmarks
    providers=providers
)
face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

print("[INFO] Face detector ready")


def detect_faces(image: np.ndarray):
    faces = face_detector.get(image)
    boxes = []

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        boxes.append((x1, y1, x2, y2))

    return boxes

# --------------------------------------------------
# Emotion Detection
# --------------------------------------------------
def detect_emotion(image: np.ndarray):
    faces = detect_faces(image)

    if not faces:
        return {
            "error": "No face detected",
            "emotion": None
        }

    results = []

    for (x1, y1, x2, y2) in faces:
        face_crop = image[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        inputs = processor(images=face_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = emotion_model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()

        conf, label = torch.max(probs, dim=0)
        emotion = emotion_model.config.id2label[int(label)]

        results.append({
            "emotion": emotion,
            "confidence": round(conf.item() * 100, 2)
        })

    # Pick most confident face
    best = max(results, key=lambda x: x["confidence"])

    return {
        "faces_detected": len(results),
        "emotion": best["emotion"],
        "confidence": best["confidence"]
    }