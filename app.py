import os
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp

app = FastAPI(title="Emotion Detection API")

# ----------------------------
# Load Emotion Model (ViT)
# ----------------------------
MODEL_NAME = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

# ----------------------------
# Initialize Mediapipe Face Detector
# ----------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ----------------------------
# Emotion → Song Map
# ----------------------------
def get_emotion_song_path(emotion: str):
    songs = {
        "happy": "./songs/happy.mp3",
        "sad": "./songs/sad.mp3",
        "angry": "./songs/angry.mp3",
        "surprise": "./songs/surprise.mp3",
        "fear": "./songs/fear.mp3",
        "neutral": "./songs/neutral.mp3",
        "disgust": "./songs/disgust.mp3"
    }
    return songs.get(emotion.lower(), None)

# ----------------------------
# Face Detection Function
# ----------------------------
def detect_faces(image: np.ndarray):
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        h, w, _ = image.shape
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x1, y1 = int(box.xmin * w), int(box.ymin * h)
            x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)
            faces.append((x1, y1, x2, y2))
    return faces

# ----------------------------
# Core Emotion Detection
# ----------------------------
def detect_emotion(image: np.ndarray):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    faces = detect_faces(image)
    if not faces:
        return {
            "error": "No face detected in the image. Please upload an image with a clear face.",
            "emotion": None,
            "confidence": None,
            "gif": None,
            "song": None
        }

    emotions = []
    for (x1, y1, x2, y2) in faces:
        face_crop = image[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        inputs = processor(images=face_pil, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = logits.softmax(dim=1).squeeze()
        
        top_prob, top_label = torch.max(probs, dim=0)
        emotion = model.config.id2label[int(top_label)]
        emotions.append((emotion, float(top_prob)))

    # Determine the most confident and most common emotions
    most_confident = max(emotions, key=lambda x: x[1])
    emotion_counts = {}
    for emo, _ in emotions:
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    most_common = max(emotion_counts, key=emotion_counts.get)
    avg_conf = np.mean([conf for _, conf in emotions])

    gif_path = f"./emoji/{most_confident[0].lower()}.gif"
    song_path = get_emotion_song_path(most_confident[0])

    response = {
        "most_confident_emotion": {
            "emotion": most_confident[0],
            "confidence": round(most_confident[1] * 100, 2)
        },
        "most_common_emotion": {
            "emotion": most_common,
            "average_confidence": round(avg_conf * 100, 2)
        },
        "gif_path": gif_path if os.path.exists(gif_path) else None,
        "song_path": song_path if song_path and os.path.exists(song_path) else None
    }

    return response

# ----------------------------
# FastAPI Endpoint
# ----------------------------
@app.post("/detect_emotion")
async def detect_emotion_api(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Run detection
        result = detect_emotion(image)
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------------
# Optional: Serve static GIFs or Songs
# ----------------------------
@app.get("/gif/{emotion_name}")
def get_gif(emotion_name: str):
    gif_path = f"./emoji/{emotion_name.lower()}.gif"
    if os.path.exists(gif_path):
        return FileResponse(gif_path, media_type="image/gif")
    return JSONResponse(content={"error": "GIF not found"}, status_code=404)

@app.get("/song/{emotion_name}")
def get_song(emotion_name: str):
    song_path = get_emotion_song_path(emotion_name)
    if song_path and os.path.exists(song_path):
        return FileResponse(song_path, media_type="audio/mpeg")
    return JSONResponse(content={"error": "Song not found"}, status_code=404)

@app.get("/")
def home():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=6868, reload=True)