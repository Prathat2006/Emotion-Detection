import gradio as gr
from deepface import DeepFace
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import dlib
from retinaface import RetinaFace
from mtcnn import MTCNN


def save_debug_info(image, result):
    debug_dir = "./debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(debug_dir, f"input_{timestamp}.png")
    cv2.imwrite(image_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    
    result_path = os.path.join(debug_dir, f"result_{timestamp}.txt")
    with open(result_path, "w") as f:
        f.write(result)

def detect_faces_with_mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces

def detect_faces_with_retinaface(image):
    faces = RetinaFace.detect_faces(image)
    return faces

def detect_faces_with_yolo(image):
    net = cv2.dnn.readNetFromDarknet("yolov3-face.cfg", "yolov3-wider_16000.weights")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getLayers() if i[0] in net.getUnconnectedOutLayers()]
    
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    faces = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                faces.append((center_x, center_y, w, h))
    return faces

def detect_faces_with_dlib(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)
    return [(face.left(), face.top(), face.width(), face.height()) for face in faces]

def detect_faces_with_ssd(image):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

def detect_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = detect_faces_with_retinaface(image_rgb)
    if not faces:
        faces = detect_faces_with_mtcnn(image_rgb)
    if not faces:
        faces = detect_faces_with_yolo(image)
    if not faces:
        faces = detect_faces_with_dlib(image_rgb)
    if not faces:
        faces = detect_faces_with_ssd(image)
    
    return faces

def detect_emotion(image):
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        faces = detect_faces(image)
        
        if not faces:
            result = "No face detected in the image. Please upload an image with a clear face."
            save_debug_info(image, result)
            return result, None, None

        emotions = []
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

        for face in faces:
            dominant_emotion = analysis[0]['dominant_emotion']
            confidence = analysis[0]['emotion'][dominant_emotion]
            emotions.append((dominant_emotion, confidence))

        most_confident_emotion = max(emotions, key=lambda x: x[1])

        # Find the most common emotion
        emotion_counts = {}
        for emotion, _ in emotions:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        most_common_emotion = max(emotion_counts, key=emotion_counts.get)
        average_confidence = sum([conf for _, conf in emotions]) / len(emotions)

        result = f"Most Confident Emotion: {most_confident_emotion[0]} ({most_confident_emotion[1]:.2f}%)\nMost Common Emotion: {most_common_emotion} (Average Confidence: {average_confidence:.2f}%)"
        # save_debug_info(image, result) # Uncomment this line to save debug info
        song_path = get_emotion_song_path(most_confident_emotion[0])
        gif_path = f"./emoji/{most_confident_emotion[0]}.gif"
        if song_path and os.path.exists(song_path):
            with open(song_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            return result, gif_path, audio_content
        else:
            return result, gif_path, None

    except Exception as e:
        result = f"Error: {str(e)}"
        save_debug_info(image, result)
        return result, None, None

def get_emotion_song_path(emotion):
    songs = {
        "happy"    : "./songs/happy.mp3",
        "sad"      : "./songs/sad.mp3",
        "angry"    : "./songs/angry.mp3",
        "surprise" : "./songs/surprise.mp3",
        "fear"     : "./songs/fear.mp3",
        "neutral"  : "./songs/neutral.mp3",
        "disgust"  : "./songs/disgust.mp3"
    }
    
    return songs.get(emotion, None)

input_image = gr.Image(type="pil", label="Upload an Image")
output_label = gr.Label(label="Detected Emotion")
output_gif = gr.Image(type="filepath", label="Emotion GIF")
output_audio = gr.Audio(type="filepath", label="Listen to this music to change your emotion", autoplay=True)

gr.Interface(
    fn=detect_emotion,
    inputs=input_image,
    outputs=[output_label, output_gif, output_audio],
    title="Improved Emotion Detection with Advanced Models"
).launch(share=True, allowed_paths=["./emoji/"])
