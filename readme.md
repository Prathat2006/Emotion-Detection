# 🎭 Emotion Detection API

This repository contains a **high-accuracy emotion detection system** built on top of cutting-edge deep learning models and a modern API-based architecture.
It detects human emotions from images using **Vision Transformers (ViT)** and provides **personalized multimedia feedback** through a custom frontend.

---

## 🚀 Features

* **High-Accuracy Emotion Detection**
  Uses the pretrained [`trpakov/vit-face-expression`](https://huggingface.co/trpakov/vit-face-expression) Vision Transformer model for robust, real-world facial emotion analysis.

* **Multi-Face Detection**
  Employs **MediaPipe**’s efficient face detection for accurate localization across lighting conditions, orientations, and multiple subjects.

* **Custom Frontend Integration**
  A dedicated frontend interacts with this FastAPI backend for seamless user experience, visualization, and audio playback.

* **Personalized Feedback**

  * Displays the **most confident** and **most common** emotions.
  * Shows an animated **GIF** corresponding to the detected emotion.
  * Plays a **mood-based song** (`.mp3`) mapped to the recognized emotion.

* **RESTful FastAPI Backend**

  * `/detect_emotion` endpoint accepts image uploads for real-time analysis.
  * `/gif/{emotion}` and `/song/{emotion}` endpoints serve emotion-specific visuals and audio.
  * Fully JSON-based responses for easy integration with any frontend or mobile app.

* **Debug-Ready Architecture**

  * Easy to extend with logging or debug image saving for development or dataset collection.

---

## 🧠 How It Works

1. **Input** → User uploads an image containing one or more faces.
2. **Detection** → Faces are identified using **MediaPipe Face Detection**.
3. **Analysis** → Each face is passed to the **ViT-based emotion model** for prediction.
4. **Aggregation** → Results are combined to compute:

   * The **most confident emotion** (highest probability).
   * The **most common emotion** among all faces.
5. **Feedback** → The API returns emotion metadata and paths to matching **GIFs** and **songs**.

---

## 🧩 Tech Stack

| Layer                  | Technology                                      |
| ---------------------- | ----------------------------------------------- |
| **Backend Framework**  | FastAPI                                         |
| **Model Architecture** | Vision Transformer (Hugging Face Transformers)  |
| **Face Detection**     | MediaPipe                                       |
| **Inference Engine**   | PyTorch                                         |
| **Frontend**           | Custom UI  |
| **Data I/O**           | JSON over HTTP                                  |
| **Optional Assets**    | GIFs (`./emoji/`), Songs (`./songs/`)           |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Prathat2006/Emotion-Detection.git
cd emotion-detection-api
```

### 2. Install dependencies

```bash
uv sync 

# or 
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare assets

Make sure these folders exist and contain your emotion resources:

```
./emoji/happy.gif
./emoji/sad.gif
./songs/happy.mp3
./songs/sad.mp3
```

---

## ▶️ Usage

### Run the FastAPI server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Endpoint          | Method | Description                                                                 |
| ----------------- | ------ | --------------------------------------------------------------------------- |
| `/detect_emotion` | `POST` | Accepts an image (`multipart/form-data`) and returns detected emotion data. |
| `/gif/{emotion}`  | `GET`  | Returns the GIF for a given emotion.                                        |
| `/song/{emotion}` | `GET`  | Returns the song (MP3) for a given emotion.                                 |

**Example (via cURL):**

```bash
curl -X POST "http://localhost:8000/detect_emotion" \
     -F "file=@face.jpg"
```

**Response Example:**

```json
{
  "most_confident_emotion": {
    "emotion": "happy",
    "confidence": 96.4
  },
  "most_common_emotion": {
    "emotion": "happy",
    "average_confidence": 91.7
  },
  "gif_path": "./emoji/happy.gif",
  "song_path": "./songs/happy.mp3"
}
```

---

## 🧰 Debug Mode

You can easily extend the code to log detections or save intermediate images by enabling debug utilities.
Simply add or uncomment the `save_debug_info()` calls in `detect_emotion()`.

---

## 🧑‍💻 Developers

| Name                 | GitHub                                                             |
| -------------------- | ------------------------------------------------------------------ |
| **Prathmesh Hatwar** | —                                                                  |
| **Kanchan Kumari**   | [github.com/KumariKanchan734](https://github.com/KumariKanchan734) |
| **Vikas Babu**       | [github.com/Vikas76-bit](https://github.com/Vikas76-bit)           |

---

## 🧾 License

This project is distributed under the **MIT License**.
You are free to use, modify, and distribute it with attribution.

---
