# Emotion Detection System

This repository contains an advanced emotion detection system that leverages state-of-the-art facial recognition models and emotion analysis techniques to detect human emotions from images. It also provides personalized audio and visual feedback to enhance the user's experience.

---

## Features

- **Emotion Detection**: Analyzes the emotions present in an image using the `DeepFace` library.
- **Multi-Face Detection**: Employs multiple face detection models (RetinaFace, MTCNN, YOLO, Dlib, and SSD) for robust detection in various scenarios.
- **Personalized Feedback**:
  - Displays the dominant emotion.
  - Shows a GIF corresponding to the detected emotion.
  - Plays a song to match or alter the user's mood.
- **Debug Mode**: Option to save debug information, including the input image and results.

---

## How It Works

1. **Input**: Upload an image containing a clear face.
2. **Detection**: The system detects faces in the image using advanced face detection models.
3. **Emotion Analysis**: The `DeepFace` library analyzes the dominant emotion and its confidence level.
4. **Feedback**:
   - Shows the most confident and most common emotions.
   - Displays an emoji GIF matching the emotion.
   - Plays an audio track tailored to the detected emotion.

---

## Requirements

### Python Dependencies

Install the required Python packages using the command:
```bash
pip install -r requirements.txt
```

---

## Usage

Run the application:
```bash
python main.py
```

1. Open the Gradio interface.
2. Upload an image with a clear face.
3. View the detected emotion, GIF, and listen to the mood-specific audio.

---

## Debugging

Enable debug mode by uncommenting the `save_debug_info` lines in the code.
Debug information (input image and results) will be saved in the `debug/` directory.

---

## Credits

This project was developed by:

- Prathmesh Hatwar
- Kanchan Kumari
- Vikas Babu


