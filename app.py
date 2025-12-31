import cv2 
import numpy as np
from detect import detect_emotion
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Emotion Detection API (GPU-ready)")

@app.post("/detect_emotion")
async def detect_emotion_api(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = detect_emotion(image)
    return JSONResponse(content=result)

@app.get("/")
def home():
    return FileResponse("index.html")


app.mount("/emoji", StaticFiles(directory="emoji"), name="emoji")
app.mount("/songs", StaticFiles(directory="songs"), name="songs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=6868)
