from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

app = FastAPI()

# لود مدل YOLOv5 (local model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    results = model(img)
    # نتایج را به دیکشنری تبدیل می‌کنیم
    return JSONResponse(content=results.pandas().xyxy[0].to_dict(orient="records"))

print("hi")
ss
