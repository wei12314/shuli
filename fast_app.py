from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import json

app = FastAPI()

# 全局变量，用于存储模型
model_path = "/home/bdhapp/shuli/runs/detect/train16/weights/best.pt"
model = YOLO(model_path)

# 定义返回的坐标格式
class DetectionResult(BaseModel):
    confidence: List[float]
    class_id: List[int]
    rect: List[List[int]]


# 处理图片上传并返回检测结果
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    global model
    image = Image.open(BytesIO(await file.read()))
    
    # 对图片进行推理
    results = model.predict(source=image, save=False, conf=0.3, iou=0.3, imgsz=1024, max_det=1000)
    result = results[0]
    print(result)
    # 将检测结果转换为json格式
    detections = []
    for result in results:
        boxes = result.boxes
        confidences, class_ids = boxes.conf.tolist(), boxes.cls.int().tolist()
        rects = boxes.xyxy.int().tolist()
        # bbox = result[:4].tolist()  # x_min, y_min, x_max, y_max
        # confidence = float(result[4])
        # class_id = int(result[5])
        # detections.append(DetectionResult(bbox=bbox, confidence=confidence, class_id=class_id))
        # print(type(confidence), type(class_ids), type(rects))
        # print(confidence, class_ids, rects)
        detections.append(DetectionResult(confidence = confidences, class_id = class_ids, rect = rects))
    # size = image.size
    # res = [size[0], size[1]]
    return {"detections": detections}

