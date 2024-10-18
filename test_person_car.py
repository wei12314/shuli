from ultralytics import YOLO
from PIL import Image
import cv2
import os

if __name__ == "__main__":
    test_dir = "/home/bdhapp/shuli/test_img/img/person_car/car"
    write_dir = "/home/bdhapp/shuli/test_img/output/person_car_test1"
    # Load a COCO-pretrained YOLOv8n model
    model_type = "yolo11s"
    model_path = "/home/bdhapp/shuli/yolo11s.pt"
    model = YOLO(model_path)
    cls_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    person_car = [0, 1, 2, 3, 5, 6, 7]
    cls = []
    batch_path = []
    n = 0
    cnt = 0

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        write_path = os.path.join(write_dir, f"{model_type}_{img_name}")
        img = Image.open(img_path)
        batch_path.append(img)
        break

    results = model.predict(source=batch_path, save=False, conf=0.1)
    for r in results:
        # print(r.boxes[0])
        boxes = r.boxes
        cur_cls = boxes.cls.cpu().tolist()
        cls.append(cur_cls)
        
    n += 1

    for nums in cls:
        for num in nums:
            int_num = int(num)
            if int_num in person_car:
                cnt += 1
                break
    print(n, cnt)
    print(f"accuracy: {round(cnt / n, 2) * 100}%")
            
        