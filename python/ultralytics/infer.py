from ultralytics import YOLO
import torch
import torch.nn as nn

if __name__ == '__main__':
# 加载模型
    model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    model = YOLO("runs/detect/train/weights/best.pt")  # 加载预训练模型（建议用于训练）
    #metrics = model.val()  # 在验证集上评估模型性能
    #results = model("asd.png") 
    #success = model.export(format="onnx",imgsz=2048)
    model.predict('1.mp4', save=True, imgsz=2048, conf=0.5,device="cpu")
    print("end")