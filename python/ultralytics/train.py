from ultralytics import YOLO 
yolo_model = YOLO('yolov8l.pt')
 
yolo_model.train(data='cvtest.yaml',epochs=20,batch=4,device=0,workers=0,save_period=2)
