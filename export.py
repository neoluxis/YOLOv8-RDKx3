import ultralytics
from ultralytics import YOLO
import cv2 as cv

print(ultralytics.__file__)

model = YOLO(model="yolov5s.yaml")
# model = YOLO('C:/QTProj/yolov8/model_zoo/runs/detect/train8/weights/best.pt')

if __name__ == "__main__":
    model.info()

    # 训练前不要忘记修改 head.py 的 forward 方法
    model.train(data="horizon1.yaml", epochs=1000, device=0, project='V5')
    # model.export(format="onnx", opset=11, simplify=True)
