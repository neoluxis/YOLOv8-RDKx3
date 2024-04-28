import ultralytics
from ultralytics import YOLO

print(ultralytics.__file__)

model = YOLO("yolov8n.yaml")

if __name__ == '__main__':
    model.info()

    # model.train(data="coco128.yaml", epochs=2, workers=0)

    model.export(format="onnx", opset=11)
