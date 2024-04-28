import ultralytics
from ultralytics import YOLO

print(ultralytics.__file__)

model = YOLO(model="yolov8n.yaml")
# model = YOLO('C:/QTProj/yolov8/model_zoo/runs/detect/train5/weights/best.pt')

if __name__ == "__main__":
    model.info()

    # 训练前不要忘记修改 head.py 的 forward 方法
    model.train(data="horizon1.yaml", epochs=400, device=0)
    model.export(format="onnx", opset=11, simplify=True)
