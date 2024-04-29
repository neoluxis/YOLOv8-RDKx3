from ultralytics import YOLO
import cv2 as cv

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')

# # Export the model to ONNX format
# model.export(format='onnx')  # creates 'yolov8n.onnx'

# Load the exported ONNX model
onnx_model = YOLO('C:/QTProj/yolov8/model_zoo/runs/detect/train8/weights/best.onnx')

# Run inference
results = onnx_model('../znc_datasets/horizon/images/52.jpg')
# print(len(results))  # 3 models in the ensemble
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
