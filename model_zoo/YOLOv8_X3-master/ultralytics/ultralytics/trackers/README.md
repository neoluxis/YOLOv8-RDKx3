# Tracker

## Supported Trackers

- [x] ByteTracker
- [x] BoT-SORT

## Usage

### python interface:

You can use the Python interface to track objects using the YOLO model.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model.track(
    source="video/streams",
    stream=True,
    tracker="botsort.yaml",  # or 'bytetrack.yaml'
    show=True,
)
```

You can get the IDs of the tracked objects using the following code:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

for result in model.track(source="video.mp4"):
    print(
        result.boxes.id.cpu().numpy().astype(int)
    )  # this will print the IDs of the tracked objects in the frame
```

If you want to use the tracker with a folder of images or when you loop on the video frames, you should use the `persist` parameter to tell the model that these frames are related to each other so the IDs will be fixed for the same objects. Otherwise, the IDs will be different in each frame because in each loop, the model creates a new object for tracking, but the `persist` parameter makes it use the same object for tracking.

```python
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("video.mp4")
model = YOLO("yolov8n.pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break
``````python
results = model.track(frame, persist=True)
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
ids = results[0].boxes.id.cpu().numpy().astype(int)
for box, id in zip(boxes, ids):
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Id {id}",
        (box[0], box[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
cv2.imshow("frame", frame)
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

## 更改跟踪器参数

您可以通过编辑位于 ultralytics/cfg/trackers 文件夹中的 `tracker.yaml` 文件来更改跟踪器参数。

## 命令行界面 (CLI)

您也可以使用命令行界面来使用 YOLO 模型跟踪物体。

```bash
yolo detect track source=... tracker=...
yolo segment track source=... tracker=...
yolo pose track source=... tracker=...
```

默认情况下，跟踪器将使用 `ultralytics/cfg/trackers` 中的配置。我们还支持使用修改后的跟踪器配置文件。请参阅 `ultralytics/cfg/trackers` 中的跟踪器配置文件。

## 贡献到我们的跟踪器部分

您是否精通多目标跟踪，并成功地使用 Ultralytics YOLO 实现或调整了跟踪算法？我们邀请您为我们的跟踪器部分做出贡献！您在实际应用和解决方案方面的经验对正在处理跟踪任务的用户可能非常宝贵。

通过为该部分做出贡献，您可以帮助扩大 Ultralytics YOLO 框架中可用的跟踪解决方案范围，为社区增加另一层功能和实用性。

要开始您的贡献，请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing)，详细说明如何提交拉取请求 (PR) 🛠️。我们很期待您能带来什么新的想法！

让我们一起增强 Ultralytics YOLO 生态系统的跟踪能力 🙏！
```