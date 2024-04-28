[English](./README.md) | 简体中文

[YOLOv8](https://github.com/ultralytics/ultralytics)是一种前沿的、最先进的模型，它在之前的YOLO版本的成功基础上进行了改进，引入了新功能和改进，以进一步提高性能和灵活性。 YOLOv8旨在设计成快速、准确且易于使用，使其成为广泛应用于目标检测和跟踪、实例分割、图像分类和姿势估计等多种任务的理想选择。

Horizon Algorithm Toolchain（也称为OpenExplorer）是由地平线机器人开发的自研工具，用于高效准确地将模型部署到地平线SoCs。它以开发包的形式发布，可通过[Horizon Algorithm Toolchain XJ3](https://developer.horizon.cc/forumDetail/136488103547258769)和[Horizon Algorithm Toolchain J5](https://developer.horizon.cc/forumDetail/118363912788935318)获取。

该项目旨在通过Horizon Algorithm Toolchain将YOLOv8模型有效地部署到地平线机器人的Sunrise-3 SoCs。部署流程包括浮点模型训练和导出为ONNX格式、模型量化和优化、模型编译以及模型部署（仅在板上展示模型性能评估）。

## <div align="center">快速开始</div>

<details open>
<summary>安装</summary>

#### YOLOv8
`yolov8_x3/ultralytics`文件夹包含官方的YOLOv8开发工具，以及适用于地平线平台的修改模型结构。有关详细的模型修改说明，请参阅[【前沿算法】地平线适配 YOLOv8](https://developer.horizon.cc/forumDetail/189779523032809473)。运行以下命令进行安装：
```bash
cd yolov8_x3/ultralytics
pip install -r requirements.txt
python setup.py install
```

#### Horizon Algorithm Toolchain
Horizon Algorithm Toolchain提供Docker镜像，为用户快速访问工具。只需运行`run_docker.sh`脚本：
```bash
sh run_docker.sh ./data/ gpu
pip uninstall horizon-nn
pip install ddk/package/host/ai_toolchain/horizon_nn_gpu-0.18.2-cp38-cp38-linux_x86_64.whl
```
这将指定数据集路径并在GPU模式下运行docker（请确保GPU环境设置正确）。

</details>

<details open>
<summary>使用</summary>

#### 构建
这将使Horizon Algorithm Toolchain自动将ONNX模型转换为可在地平线SoCs上部署的格式。
```bash
sh build.sh
```

#### 在主机端评估
在运行以下命令之前，请下载[coco_val2017](https://cocodataset.org/)到`yolov8_x3/ptq_project/coco`文件夹中，并在`evaluate.sh`文件中指定当前数据集路径。

评估具有完整coco_val2017数据集的原始ONNX模型精度：
```bash
sh evaluate.sh origin
```

评估具有完整coco_val2017数据集的量化模型精度：
```bash
sh evaluate.sh quanti
``````
To evaluate the accuracy of original ONNX model just for test:
```bash
sh evaluate.sh origin 20
```

To evaluate the accuracy of quantized model just for test:
```bash
sh evaluate.sh quanti 20
``` 
```