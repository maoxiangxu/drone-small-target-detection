# 无人机小目标检测系统

> 基于 YOLOv5-CBAM 与千问-VL 的级联目标检测框架

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-GPL%203.0-yellow)](LICENSE)

针对复杂背景下**无人机与飞鸟**外形高度相似、小目标特征提取困难的问题，提出了一种结合 CBAM 注意力机制与千问-VL 大语言视觉模型的两阶段级联识别方法，并基于 PyQt5 开发了可视化检测系统。

<p align="center">
  <img src="https://img.shields.io/badge/检测类别-drone%20%7C%20bird-0ea5e9" alt="classes">
  <img src="https://img.shields.io/badge/模型-YOLOv5s__CBAM-06b6d4" alt="model">
  <img src="https://img.shields.io/badge/级联框架-Qwen--VL%20%2B%20YOLOv5-8b5cf6" alt="cascade">
</p>

---

## 主要特点

- **CBAM 注意力机制**：在 YOLOv5 主干网络引入 CBAM (Convolutional Block Attention Module)，通过通道与空间双重特征权重分配，抑制复杂林地/天空等高频背景噪声
- **两阶段级联检测**：千问-VL 大模型粗定位 → YOLOv5s_CBAM 精检测，有效降低复杂背景对小目标特征的干扰
- **PyQt5 可视化界面**：支持图片选择、置信度阈值调节、检测模式切换、结果保存等功能
- **双模式运行**：纯 YOLO 本地检测（快速）/ 千问+YOLO 级联检测（复杂背景高精度）

## 系统架构

```
┌────────────────────────────────────────────────────┐
│                  PyQt5 可视化界面                    │
├────────────────────────────────────────────────────┤
│  模式一: YOLOv5s_CBAM 直接检测                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ 输入图像  │ → │ Backbone+CBAM │ → │ 检测结果  │  │
│  └──────────┘    └──────────────┘    └──────────┘  │
│                                                     │
│  模式二: Qwen-VL + YOLOv5s_CBAM 级联检测            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ 输入图像  │→│ Qwen-VL  │→│ YOLOv5s_CBAM │→ 结果│
│  │          │  │ 粗定位    │  │ 区域精检      │     │
│  └──────────┘  └──────────┘  └──────────────┘     │
└────────────────────────────────────────────────────┘
```

### CBAM 模块位置

在 Backbone 的 P5 层（SPPF 之前）引入 CBAM 注意力模块：

```
Backbone: Conv → C3 → Conv → C3 → Conv → C3 → Conv → C3 → [CBAM] → SPPF
                                                                ↑
                                                        通道+空间注意力
```

## 快速开始

### 环境要求

- Python 3.8 ~ 3.10
- PyTorch ≥ 1.9.0
- CUDA 11.7+（可选，CPU 模式也可运行）

### 安装

```bash
# 克隆仓库
git clone https://github.com/maoxiangxu/drone-small-target-detection.git
cd drone-small-target-detection

# 安装依赖
pip install -r requirements.txt
pip install PyQt5>=5.15

# 千问-VL 模式需要（可选）
pip install dashscope
```

### 运行可视化界面

```bash
python drone_detector_qt.py
```

操作流程：
1. 点击 **打开图片** 选择待检测图像
2. 选择检测模式（YOLO+CBAM / 千问+YOLO）
3. 调节置信度阈值滑块
4. 点击 **开始检测**
5. 点击 **保存结果** 导出标注图像

### 命令行检测

```bash
# 使用训练好的模型进行检测
python detect.py --weights runs/train/exp2/weights/best.pt --source your_image.jpg
```

## 项目结构

```
├── drone_detector_qt.py          # PyQt5 可视化检测主程序
├── detect.py                     # 命令行检测脚本
├── train.py                      # 模型训练脚本
├── val.py                        # 模型验证脚本
├── models/
│   ├── common.py                 # 模型组件（含 CBAM 实现）
│   ├── yolo.py                   # YOLO 模型结构
│   └── yolov5s_CBAM.yaml         # CBAM 增强模型配置
├── data/
│   └── drone_vs_bird.yaml        # 无人机-飞鸟数据集配置
├── utils/                        # 工具函数库
├── runs/train/exp2/weights/
│   └── best.pt                   # 训练好的模型权重
└── requirements.txt              # Python 依赖
```

## 数据集

本项目的训练数据集为自建无人机-飞鸟图像数据集，包含复杂自然场景（林地、天空、城市等）下的两类目标：

| 类别 | 标签 | 说明 |
|------|------|------|
| 无人机 | drone | 各类消费级/工业级旋翼无人机 |
| 飞鸟 | bird | 常见鸟类（易与无人机混淆的目标） |

## 可视化界面

三栏布局设计：

- **左栏**：控制面板（操作按钮、检测模式、置信度阈值、API Key 设置）
- **中栏**：双图对比区（原始图像 ↔ 检测结果）
- **右栏**：统计面板（目标统计、类别分布、图片信息、检测日志）

<img width="1504" height="934" alt="image" src="https://github.com/user-attachments/assets/3aacfb85-436b-4080-b65c-361c66d8be63" />


## 引用

如果你的研究使用了本项目，请引用：

```
@thesis{xu2026drone,
  title     = {基于机器学习的无人机目标图像类型识别方法研究},
  author    = {徐茂翔},
  school    = {},
  year      = {2026}
}
```

## 致谢

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [CBAM: Convolutional Block Attention Module](https://github.com/Jongchan/attention-module)
- [Qwen-VL 通义千问视觉模型](https://github.com/QwenLM/Qwen-VL)

## License

本项目基于 [GPL 3.0](LICENSE) 协议开源。

