import os
import re
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Optional, Tuple

import cv2
import torch
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QGroupBox,
    QSizePolicy,
)

try:
    import dashscope
except Exception:
    dashscope = None


YOLO_MODEL_PATH = r"D:\project\yolov5\runs\train\exp2\weights\best.pt"
YOLO_REPO_PATH = r"D:\project\yolov5"
DEFAULT_QWEN_MODEL = "qwen-vl-max"


@dataclass
class Detection:
    label: str
    conf: float
    box: Tuple[int, int, int, int]


def draw_detections(img_bgr, detections: List[Detection], title: str = ""):
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 210, 0), 2)
        text = f"{det.label} {det.conf:.2f}"
        cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2)
    if title:
        cv2.putText(out, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 180, 0), 2)
    return out


def parse_qwen_boxes(text: str, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
    h_img, w_img, _ = image_shape
    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.findall(pattern, text)
    boxes: List[Tuple[int, int, int, int]] = []
    for m in matches:
        x1_n, y1_n, x2_n, y2_n = map(int, m)
        x1 = int((x1_n / 1000.0) * w_img)
        y1 = int((y1_n / 1000.0) * h_img)
        x2 = int((x2_n / 1000.0) * w_img)
        y2 = int((y2_n / 1000.0) * h_img)
        if x2 > x1 and y2 > y1:
            boxes.append((max(0, x1), max(0, y1), min(w_img - 1, x2), min(h_img - 1, y2)))
    return boxes


class Detector:
    def __init__(self, yolo_repo: str, yolo_weight: str):
        self.yolo_repo = yolo_repo
        self.yolo_weight = yolo_weight
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            self.model = torch.hub.load(
                self.yolo_repo,
                "custom",
                path=self.yolo_weight,
                source="local",
                force_reload=False,
            )
            self.model.to(self.device)
            self.model.eval()

    def yolo_detect(self, img_bgr, conf_thres: float) -> List[Detection]:
        self.load()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=640)
        df = results.pandas().xyxy[0]
        detections: List[Detection] = []
        if df.empty:
            return detections
        for _, row in df.iterrows():
            conf = float(row["confidence"])
            if conf < conf_thres:
                continue
            detections.append(
                Detection(
                    label=str(row["name"]),
                    conf=conf,
                    box=(int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])),
                )
            )
        return detections

    def qwen_coarse(self, image_path: str, image_shape: Tuple[int, int, int], api_key: str) -> List[Tuple[int, int, int, int]]:
        if dashscope is None:
            return []
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["all_proxy"] = ""
        dashscope.api_key = api_key
        prompt = (
            "请检测图中是否存在无人机(drone)和鸟(bird)，输出所有目标坐标，"
            "格式为 [xmin, ymin, xmax, ymax]，坐标为 0-1000 归一化整数。"
        )
        messages = [
            {"role": "user", "content": [{"image": f"file://{image_path}"}, {"text": prompt}]},
        ]
        response = dashscope.MultiModalConversation.call(model=DEFAULT_QWEN_MODEL, messages=messages)
        if response.status_code != HTTPStatus.OK:
            return []
        text = response.output.choices[0].message.content[0]["text"]
        return parse_qwen_boxes(text, image_shape)

    def qwen_yolo_detect(self, image_path: str, img_bgr, conf_thres: float, api_key: str) -> List[Detection]:
        coarse_boxes = self.qwen_coarse(image_path, img_bgr.shape, api_key)
        if not coarse_boxes:
            return []
        h, w, _ = img_bgr.shape
        padding = 30
        final: List[Detection] = []
        for x1, y1, x2, y2 in coarse_boxes:
            cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
            cx2, cy2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            local_dets = self.yolo_detect(crop, conf_thres)
            for det in local_dets:
                lx1, ly1, lx2, ly2 = det.box
                final.append(
                    Detection(
                        label=det.label,
                        conf=det.conf,
                        box=(cx1 + lx1, cy1 + ly1, cx1 + lx2, cy1 + ly2),
                    )
                )
        return final


class DetectWorker(QThread):
    done = pyqtSignal(object, str, float, object)
    failed = pyqtSignal(str)

    def __init__(self, detector: Detector, image_path: str, mode: str, conf: float, api_key: str):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.mode = mode
        self.conf = conf
        self.api_key = api_key

    def run(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                raise RuntimeError("图片读取失败，请检查路径或文件格式。")
            t0 = time.time()
            if self.mode == "YOLO+CBAM检测":
                detections = self.detector.yolo_detect(img, self.conf)
                out = draw_detections(img, detections, f"YOLO: {len(detections)}")
            else:
                if not self.api_key.strip():
                    raise RuntimeError("千问模式需要 DashScope API Key。")
                detections = self.detector.qwen_yolo_detect(self.image_path, img, self.conf, self.api_key.strip())
                out = draw_detections(img, detections, f"Qwen+YOLO: {len(detections)}")
            elapsed = time.time() - t0
            self.done.emit(out, f"检测完成，目标数: {len(detections)}", elapsed, detections)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("imageCard")
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        header_row = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.info_label = QLabel("")
        self.info_label.setObjectName("cardInfo")
        header_row.addWidget(self.title)
        header_row.addStretch()
        header_row.addWidget(self.info_label)

        self.viewer = QLabel("等待加载...")
        self.viewer.setAlignment(Qt.AlignCenter)
        self.viewer.setMinimumSize(420, 360)
        self.viewer.setObjectName("imagePanel")
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        vbox.addLayout(header_row)
        vbox.addWidget(self.viewer, 1)

    def set_image(self, img_rgb):
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.viewer.setPixmap(pix)

    def set_info(self, text: str):
        self.info_label.setText(text)


class StatsPanel(QFrame):
    """右侧统计面板"""

    def __init__(self):
        super().__init__()
        self.setObjectName("statsPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # --- 检测统计 ---
        stats_group = QGroupBox("检测统计")
        stats_group.setObjectName("statsGroup")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(8)

        self.stat_total = QLabel("目标总数: --")
        self.stat_total.setObjectName("statValue")
        self.stat_drone = QLabel("无人机 (drone): --")
        self.stat_drone.setObjectName("statDrone")
        self.stat_bird = QLabel("飞鸟 (bird): --")
        self.stat_bird.setObjectName("statBird")
        self.stat_avg_conf = QLabel("平均置信度: --")
        self.stat_avg_conf.setObjectName("statValue")
        self.stat_max_conf = QLabel("最高置信度: --")
        self.stat_max_conf.setObjectName("statValue")
        self.stat_elapsed = QLabel("检测耗时: --")
        self.stat_elapsed.setObjectName("statValue")

        stats_layout.addWidget(self.stat_total)
        stats_layout.addWidget(self.stat_drone)
        stats_layout.addWidget(self.stat_bird)
        stats_layout.addWidget(self._make_separator())
        stats_layout.addWidget(self.stat_avg_conf)
        stats_layout.addWidget(self.stat_max_conf)
        stats_layout.addWidget(self._make_separator())
        stats_layout.addWidget(self.stat_elapsed)
        layout.addWidget(stats_group)

        # --- 图片信息 ---
        img_group = QGroupBox("图片信息")
        img_group.setObjectName("statsGroup")
        img_layout = QVBoxLayout(img_group)
        img_layout.setSpacing(6)
        self.img_filename = QLabel("文件名: --")
        self.img_filename.setObjectName("statValue")
        self.img_resolution = QLabel("分辨率: --")
        self.img_resolution.setObjectName("statValue")
        self.img_filesize = QLabel("文件大小: --")
        self.img_filesize.setObjectName("statValue")
        self.img_mode_label = QLabel("检测模式: --")
        self.img_mode_label.setObjectName("statValue")
        img_layout.addWidget(self.img_filename)
        img_layout.addWidget(self.img_resolution)
        img_layout.addWidget(self.img_filesize)
        img_layout.addWidget(self.img_mode_label)
        layout.addWidget(img_group)

        # --- 检测日志 ---
        log_group = QGroupBox("检测日志")
        log_group.setObjectName("statsGroup")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setMaximumHeight(200)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group)

        layout.addStretch()

    def _make_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("statSep")
        return sep

    def update_stats(self, detections: List[Detection], elapsed: float, mode: str):
        n = len(detections)
        self.stat_total.setText(f"目标总数: {n}")
        drones = [d for d in detections if d.label == "drone"]
        birds = [d for d in detections if d.label == "bird"]
        self.stat_drone.setText(f"无人机 (drone): {len(drones)}")
        self.stat_bird.setText(f"飞鸟 (bird): {len(birds)}")
        if n > 0:
            avg_c = sum(d.conf for d in detections) / n
            max_c = max(d.conf for d in detections)
            self.stat_avg_conf.setText(f"平均置信度: {avg_c:.3f}")
            self.stat_max_conf.setText(f"最高置信度: {max_c:.3f}")
        else:
            self.stat_avg_conf.setText("平均置信度: --")
            self.stat_max_conf.setText("最高置信度: --")
        self.stat_elapsed.setText(f"检测耗时: {elapsed:.2f}s")
        self.img_mode_label.setText(f"检测模式: {mode}")

    def update_image_info(self, filepath: str, img):
        h, w = img.shape[:2]
        self.img_filename.setText(f"文件名: {os.path.basename(filepath)}")
        self.img_resolution.setText(f"分辨率: {w} x {h}")
        try:
            size_kb = os.path.getsize(filepath) / 1024
            if size_kb > 1024:
                self.img_filesize.setText(f"文件大小: {size_kb / 1024:.2f} MB")
            else:
                self.img_filesize.setText(f"文件大小: {size_kb:.1f} KB")
        except Exception:
            self.img_filesize.setText("文件大小: --")

    def append_log(self, text: str):
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{ts}] {text}")

    def reset(self):
        self.stat_total.setText("目标总数: --")
        self.stat_drone.setText("无人机 (drone): --")
        self.stat_bird.setText("飞鸟 (bird): --")
        self.stat_avg_conf.setText("平均置信度: --")
        self.stat_max_conf.setText("最高置信度: --")
        self.stat_elapsed.setText("检测耗时: --")
        self.img_mode_label.setText("检测模式: --")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人机目标检测系统")
        self.resize(1500, 880)
        self.setMinimumSize(1280, 760)
        self.img_path: Optional[str] = None
        self.last_result: Optional[np.ndarray] = None
        self.last_detections: List[Detection] = []
        self.worker: Optional[DetectWorker] = None
        self.detector = Detector(YOLO_REPO_PATH, YOLO_MODEL_PATH)
        self._build_ui()
        self._apply_style()
        self._safe_preload_model()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(16, 14, 16, 14)
        main.setSpacing(10)

        # ---- 标题栏 ----
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(4, 0, 4, 0)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        self.header_title = QLabel("无人机目标检测系统")
        self.header_title.setObjectName("headerTitle")
        self.header_title.setFont(QFont("Microsoft YaHei UI", 20, QFont.Bold))
        self.header_subtitle = QLabel("基于 YOLOv5-CBAM 与千问-VL 的级联检测框架")
        self.header_subtitle.setObjectName("headerSubtitle")
        title_block.addWidget(self.header_title)
        title_block.addWidget(self.header_subtitle)

        header_layout.addLayout(title_block)
        header_layout.addStretch()

        self.model_badge = QLabel("")
        self.model_badge.setObjectName("modelBadge")
        header_layout.addWidget(self.model_badge)

        main.addWidget(header_widget)

        # ---- 主体三栏布局 ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # 左栏 —— 控制面板
        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        # 中栏 —— 图像显示区
        center_panel = self._build_center_panel()
        splitter.addWidget(center_panel)

        # 右栏 —— 统计与日志
        self.stats_panel = StatsPanel()
        splitter.addWidget(self.stats_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 820, 280])

        main.addWidget(splitter, 1)

        # ---- 底部状态栏 ----
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(4, 2, 4, 2)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusText")

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("detectProgress")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(18)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addStretch()

        self.device_label = QLabel(f"设备: {self.detector.device.upper()}")
        self.device_label.setObjectName("statusInfo")
        status_layout.addWidget(self.device_label)

        main.addWidget(status_widget)

        # 信号连接
        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect.clicked.connect(self.detect)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_clear.clicked.connect(self.clear_all)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)

    def _build_left_panel(self):
        panel = QFrame()
        panel.setObjectName("controlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(10)

        # 操作按钮区
        btn_group = QGroupBox("操作")
        btn_group.setObjectName("ctrlGroup")
        btn_layout = QVBoxLayout(btn_group)
        btn_layout.setSpacing(8)

        self.btn_open = QPushButton("📂  打开图片")
        self.btn_open.setObjectName("ctrlBtn")
        self.btn_detect = QPushButton("🔍  开始检测")
        self.btn_detect.setObjectName("primaryBtn")
        self.btn_save = QPushButton("💾  保存结果")
        self.btn_save.setObjectName("ctrlBtn")
        self.btn_clear = QPushButton("🗑  清空界面")
        self.btn_clear.setObjectName("ctrlBtnSecondary")

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_clear)
        layout.addWidget(btn_group)

        # 检测设置区
        setting_group = QGroupBox("检测设置")
        setting_group.setObjectName("ctrlGroup")
        setting_layout = QVBoxLayout(setting_group)
        setting_layout.setSpacing(8)

        mode_label = QLabel("检测模式")
        mode_label.setObjectName("settingLabel")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["YOLO+CBAM检测", "千问+YOLO检测"])

        self.label_conf = QLabel("置信度阈值: 0.25")
        self.label_conf.setObjectName("settingLabel")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(25)

        setting_layout.addWidget(mode_label)
        setting_layout.addWidget(self.combo_mode)
        setting_layout.addWidget(self.label_conf)
        setting_layout.addWidget(self.conf_slider)
        layout.addWidget(setting_group)

        # API 设置区
        api_group = QGroupBox("千问 API")
        api_group.setObjectName("ctrlGroup")
        api_layout = QVBoxLayout(api_group)
        api_layout.setSpacing(6)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("DashScope API Key")
        self.api_key_edit.setEchoMode(QLineEdit.Password)

        api_hint = QLabel("仅「千问+YOLO检测」模式调用")
        api_hint.setObjectName("apiHintText")
        api_hint.setWordWrap(True)

        env_key = os.getenv("DASHSCOPE_API_KEY", "")
        if env_key:
            self.api_key_edit.setText(env_key)

        api_layout.addWidget(self.api_key_edit)
        api_layout.addWidget(api_hint)
        layout.addWidget(api_group)

        layout.addStretch()
        return panel

    def _build_center_panel(self):
        panel = QFrame()
        panel.setObjectName("centerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        image_row = QHBoxLayout()
        image_row.setSpacing(10)
        self.src_card = ImageCard("原始图像")
        self.res_card = ImageCard("检测结果")
        image_row.addWidget(self.src_card, 1)
        image_row.addWidget(self.res_card, 1)
        layout.addLayout(image_row, 1)
        return panel

    def _apply_style(self):
        self.setStyleSheet(
            """
            /* === 全局基调 === */
            QMainWindow, QWidget {
                background: #0b1121;
                color: #cbd5e1;
                font-family: "Microsoft YaHei UI";
                font-size: 13px;
            }

            /* === 标题 === */
            #headerTitle {
                color: #f1f5f9;
            }
            #headerSubtitle {
                color: #64748b;
                font-size: 12px;
            }
            #modelBadge {
                background: #1e3a5f;
                color: #7dd3fc;
                border-radius: 10px;
                padding: 4px 14px;
                font-weight: 600;
                font-size: 12px;
            }

            /* === 控制面板 === */
            #controlPanel {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
            #ctrlGroup {
                background: transparent;
                border: none;
                color: #94a3b8;
                font-weight: 600;
                font-size: 12px;
                padding-top: 4px;
            }
            #ctrlGroup::title {
                color: #64748b;
                padding-bottom: 4px;
            }
            #settingLabel {
                color: #94a3b8;
                font-size: 12px;
                margin-top: 2px;
            }
            #apiHintText {
                color: #64748b;
                font-size: 11px;
                padding: 2px 0;
            }

            /* === 按钮（通用） === */
            #ctrlBtn {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px 12px;
                font-weight: 600;
                font-size: 13px;
            }
            #ctrlBtn:hover {
                background: #273449;
                border-color: #475569;
            }
            #ctrlBtnSecondary {
                background: transparent;
                color: #94a3b8;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 9px 12px;
                font-size: 13px;
            }
            #ctrlBtnSecondary:hover {
                background: #1a2332;
                color: #cbd5e1;
            }
            #primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #0284c7, stop:1 #06b6d4);
                color: #f8fafc;
                border: none;
                border-radius: 8px;
                padding: 11px 12px;
                font-weight: 700;
                font-size: 13px;
            }
            #primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #0ea5e9, stop:1 #22d3ee);
            }
            #primaryBtn:disabled, #ctrlBtn:disabled, #ctrlBtnSecondary:disabled {
                background: #1a2332;
                color: #475569;
            }

            /* === 下拉框 & 输入框 === */
            QComboBox {
                background: #020617;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 7px 10px;
                min-height: 30px;
            }
            QComboBox:hover {
                border-color: #475569;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 6px;
            }
            QComboBox QAbstractItemView {
                background: #0f172a;
                border: 1px solid #334155;
                selection-background-color: #1e3a5f;
            }
            QLineEdit {
                background: #020617;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 7px 10px;
                min-height: 30px;
            }
            QLineEdit:hover {
                border-color: #475569;
            }

            /* === 滑块 === */
            QSlider::groove:horizontal {
                background: #1e293b;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #22d3ee;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #67e8f9;
            }

            /* === 图像卡片 === */
            #imageCard {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 8px;
            }
            #cardTitle {
                color: #bae6fd;
                font-weight: 700;
                font-size: 14px;
            }
            #cardInfo {
                color: #64748b;
                font-size: 11px;
            }
            #imagePanel {
                background: #020617;
                border: 1px dashed #334155;
                border-radius: 10px;
                color: #475569;
            }

            /* === 右侧统计面板 === */
            #statsPanel {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
            #statsGroup {
                background: transparent;
                border: none;
                color: #94a3b8;
                font-weight: 600;
                font-size: 12px;
                padding-top: 2px;
            }
            #statsGroup::title {
                color: #64748b;
                padding-bottom: 4px;
            }
            #statValue {
                color: #cbd5e1;
                font-size: 12px;
                padding: 1px 0;
            }
            #statDrone {
                color: #f97316;
                font-size: 12px;
                padding: 1px 0;
            }
            #statBird {
                color: #22d3ee;
                font-size: 12px;
                padding: 1px 0;
            }
            #statSep {
                background: #1e293b;
                max-height: 1px;
                margin: 2px 0;
            }
            #logView {
                background: #020617;
                border: 1px solid #1e293b;
                border-radius: 8px;
                color: #94a3b8;
                font-size: 11px;
                padding: 6px;
            }

            /* === 进度条 === */
            #detectProgress {
                background: #1e293b;
                border: none;
                border-radius: 4px;
                max-height: 6px;
            }
            #detectProgress::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #06b6d4, stop:1 #22d3ee);
                border-radius: 4px;
            }

            /* === 状态栏 === */
            #statusText {
                color: #a5f3fc;
                font-size: 12px;
            }
            #statusInfo {
                color: #64748b;
                font-size: 11px;
                margin-left: 12px;
            }

            /* === 滚动条 === */
            QScrollBar:vertical {
                background: #0f172a;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            /* === Splitter === */
            QSplitter::handle {
                background: #0b1121;
            }
            """
        )

    def _safe_preload_model(self):
        try:
            self.detector.load()
            self.status_label.setText("模型已加载，等待操作")
            self.model_badge.setText(f"模型已就绪  |  {self.detector.device.upper()}")
        except Exception as exc:
            self.status_label.setText(f"模型加载失败: {exc}")
            self.model_badge.setText("模型加载失败")

    # ---------- slots ----------
    def _on_conf_changed(self):
        conf = self.conf_slider.value() / 100.0
        self.label_conf.setText(f"置信度阈值: {conf:.2f}")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.critical(self, "错误", "图片读取失败。")
            return
        self.img_path = file_path
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.src_card.set_image(img_rgb)
        h, w = img.shape[:2]
        self.src_card.set_info(f"{w} x {h}")
        self.res_card.viewer.clear()
        self.res_card.viewer.setText("点击「开始检测」生成结果")
        self.res_card.set_info("")
        self.stats_panel.reset()
        self.stats_panel.update_image_info(file_path, img)
        self.status_label.setText(f"已加载: {os.path.basename(file_path)}")
        self.stats_panel.append_log(f"加载图片: {os.path.basename(file_path)}")

    def _set_busy(self, busy: bool):
        self.btn_open.setDisabled(busy)
        self.btn_detect.setDisabled(busy)
        self.btn_save.setDisabled(busy)
        self.btn_clear.setDisabled(busy)
        self.combo_mode.setDisabled(busy)
        self.conf_slider.setDisabled(busy)
        self.api_key_edit.setDisabled(busy)
        self.progress_bar.setVisible(busy)

    def detect(self):
        if not self.img_path:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        if self.worker and self.worker.isRunning():
            return
        mode = self.combo_mode.currentText()
        conf = self.conf_slider.value() / 100.0
        api_key = self.api_key_edit.text()
        self._set_busy(True)
        self.status_label.setText("检测中，请稍候...")
        self.stats_panel.append_log(f"开始检测 [{mode}]")
        self.worker = DetectWorker(self.detector, self.img_path, mode, conf, api_key)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def save_result(self):
        if self.last_result is None:
            QMessageBox.information(self, "提示", "暂无检测结果可保存。")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "detection_result.jpg", "JPEG (*.jpg);;PNG (*.png)"
        )
        if not file_path:
            return
        cv2.imwrite(file_path, self.last_result)
        self.status_label.setText(f"结果已保存: {os.path.basename(file_path)}")
        self.stats_panel.append_log(f"保存结果: {os.path.basename(file_path)}")

    def clear_all(self):
        self.img_path = None
        self.last_result = None
        self.last_detections = []
        self.src_card.viewer.clear()
        self.src_card.viewer.setText("等待加载...")
        self.src_card.set_info("")
        self.res_card.viewer.clear()
        self.res_card.viewer.setText("等待加载...")
        self.res_card.set_info("")
        self.stats_panel.reset()
        self.status_label.setText("就绪")
        self.stats_panel.append_log("界面已清空")

    def _on_done(self, out_bgr, msg: str, elapsed: float, detections):
        self.last_result = out_bgr
        self.last_detections = detections
        img_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        self.res_card.set_image(img_rgb)
        h, w = out_bgr.shape[:2]
        self.res_card.set_info(f"{w} x {h}")
        self.status_label.setText(msg)
        mode = self.combo_mode.currentText()
        self.stats_panel.update_stats(detections, elapsed, mode)
        self.stats_panel.append_log(f"检测完成 | {msg} | 耗时 {elapsed:.2f}s")

    def _on_failed(self, err: str):
        QMessageBox.critical(self, "检测失败", err)
        self.status_label.setText(f"失败: {err}")
        self.stats_panel.append_log(f"检测失败: {err}")




def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
