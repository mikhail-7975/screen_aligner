#!/usr/bin/env python3
"""
Перспективная коррекция (keystone) для вывода на проектор под macOS.

Как пользоваться:
  1. Подключите проектор как второй дисплей (зеркало или расширение — как удобнее).
  2. Запустите: python keystone.py
  3. В режиме калибровки (по умолчанию при первом запуске без config.json)
     перетащите цветные маркеры углов так, чтобы на стене картинка стала прямоугольной.
  4. Нажмите S — сохранить настройки в config.json, Q или Escape — выход.

На macOS в Системных настройках → Конфиденциальность → Запись экрана
разрешите доступ для Terminal / вашего терминала / Python.

Углы задаются в долях от размера окна вывода (0…1): левый верх, правый верх,
правый низ, левый низ — порядок как у прямоугольника после коррекции на стене.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mss
import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPen, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def default_corners() -> list[list[float]]:
    return [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


class KeystoneWidget(QWidget):
    HANDLE_RADIUS = 14
    CORNER_COLORS = (
        QColor(255, 80, 80),
        QColor(80, 255, 80),
        QColor(80, 140, 255),
        QColor(255, 220, 80),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._corners_norm = default_corners()  # tl, tr, br, bl
        self.calibration_mode = True
        self.drag_idx: int | None = None
        self.last_frame: np.ndarray | None = None  # BGR

    def set_corners_normalized(self, corners: list[list[float]]) -> None:
        if len(corners) != 4:
            return
        self._corners_norm = [[clamp01(c[0]), clamp01(c[1])] for c in corners]

    def corners_pixel(self) -> np.ndarray:
        w, h = self.width(), self.height()
        pts = []
        for nx, ny in self._corners_norm:
            pts.append([nx * w, ny * h])
        return np.float32(pts)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.last_frame is not None:
            rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            painter.drawImage(self.rect(), qimg)

        if self.calibration_mode:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            pix = self.corners_pixel()
            for i in range(4):
                j = (i + 1) % 4
                painter.drawLine(
                    int(pix[i][0]),
                    int(pix[i][1]),
                    int(pix[j][0]),
                    int(pix[j][1]),
                )
            for i in range(4):
                c = self.CORNER_COLORS[i]
                painter.setBrush(c)
                painter.setPen(QPen(QColor(40, 40, 40), 2))
                cx, cy = float(pix[i][0]), float(pix[i][1])
                r = self.HANDLE_RADIUS
                painter.drawEllipse(QPointF(cx, cy), r, r)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.calibration_mode or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        px, py = pos.x(), pos.y()
        best, best_d = None, float("inf")
        for i, (nx, ny) in enumerate(self._corners_norm):
            cx, cy = nx * self.width(), ny * self.height()
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best_d = d
                best = i
        if best is not None and best_d <= (self.HANDLE_RADIUS + 8) ** 2:
            self.drag_idx = best

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drag_idx is None:
            return
        pos = event.position()
        nx = clamp01(pos.x() / max(self.width(), 1))
        ny = clamp01(pos.y() / max(self.height(), 1))
        self._corners_norm[self.drag_idx] = [nx, ny]
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.drag_idx = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.capture_monitor = int(self.cfg.get("capture_monitor", 0))
        self.output_monitor = int(self.cfg.get("output_monitor", 1))
        self.fps_cap = max(1, int(self.cfg.get("capture_fps_cap", 45)))

        corners = self.cfg.get("corners")
        cal_default = bool(self.cfg.get("show_calibration_handles", not CONFIG_PATH.exists()))

        self.widget = KeystoneWidget()
        if corners and isinstance(corners, list) and len(corners) == 4:
            self.widget.set_corners_normalized(corners)
        self.widget.calibration_mode = cal_default

        self.setCentralWidget(self.widget)
        self.setWindowTitle("Keystone / трапеция для проектора")

        self.sct = mss.mss()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def move_to_output_screen(self, screens: list):
        idx = self.output_monitor
        if idx < 0 or idx >= len(screens):
            idx = min(len(screens) - 1, max(0, 1 if len(screens) > 1 else 0))
            self.output_monitor = idx
        geo = screens[idx].geometry()
        self.setGeometry(geo)
        self.showFullScreen()

    def _capture_region(self) -> tuple[int, int, int, int]:
        mons = self.sct.monitors[1:]  # skip "all in one"
        if not mons:
            mons = self.sct.monitors[1:2]
        ci = self.capture_monitor
        if ci < 0 or ci >= len(mons):
            ci = 0
            self.capture_monitor = ci
        m = mons[ci]
        return int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"])

    def _tick(self):
        left, top, w, h = self._capture_region()
        try:
            shot = self.sct.grab({"left": left, "top": top, "width": w, "height": h})
        except mss.exception.ScreenShotError:
            return

        frame = np.asarray(shot)[:, :, :3]  # BGRA -> take BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        out_w, out_h = max(self.widget.width(), 1), max(self.widget.height(), 1)
        sw, sh = frame.shape[1], frame.shape[0]

        pts_src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
        pts_dst = self.widget.corners_pixel().astype(np.float32)

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_full = cv2.warpPerspective(
            frame,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        self.widget.last_frame = warped_full
        self.widget.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
            return
        if key == Qt.Key.Key_S:
            cfg = {
                "capture_monitor": self.capture_monitor,
                "output_monitor": self.output_monitor,
                "corners": self.widget._corners_norm,
                "capture_fps_cap": self.fps_cap,
                "show_calibration_handles": self.widget.calibration_mode,
            }
            save_config(cfg)
            self.setWindowTitle("Сохранено → config.json")
            return
        if key == Qt.Key.Key_C:
            self.widget.calibration_mode = not self.widget.calibration_mode
            self.widget.update()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        interval_ms = max(1, int(1000 / self.fps_cap))
        self.timer.start(interval_ms)

    def closeEvent(self, event):
        self.timer.stop()
        self.sct.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    screens = app.screens()
    if not screens:
        print("Нет доступных экранов.", file=sys.stderr)
        sys.exit(1)

    win = MainWindow()
    win.move_to_output_screen(screens)

    if len(screens) < 2:
        print(
            "Подключён только один дисплей: окно откроется на нём. "
            "Для проектора подключите второй монитор и при необходимости "
            "измените output_monitor в config.json (индекс экрана 0, 1, …).",
            file=sys.stderr,
        )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
