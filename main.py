import sys
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QHBoxLayout, QPushButton)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


# ==================== ПОТОК ОБРАБОТКИ ВИДЕО ====================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_info_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.paused = False

        self.DIST_THRESHOLD = 30
        self.TAP_TIME = 0.3
        self.SMOOTHING_ALPHA = 0.6

        self.contact_start_time = None
        self.is_contacting = False
        self.smooth_x, self.smooth_y = None, None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")

        self.screen_w, self.screen_h = pyautogui.size()
        self.COLOR_FIRST_HAND = (0, 255, 0)
        self.COLOR_OTHER_HAND = (128, 128, 128)
        self.COLOR_CONTACT = (0, 0, 255)
        self.tracked_hand_label = None

    def run(self):
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            h, w, _ = frame.shape
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            all_hands = results.multi_hand_landmarks
            handedness = results.multi_handedness

            if all_hands:
                if self.tracked_hand_label is None:
                    if handedness and len(handedness) > 0:
                        self.tracked_hand_label = handedness[0].classification[0].label
                    else:
                        self.tracked_hand_label = "first"

                found = False
                for idx, hand in enumerate(all_hands):
                    label = handedness[idx].classification[0].label if handedness and idx < len(handedness) else None
                    if label == self.tracked_hand_label:
                        found = True
                        break
                if not found:
                    if handedness and len(handedness) > 0:
                        self.tracked_hand_label = handedness[0].classification[0].label
                    else:
                        self.tracked_hand_label = "first"

            first_hand_landmarks = None
            if all_hands:
                for idx, hand_landmarks in enumerate(all_hands):
                    label = handedness[idx].classification[0].label if handedness and idx < len(handedness) else None
                    is_tracked = (label == self.tracked_hand_label)
                    color = self.COLOR_FIRST_HAND if is_tracked else self.COLOR_OTHER_HAND

                    self.mp_drawing.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=color, thickness=2, circle_radius=3
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=color, thickness=2
                        )
                    )
                    if is_tracked:
                        first_hand_landmarks = hand_landmarks

            coord_text = "ожидание..."
            status_text = "РАБОТАЕТ" if not self.paused else "ПАУЗА"

            if not self.paused and first_hand_landmarks is not None:
                thumb_tip = first_hand_landmarks.landmark[4]
                index_tip = first_hand_landmarks.landmark[8]
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                dist = np.sqrt((ix - tx)**2 + (iy - ty)**2)

                if dist < self.DIST_THRESHOLD:
                    cv2.circle(canvas, ((ix + tx)//2, (iy + ty)//2), 10, self.COLOR_CONTACT, -1)

                    if not self.is_contacting:
                        self.contact_start_time = time.time()
                        self.is_contacting = True
                        raw_x = int(index_tip.x * self.screen_w)
                        raw_y = int(index_tip.y * self.screen_h)
                        self.smooth_x, self.smooth_y = raw_x, raw_y

                    if time.time() - self.contact_start_time > self.TAP_TIME:
                        raw_x = int(index_tip.x * self.screen_w)
                        raw_y = int(index_tip.y * self.screen_h)
                        raw_x = max(0, min(self.screen_w - 1, raw_x))
                        raw_y = max(0, min(self.screen_h - 1, raw_y))

                        if self.smooth_x is None or self.smooth_y is None:
                            self.smooth_x, self.smooth_y = raw_x, raw_y
                        else:
                            self.smooth_x = int(self.smooth_x * self.SMOOTHING_ALPHA +
                                                raw_x * (1 - self.SMOOTHING_ALPHA))
                            self.smooth_y = int(self.smooth_y * self.SMOOTHING_ALPHA +
                                                raw_y * (1 - self.SMOOTHING_ALPHA))

                        pyautogui.moveTo(self.smooth_x, self.smooth_y)
                else:
                    if self.is_contacting:
                        duration = time.time() - self.contact_start_time
                        if duration < self.TAP_TIME:
                            pyautogui.click(button='left')
                            print("Левый клик")
                        self.is_contacting = False
                        self.contact_start_time = None
                        self.smooth_x, self.smooth_y = None, None

            if self.smooth_x is not None and self.smooth_y is not None:
                coord_text = f"{self.smooth_x}, {self.smooth_y}"

            self.update_info_signal.emit(coord_text, status_text)

            alpha = np.where(np.any(canvas != 0, axis=2), 255, 0).astype(np.uint8)
            rgba = np.dstack((canvas, alpha))
            self.change_pixmap_signal.emit(rgba)

        self.cap.release()
        self.hands.close()

    def stop(self):
        self._run_flag = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused


# ==================== ГЛАВНОЕ ОКНО ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self.window_width = 400
        self.window_height = 300
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(10,
                         screen.height() - self.window_height - 10,
                         self.window_width,
                         self.window_height)

        central = QWidget()
        central.setStyleSheet("background: transparent;")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central.setLayout(main_layout)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background: transparent;")
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        panel = QWidget()
        panel.setStyleSheet("""
            background-color: rgba(30, 30, 30, 200);
            border-radius: 8px;
            margin: 5px;
        """)
        panel.setFixedHeight(50)
        panel_layout = QHBoxLayout()
        panel_layout.setContentsMargins(12, 5, 12, 5)
        panel.setLayout(panel_layout)

        # Кнопка – шрифт 9pt (уменьшен), ширина 100px
        self.btn_pause = QPushButton("РАБОТАЕТ")
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #2d8f2d;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:pressed {
                background-color: #1f6a1f;
            }
        """)
        self.btn_pause.setFixedWidth(100)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        panel_layout.addWidget(self.btn_pause)

        # Координаты – шрифт 12pt (без изменений)
        self.coord_label = QLabel("0, 0")
        self.coord_label.setStyleSheet("color: white; font-size: 12pt; font-weight: bold;")
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        panel_layout.addWidget(self.coord_label, 1)

        main_layout.addWidget(panel)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_info_signal.connect(self.update_info)
        self.video_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def update_image(self, rgba_img):
        h, w, ch = rgba_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgba_img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        scaled = qt_img.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def update_info(self, coord_text, status_text):
        self.coord_label.setText(coord_text)
        self.btn_pause.setText(status_text)
        if status_text == "ПАУЗА":
            self.btn_pause.setStyleSheet("""
                QPushButton {
                    background-color: #8f2d2d;
                    color: white;
                    font-size: 9pt;
                    font-weight: bold;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 10px;
                }
                QPushButton:pressed {
                    background-color: #6a1f1f;
                }
            """)
        else:
            self.btn_pause.setStyleSheet("""
                QPushButton {
                    background-color: #2d8f2d;
                    color: white;
                    font-size: 9pt;
                    font-weight: bold;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 10px;
                }
                QPushButton:pressed {
                    background-color: #1f6a1f;
                }
            """)

    def on_pause_clicked(self):
        paused = self.video_thread.toggle_pause()
        new_status = "ПАУЗА" if paused else "РАБОТАЕТ"
        self.btn_pause.setText(new_status)
        if paused:
            self.btn_pause.setStyleSheet("""
                QPushButton {
                    background-color: #8f2d2d;
                    color: white;
                    font-size: 9pt;
                    font-weight: bold;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 10px;
                }
                QPushButton:pressed {
                    background-color: #6a1f1f;
                }
            """)
        else:
            self.btn_pause.setStyleSheet("""
                QPushButton {
                    background-color: #2d8f2d;
                    color: white;
                    font-size: 9pt;
                    font-weight: bold;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 10px;
                }
                QPushButton:pressed {
                    background-color: #1f6a1f;
                }
            """)
        print("Трекинг приостановлен" if paused else "Трекинг возобновлён")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_X:
            self.on_pause_clicked()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())