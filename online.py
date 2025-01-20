# -*- coding: utf-8 -*-

import os
import cv2
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from threading import Thread
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QStatusBar, QMessageBox

from utils.parser import ParserUse
from utils.guis import PhaseCom
from utils.report_tools import generate_report, get_meta

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    process_img_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, video_path=None):
        super().__init__()
        self._run_flag = True
        if not video_path:
            logging.error("No video path provided")
            return
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return
        self.video_path = video_path

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_idx = 0
            
            while self._run_flag and cap.isOpened():
                ret, cv_img = cap.read()
                if ret:
                    frame_idx += 1
                    self.change_pixmap_signal.emit(cv_img)
                    self.process_img_signal.emit(cv_img, frame_idx)
                    # Control playback speed
                    time.sleep(1/30)  # Adjust for smoother playback
                else:
                    break
                    
            cap.release()
        except Exception as e:
            logging.error(f"Error in video thread: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()

class Ui_iPhaser(QMainWindow):
    def __init__(self):
        super(Ui_iPhaser, self).__init__()

    def setupUi(self, cfg):
        self.setObjectName("iPhaser")
        self.resize(1300, 930)
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)

        # Initialize base attributes
        self.disply_width = 1200
        self.display_height = 900
        self.save_folder = "../Records"
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        
        # Initialize video-related attributes before creating thread
        self.video_path = cfg.video_path if hasattr(cfg, 'video_path') else None
        self.FRAME_WIDTH, self.FRAME_HEIGHT, self.stream_fps = self.get_frame_size()
        self.CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        
        # Initialize other parameters
        self.down_ratio = cfg.down_ratio
        self.start_time = "--:--:--"
        self.trainee_name = "--"
        self.manual_set = "--"
        self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
        self.fps = 0
        self.pred = "--"
        self.log_data = []
        
        # Initialize phase segmentation
        self.phaseseg = PhaseCom(arg=cfg)
        
        # Initialize status
        self.init_status()
        self.MANUAL_FRAMES = self.stream_fps * cfg.manual_set_fps_ratio
        self.manual_frame = 0

        # Main layout
        main_layout = QVBoxLayout(self.centralwidget)

        # --- Video Selection Layout ---
        video_selection_layout = QHBoxLayout()

        # Video File Label
        self.VideoFileLabel = QLabel("Video File:", self.centralwidget)
        video_selection_layout.addWidget(self.VideoFileLabel)

        # Video Path Display
        self.VideoPathDisplay = QLineEdit(self.centralwidget)
        self.VideoPathDisplay.setReadOnly(True)
        video_selection_layout.addWidget(self.VideoPathDisplay)

        # Browse Button
        self.ChooseVideo = QPushButton("Browse", self.centralwidget)
        self.ChooseVideo.clicked.connect(self.click_choose_video)
        video_selection_layout.addWidget(self.ChooseVideo)

        # Add Video Selection Layout to Main Layout
        main_layout.addLayout(video_selection_layout)

        # --- Time Label Layout ---
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(50, 34, 250, 40))
        self.layoutWidget2.setObjectName("layoutWidget2")
        
        time_layout = QtWidgets.QHBoxLayout(self.layoutWidget2)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setObjectName("horizontalLayout_3")
        
        # Create TimeLabel
        self.TimeLabel = QtWidgets.QLabel("Time start:", self.layoutWidget2)
        self.TimeLabel.setObjectName("TimeLabel")
        time_layout.addWidget(self.TimeLabel)
        
        # Create time display label
        self.label = QtWidgets.QLabel(self.layoutWidget2)
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setText("--:--:--")
        self.label.setObjectName("label")
        time_layout.addWidget(self.label)

        # Add to main layout after video selection layout
        main_layout.addWidget(self.layoutWidget2)

        # --- Video Display Label ---
        self.DisplayVideo = QLabel(self.centralwidget)
        self.DisplayVideo.setFixedSize(self.disply_width, self.display_height)
        self.DisplayVideo.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.DisplayVideo.setAlignment(QtCore.Qt.AlignCenter)
        self.DisplayVideo.setObjectName("DisplayVideo")
        main_layout.addWidget(self.DisplayVideo)

        # --- Information and Control Layout ---
        info_control_layout = QHBoxLayout()

        # Trainee Information
        self.TraineeLabel = QLabel("Trainee:", self.centralwidget)
        self.TraineeName = QLineEdit(self.centralwidget)
        info_control_layout.addWidget(self.TraineeLabel)
        info_control_layout.addWidget(self.TraineeName)

        # Trainer Information
        self.TrainerLabel = QLabel("Trainer:", self.centralwidget)
        self.TrainerName = QLineEdit(self.centralwidget)
        info_control_layout.addWidget(self.TrainerLabel)
        info_control_layout.addWidget(self.TrainerName)

        # Bed Information
        self.BedLabel = QLabel("Bed:", self.centralwidget)
        self.BedName = QLineEdit(self.centralwidget)
        info_control_layout.addWidget(self.BedLabel)
        info_control_layout.addWidget(self.BedName)

        # Case Information
        self.CaseLabel = QLabel("Case:", self.centralwidget)
        self.CaseName = QLineEdit(self.centralwidget)
        info_control_layout.addWidget(self.CaseLabel)
        info_control_layout.addWidget(self.CaseName)

        # Add Information and Control Layout to Main Layout
        main_layout.addLayout(info_control_layout)

        # --- Control Buttons Layout ---
        control_buttons_layout = QHBoxLayout()

        # Start Button
        self.Start = QPushButton("Start", self.centralwidget)
        self.Start.clicked.connect(self.click_start)
        control_buttons_layout.addWidget(self.Start)

        # Stop Button
        self.Stop = QPushButton("Stop", self.centralwidget)
        self.Stop.clicked.connect(self.click_stop)
        self.Stop.setEnabled(False)  # Initially disabled
        control_buttons_layout.addWidget(self.Stop)

        # Add Control Buttons Layout to Main Layout
        main_layout.addLayout(control_buttons_layout)

        # --- Action Buttons Layout ---
        action_buttons_layout = QHBoxLayout()
        
        self.ActionIndependent = QPushButton("Independent", self.centralwidget)
        self.ActionIndependent.setObjectName("ActionIndependent")
        self.ActionIndependent.clicked.connect(self.click_independent)
        action_buttons_layout.addWidget(self.ActionIndependent)
        
        self.ActionHelp = QPushButton("Help", self.centralwidget)
        self.ActionHelp.setObjectName("ActionHelp")
        self.ActionHelp.clicked.connect(self.click_help)
        action_buttons_layout.addWidget(self.ActionHelp)
        
        self.ActionTakeOver = QPushButton("Take Over", self.centralwidget)
        self.ActionTakeOver.setObjectName("ActionTakeOver")
        self.ActionTakeOver.clicked.connect(self.click_take_over)
        action_buttons_layout.addWidget(self.ActionTakeOver)
        
        self.Report = QPushButton("Report", self.centralwidget)
        self.Report.setObjectName("Report")
        self.Report.clicked.connect(self.click_report)
        action_buttons_layout.addWidget(self.Report)

        # Add action buttons layout to main layout
        main_layout.addLayout(action_buttons_layout)

        # Add Phase Display Label
        self.phase_display = QLabel(self.centralwidget)
        self.phase_display.setStyleSheet("""
            QLabel {
                background-color: black;
                color: lime;
                padding: 5px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.phase_display.setAlignment(Qt.AlignCenter)
        self.phase_display.setText("Phase: --")
        main_layout.addWidget(self.phase_display)

        # --- Status Bar ---
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        # Set main layout
        self.centralwidget.setLayout(main_layout)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        # Add video file selection
        self.video_path = cfg.video_path if hasattr(cfg, 'video_path') else None
        
        # Get video properties from file instead of camera
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            self.FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.stream_fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

        # Initialize video thread with video path
        self.thread = VideoThread(self.video_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.process_img_signal.connect(self.process_img)
        self.thread.start()

    def init_status(self):
        self.WORKING = False
        self.INIT = False
        self.TRAINEE = "NONE"
        self.PAUSE_times = 0
        self.INDEPENDENT = True
        self.HELP = False
        self.STATUS = "--"
        # Initialize these attributes
        self.case_name = "--"
        self.trainee_name = "--"
        self.trainer_name = "--"
        self.bed_name = "--"

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("iPhaser", "iPhaser"))
        self.TraineeLabel.setText(_translate("iPhaser", "Trainee:"))
        self.TrainerLabel.setText(_translate("iPhaser", "Mentor:"))
        self.BedLabel.setText(_translate("iPhaser", "Bed:"))
        self.CaseLabel.setText(_translate("iPhaser", "Case:"))
        self.Start.setText(_translate("iPhaser", "Start"))
        self.Stop.setText(_translate("iPhaser", "Stop"))
        self.TimeLabel.setText(_translate("iPhaser", "Time start:"))
        self.ActionIndependent.setText(_translate("iPhaser", "Independent"))
        self.ActionHelp.setText(_translate("iPhaser", "With help"))
        self.ActionTakeOver.setText(_translate("iPhaser", "Take over"))
        self.Report.setText(_translate("iPhaser", "> Generate report <"))

    # Buttons
    def click_start(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a video file first!")
            return

        try:
            # Get video properties first
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {self.video_path}")
                
            # Initialize processing status
            self.WORKING = True
            self.INIT = True
            self.log_data = []
            self.manual_frame = 0
            self.manual_set = "--"
            
            # Get input values
            self.trainee_name = self.TraineeName.text() or "unnamed"
            self.trainer_name = self.TrainerName.text() or "unnamed"
            self.bed_name = self.BedName.text() or "unnamed"
            self.case_name = self.CaseName.text() or "unnamed"
            
            # Set up logging
            self.start_time = datetime.now().strftime("%H:%M:%S")
            self.label.setText(self.start_time)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create log and video files
            base_name = f"{self.case_name}_{self.trainee_name}_{timestamp}"
            self.log_file = os.path.join(self.save_folder, f"{base_name}.csv")
            video_path = os.path.join(self.save_folder, f"{base_name}.avi")
            
            # Initialize video writer
            self.output_video = cv2.VideoWriter(
                video_path,
                self.CODEC,
                max(1, self.stream_fps),
                (self.FRAME_WIDTH, self.FRAME_HEIGHT)
            )

            # Start video processing
            self.thread = VideoThread(self.video_path)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.process_img_signal.connect(self.process_img)
            self.thread.start()

            # Update UI
            self.Start.setEnabled(False)
            self.Stop.setEnabled(True)
            self.ChooseVideo.setEnabled(False)
            self.ActionIndependent.setEnabled(True)
            self.ActionHelp.setEnabled(True)
            self.ActionTakeOver.setEnabled(True)
            
            # Disable inputs while processing
            self.TraineeName.setEnabled(False)
            self.TrainerName.setEnabled(False)
            self.BedName.setEnabled(False)
            self.CaseName.setEnabled(False)

        except Exception as e:
            logging.error(f"Error in click_start: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.click_stop()

    def click_stop(self):
        try:
            self.WORKING = False
            if hasattr(self, 'thread'):
                self.thread.stop()
            if hasattr(self, 'output_video'):
                self.output_video.release()
            if hasattr(self, 'log_data') and self.log_data:
                self.save_log_data()
            
            # Reset UI
            self.label.setText("--:--:--")
            self.phase_display.setText("Phase: --")
            self.phase_display.setStyleSheet("""
                QLabel {
                    background-color: black;
                    color: white;
                    padding: 5px;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            
            self.init_status()
            self.ChooseVideo.setEnabled(True)
            self.Start.setEnabled(True)
            self.Stop.setEnabled(False)
            
        except Exception as e:
            logging.error(f"Error in click_stop: {e}")

    def click_choose_report(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "选取文件",
                                                                os.getcwd(),  # Default dir
                                                                "All Files (*);;Text Files (*.csv)")  # 设置文件扩展名过滤,用双分号间隔

        self.reportfile = fileName_choose
        self.ReportFile.setText(fileName_choose)

    def click_report(self):
        try:
            self.Report.setEnabled(False)
            records_dir = "../Records"
            
            if not os.path.exists(records_dir):
                QtWidgets.QMessageBox.warning(self, "Warning", "No records found in ../Records directory")
                return
                
            log_files = glob(os.path.join(records_dir, "*.csv"))
            if not log_files:
                QtWidgets.QMessageBox.warning(self, "Warning", "No log files found to generate report")
                return
                
            report_path = generate_report(records_dir)
            if report_path:
                report_dir = os.path.realpath("./reports")
                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(report_dir))
                QtWidgets.QMessageBox.information(self, "Success", "Report generated successfully!")
            
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")
        finally:
            self.Report.setEnabled(True)

    def click_independent(self):
        self.INDEPENDENT = True
        self.INIT = True
        self.STATUS = "Indepedent"
        # self.log_data.append(["Independent"] * 5)
        self.ActionIndependent.setEnabled(False)
        self.ActionHelp.setEnabled(True)
        self.ActionTakeOver.setEnabled(True)

    def click_help(self):
        # self.log_data.append(["Help"] * 5)
        self.INIT = True
        self.STATUS = "Help"
        self.ActionIndependent.setEnabled(True)
        self.ActionHelp.setEnabled(False)
        self.ActionTakeOver.setEnabled(True)

    def click_take_over(self):
        self.INIT = True
        self.STATUS = "TakeOver"
        # self.log_data.append(["TakeOver"] * 5)
        self.ActionIndependent.setEnabled(True)
        self.ActionHelp.setEnabled(True)
        self.ActionTakeOver.setEnabled(False)

    def save_log_data(self):
        if not self.log_data:
            return
            
        try:
            datas = zip(*self.log_data)
            data_dict = {}
            names = ["Time", "Frame", "Trainee", "Trainer", "Bed", "Status", "FPS", "Prediction", "Correction"]
            
            for name, data in zip(names, datas):
                # Convert all data to strings
                data_dict[name] = [str(x) if x is not None else "--" for x in data]
                
            pd_log = pd.DataFrame.from_dict(data_dict)
            
            # Handle predictions and corrections
            preds = pd_log["Prediction"].tolist()
            correcs = pd_log["Correction"].tolist()
            combines = [corr if corr != "--" else pred for pred, corr in zip(preds, correcs)]
            pd_log["Combine"] = combines
            
            # Save to file
            current_time = datetime.now().strftime("%H-%M-%S")
            save_path = self.log_file.replace(".csv", f"_{current_time}.csv")
            pd_log.to_csv(save_path, index=False, header=True)
            
        except Exception as e:
            logging.error(f"Error saving log data: {e}")

    def get_frame_size(self):
        """Get default frame size or use standard values"""
        try:
            if hasattr(self, 'video_path') and self.video_path:
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    return frame_width, frame_height, fps
            
            # Default values if no video or failed to get properties
            return 1920, 1080, 30
                
        except Exception as e:
            logging.error(f"Error getting frame size: {e}")
            return 1920, 1080, 30  # Default values

    def process_img(self, cv_img, frame_idx):
        if self.WORKING:
            try:
                # Process frame
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                if frame_idx % self.down_ratio == 0:
                    self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
                    start_time = time.time()
                    
                    # Get model prediction
                    self.pred = self.phaseseg.seg_frame(rgb_image)
                    end_time = time.time()
                    self.fps = 1/np.round(end_time - start_time, 3)

                    # Update phase display
                    self.phase_display.setText(f"Phase: {self.pred}")
                    
                    # Color-code based on phase
                    phase_colors = {
                        'idle': 'lightblue',
                        'marking': 'yellow',
                        'injection': 'orange', 
                        'dissection': 'red'
                    }
                    color = phase_colors.get(self.pred, 'white')
                    self.phase_display.setStyleSheet(f"""
                        QLabel {{
                            background-color: black;
                            color: {color};
                            padding: 5px;
                            font-size: 16px;
                            font-weight: bold;
                        }}
                    """)

                    # Log data
                    self.log_data.append([
                        self.date_time,
                        str(frame_idx).zfill(7),
                        self.trainee_name,
                        self.trainer_name,
                        self.bed_name,
                        self.STATUS,
                        f"{self.fps:>7.4f}",
                        self.pred,
                        self.manual_set
                    ])
                
            except Exception as e:
                logging.error(f"Error in process_img: {e}")

    def keyPressEvent(self, e):
        pressed_key = e.text()
        if pressed_key == "a":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "idle"
        elif pressed_key == "s":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "marking"
        elif pressed_key == "d":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "injection"
        elif pressed_key == "f":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "dissection"

    def update_image(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Handle manual frame countdown
            self.manual_frame = self.manual_frame - 1
            if self.manual_frame <= 0:
                self.manual_frame = 0
                self.manual_set = "--"
                
            # Add overlay text if initialized
            if self.INIT:
                self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
                if self.manual_frame > 0:
                    self.pred = self.manual_set
                rgb_image = self.phaseseg.add_text(
                    self.date_time, 
                    self.pred,
                    self.trainee_name,
                    rgb_image
                )

            # Convert to Qt format and display
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(
                rgb_image.data, 
                w, h, 
                bytes_per_line,
                QtGui.QImage.Format_RGB888
            )
            p = convert_to_Qt_format.scaled(
                self.disply_width, 
                self.display_height, 
                Qt.KeepAspectRatio
            )
            self.DisplayVideo.setPixmap(QPixmap.fromImage(p))
            
        except Exception as e:
            logging.error(f"Error in update_image: {e}")

    def click_choose_video(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            os.getcwd(),
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if fileName_choose:
            self.video_path = fileName_choose
            self.VideoPathDisplay.setText(fileName_choose)

    def closeEvent(self, event):
        self.click_stop()
        event.accept()

if not os.path.exists("./configs/report_template.png"):
    logging.error("Report template file not found")
    raise FileNotFoundError("Report template file missing: ./configs/report_template.png")

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'qt', 'plugins')

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s", default=False, action='store_true', help="Whether save predictions")
    parse.add_argument("-q", default=False, action='store_true', help="Display video")
    parse.add_argument("--cfg", default="test", type=str)
    parse.add_argument("--video_path", type=str, required=False, help="Path to input video file")

    cfg = parse.parse_args()
    cfg = ParserUse(cfg.cfg, "camera").add_args(cfg)
    
    # Add video path to config
    cfg.video_path = cfg.video_path

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    ui = Ui_iPhaser()
    ui.setupUi(cfg)
    ui.show()
    sys.exit(app.exec_())

