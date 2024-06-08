import sys
import os
import shutil
import subprocess
import glob
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar,
    QFileDialog, QMessageBox, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal

class ProcessImagesThread(QThread):
    finished = pyqtSignal()
    progress_updated = pyqtSignal(int)  # Signal for progress updates

    def __init__(self, image_folder, weights_file, save_folder):
        super().__init__()
        self.image_folder = image_folder
        self.weights_file = weights_file
        self.save_folder = save_folder

    def run(self):
        try:
            image_files = glob.glob(os.path.join(self.image_folder, '*.tif'))
            total_files = len(image_files)
            original_paths = {os.path.basename(tif_file): tif_file for tif_file in image_files}

            for index, tif_file in enumerate(image_files, start=1):
                shutil.copy(tif_file, os.getcwd())
                self.progress_updated.emit(int((index / total_files) * 50))  # Update progress up to 50%

            process_command = f"python3 process_images.py *.tif --weights_file '{self.weights_file}'"
            subprocess.run(process_command, shell=True, check=True)

            output_folder = self.find_available_folder_name(os.path.join(self.save_folder, 'PP GUI Output'))
            os.makedirs(output_folder, exist_ok=True)
            for tif_file in glob.glob('*.tif'):
                shutil.move(tif_file, output_folder)

            for index, tif_file_name in enumerate(original_paths, start=1):
                shutil.move(os.path.join(output_folder, tif_file_name), original_paths[tif_file_name])
                self.progress_updated.emit(50 + int((index / total_files) * 50))  # Update progress from 50% to 100%

            if os.path.exists('csv'):
                shutil.move('csv', output_folder)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e.stderr.decode()}")
        finally:
            self.finished.emit()

    def find_available_folder_name(self, base_name):
        counter = 1
        new_name = base_name
        while os.path.exists(new_name):
            new_name = f"{base_name} {counter}"
            counter += 1
        return new_name

class PhotoPhenosizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('PhotoPhenosizer GUI')

        layout = QVBoxLayout()

        self.image_folder_edit = QLineEdit(self)
        self.weights_file_edit = QLineEdit(self)
        self.save_folder_edit = QLineEdit(self)

        self.browse_image_folder_btn = QPushButton('Browse Image Folder', self)
        self.browse_image_folder_btn.clicked.connect(self.browse_image_folder)

        self.browse_weights_file_btn = QPushButton('Browse Weights File', self)
        self.browse_weights_file_btn.clicked.connect(self.browse_weights_file)

        self.browse_save_folder_btn = QPushButton('Browse Save Location', self)
        self.browse_save_folder_btn.clicked.connect(self.browse_save_folder)

        self.run_button = QPushButton('Run', self)
        self.run_button.clicked.connect(self.run_process)

        self.progress = QProgressBar(self)

        layout.addWidget(QLabel('Image Folder:'))
        layout.addWidget(self.image_folder_edit)
        layout.addWidget(self.browse_image_folder_btn)

        layout.addWidget(QLabel('Weights File:'))
        layout.addWidget(self.weights_file_edit)
        layout.addWidget(self.browse_weights_file_btn)

        layout.addWidget(QLabel('Save Location:'))
        layout.addWidget(self.save_folder_edit)
        layout.addWidget(self.browse_save_folder_btn)

        layout.addWidget(self.run_button)
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def browse_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder_edit.setText(folder)

    def browse_weights_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Weights File", "", "PyTorch Weights Files (*.pt)")
        if file:
            self.weights_file_edit.setText(file)

    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Location")
        if folder:
            self.save_folder_edit.setText(folder)

    def run_process(self):
        image_folder = self.image_folder_edit.text()
        weights_file = self.weights_file_edit.text()
        save_folder = self.save_folder_edit.text()

        if not all([image_folder, weights_file, save_folder]):
            QMessageBox.warning(self, 'Error', 'Please select all required paths.')
            return

        self.run_button.setEnabled(False)
        self.thread = ProcessImagesThread(image_folder, weights_file, save_folder)
        self.thread.finished.connect(self.on_finished)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.start()

    def update_progress(self, value):
        self.progress.setValue(value)

    def on_finished(self):
        QMessageBox.information(self, 'Success', 'Image processing completed.')
        self.run_button.setEnabled(True)
        self.progress.setValue(0)
        self.close()  # Close the application

def main():
    app = QApplication(sys.argv)
    ex = PhotoPhenosizerGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
