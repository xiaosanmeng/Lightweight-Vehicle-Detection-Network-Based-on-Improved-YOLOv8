from ultralytics import YOLO

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    model = YOLO('F:/YOLOv8/yolov8-main/runs/detect/train/weights/last.pt')
    model.train(resume=True)