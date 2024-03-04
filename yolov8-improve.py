from ultralytics import YOLO

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    model = YOLO('F:/YOLOv8/yolov8-main/ultralytics/cfg/yolov8s-fasternet-bifpn-dyhead.yaml')
    model.train(**{'cfg': 'F:/YOLOv8/yolov8-main/ultralytics/cfg/default.yaml',
                   'data': 'F:/YOLOv8/dataset_car/data_car/data_set.yaml'})

    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model.load('yolov8n.pt')
    # model.train(**{'cfg': 'ultralytics/cfg/exp1.yaml', 'data': 'dataset/data.yaml'})
    #
    #
    # model = YOLO('runs/detect/yolov8n_exp/weights/best.pt')
    # model.val(**{'data': 'dataset/data.yaml'})
    #
    #
    # model = YOLO('runs/detect/yolov8n_exp/weights/best.pt')
    # model.predict(source='dataset/images/test', **{'save': True})
