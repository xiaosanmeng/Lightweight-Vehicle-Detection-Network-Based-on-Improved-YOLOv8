from ultralytics import YOLO
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# if __name__ == '__main__':
#     model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
#     model.load('yolov8n.pt') # loading pretrain weights
#     model.train(data='dataset/data.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=100,
#                 batch=16,
#                 close_mosaic=10,
#                 workers=4,
#                 device='0',
#                 optimizer='SGD', # using SGD
#                 # resume='', # last.pt path
#                 # amp=False # close amp
#                 # fraction=0.2,
#                 project='runs/train',
#                 name='exp',
#                 )
if __name__ == '__main__':
    model = YOLO('F:/YOLOv8/yolov8-main/ultralytics/cfg/models/v8/yolov8s.yaml')
    model.train(
        **{'cfg': 'F:/YOLOv8/yolov8-main/ultralytics/cfg/test1.yaml',
           'data': 'F:/YOLOv8/dataset_car/data_car/data_set.yaml',
           'name': 'test_SIOU'})
