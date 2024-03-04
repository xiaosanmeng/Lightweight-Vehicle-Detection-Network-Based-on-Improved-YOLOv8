import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.val(data='F:/YOLOv8/dataset_car/data_car/data_set.yaml',
              split='val',
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='val-fasternet-bifpn-NWDloss-SIoU-dyhead',
              )
# if __name__ == '__main__':
#     model = YOLO('F:/YOLOv8/yolov8-main/runs/detect/yolov8-C2f-SCcConv-n/weights/best.pt')
#     model.val(**{'data': 'F:/YOLOv8/dataset_car/data_car/data_set.yaml'})
