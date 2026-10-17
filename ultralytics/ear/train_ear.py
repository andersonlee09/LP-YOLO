from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train():
    # model = YOLO("./ultralytics/cfg/models/v8/mtyolov8.yaml")
    # model = YOLO("./yolov8n.pt")
    model = YOLO("../cfg/models/11/yolo11-ear-pose.yml", verbose=True)
    model.train(data="./mt.yaml", epochs=200)
    result = model.val()



if __name__ == '__main__':
    train()
