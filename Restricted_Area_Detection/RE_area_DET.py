#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:58:56 2024

@author: zok
"""

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
from shapely.geometry import Polygon

model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture("/content/drive/MyDrive/20240116_161459.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center_point = (-10, h)
# Get ROI of restrited area
restricted_area = [(624,442),(1602,1057),(262,1062),(0,690)] # Change coordinates according your preferred restricted area
restricted_area_shapely = Polygon(restricted_area)

while True:
    ret, im0 = cap.read()

    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
          x1, y1, x2, y2 = box
          x3, y3 = x1+abs(x2-x1), y1
          x4, y4 = x1, y1+abs(y1-y2)

          restricted_area_np = np.array(restricted_area)
          cv2.polylines(im0, [restricted_area_np], True,
                    (255, 0, 0), 4)

          person_polygon_shapely = Polygon(
            [(x1, y1), (x4, y4), (x2, y2), (x3, y3)])
          intersection_area = restricted_area_shapely.intersection(
            person_polygon_shapely).area
          union_area = restricted_area_shapely.union(person_polygon_shapely).area
          iou = intersection_area / union_area if union_area > 0 else 0

          label2 = str("Person_in_restricted_area")

          if names.get(cls) == 'person':
            if iou > 0.01:
              annotator.box_label(box, label=(str(track_id)+label2), color=colors(int(track_id)))
              annotator.visioneye(box, center_point)
            else:
              annotator.box_label(box, label=names[int(cls)], color=colors(int(track_id)))
              annotator.visioneye(box, center_point)

    out.write(im0)
    # cv2.imshow("visioneye-pinpoint", im0)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

out.release()
cap.release()
cv2.destroyAllWindows()
