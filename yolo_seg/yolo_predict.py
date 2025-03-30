#! /usr/bin/env python3

import time
import cv2 
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import os
from darknet_images import *
from yolo_seg_node import *

thresh = 0.25
yolo_freq = 5

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    os.path.join(yolo_path, config_file),
    os.path.join(yolo_path, data_file),
    os.path.join(yolo_path, weights),
    batch_size=1
)

path = '/media/nlc20/Research/datas/urban_driving/cam0/image_raw'

def yolo_predict(last_image, image, last_detections):
    detections, p0_list, p1_list, index_list = feature_track(last_image, image, last_detections)
    return detections, p0_list, p1_list, index_list

def draw_points(image, p_list, index_list):
    for i in range(len(p_list)):
        # points[index] green, others red
        points = p_list[i]
        index = index_list[i]
        for p in points:
            p = p[0]
            cv2.circle(image, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        p_green = points[index]
        for p in p_green:
            p = p[0]
            cv2.circle(image, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        
    return image

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        if label in dynamic_objects:
            left, top, right, bottom = bbox2points(bbox)
            cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[label], 2)
    return image

    

if __name__ == "__main__":
    images = os.listdir(path)
    images.sort()
    # select images with step 3
    images = images[::2]
    last_image = None
    last_detections = None
    last_yolo_image = None
    for image in images:
        image = cv2.imread(os.path.join(path, image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if last_image is not None and last_detections is not None:
            detections, p0_list, p1_list, index_list = yolo_predict(last_image, image, last_detections)
            last_yolo_image = draw_boxes(last_detections, cv2.cvtColor(last_image, cv2.COLOR_GRAY2BGR), class_colors)
            yolo_image = draw_boxes(detections, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), class_colors)
            # show last yolo image and current yolo image
            cv2.imshow("last_yolo_image", last_yolo_image)
            cv2.imshow("yolo_image", yolo_image)
            last_yolo_image_points = draw_points(last_yolo_image, p0_list, index_list)
            yolo_image_points = draw_points(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), p1_list, index_list)
            # show last yolo image and current yolo image with points
            cv2.imshow("last_yolo_image_points", last_yolo_image_points)
            cv2.imshow("yolo_image_points", yolo_image_points)
            # # if press 's', save this four images
            # save_path = "~/Ingvio/src/InGVIO/yolo_seg/figures"
            # if cv2.waitKey(0) & 0xFF == ord('s'):
            #     cv2.imwrite(os.path.join(save_path, "last_yolo_image.jpg"), last_yolo_image)
            #     cv2.imwrite(os.path.join(save_path, "yolo_image.jpg"), yolo_image)
            #     cv2.imwrite(os.path.join(save_path, "last_yolo_image_points.jpg"), last_yolo_image_points)
            #     cv2.imwrite(os.path.join(save_path, "yolo_image_points.jpg"), yolo_image_points)
            cv2.waitKey(0)
            
        last_yolo_image, detections = image_detection2(image, network, class_names, class_colors, thresh)
        last_detections = detections
        last_image = image