#! /usr/bin/env python3
import rospy
import time
import cv2 
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import yaml
from darknet_images import *

thresh = 0.5
yolo_freq = 5
yolo_show = False
show_time = False

from sensor_msgs.msg import Image
last_detections = None
last_detection_time = None
last_image = None

yolo_time = []
yolo_predict_time = []

# get this file path
file_path = os.path.dirname(os.path.realpath(__file__))
yolo_path = os.path.join(file_path, 'yolo')
config_file = os.path.join(yolo_path, "ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1-xl.cfg")
data_file   = os.path.join(yolo_path, "ModelZoo/yolo-fastest-1.1_coco/coco.data")
weights     = os.path.join(yolo_path, "ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1-xl.weights")
dynamic_objects = ["person", "car", "bus", "truck", "bicycle", "motorbike", "train"]
# load yolo network
random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    os.path.join(yolo_path, config_file),
    os.path.join(yolo_path, data_file),
    os.path.join(yolo_path, weights),
    batch_size=1
)


def feature_track(last_image, image, last_detections):
    # Yolo prediction using optical flow
    # print("type:", type(last_detections))
    # print("len:", len(last_detections))
    detections = []
    p0_list = []
    p1_list = []
    index_list = []
    
    for detection in last_detections:
        label, confidence = detection[0], detection[1]
        if label in dynamic_objects:
            x, y, w, h = detection[2]
            # print("x: ", x, "y: ", y, "w: ", w, "h: ", h)
            left = max(int(x - w / 2), 0)
            top = max(int(y - h / 2), 0)
            right = min(int(x + w / 2), image.shape[1])
            bottom = min(int(y + h / 2), image.shape[0])
            # get template image
            img_tmp = last_image[top:bottom, left:right]
            # goodFeaturesToTrack
            p0 = cv2.goodFeaturesToTrack(img_tmp, maxCorners=100, qualityLevel=0.01, minDistance=7)
            if p0 is None:
                continue
            # get p0 in original image
            # print("p0: ", p0)
            p0 = p0 + np.array([left, top])
            p0 = np.array(p0, dtype=np.float32)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(last_image, image, p0, None, maxLevel=3)
            st = st.squeeze()
            p1 = p1[st == 1]
            if len(p1) == 0:
                continue
            p1 = np.array(p1)
            p0 = p0[st == 1]
            # print("p1: ", p1.shape)
            # calculate everage displacement
            displacement = p1 - p0
            # delete outliers using 2sigma rule
            displacement = displacement.reshape(displacement.shape[0], 2)
            mean = np.mean(displacement, axis=0)
            std = np.std(displacement, axis=0)
            index = np.where(np.abs(displacement[:, 0] - mean[0]) < 2 * std[0]) and np.where(np.abs(displacement[:, 1] - mean[1]) < 2 * std[1])
            displacement = displacement[index]
            if len(displacement) == 0:
                continue
            # if len(displacement) < len(p0):
            #     print("outliers: ", len(p0) - len(displacement))
            displacement = np.mean(displacement, axis=0)
            index_list.append(index)
            p0_list.append(p0)
            p1_list.append(p1)
            # detections.append((label, confidence, (x + displacement[0], y + displacement[1], w, h)))
            
            # use affine transformation to predict bbox
            p0_well = p0[index]
            p0_well = p0_well.reshape(p0_well.shape[0], 2)
            p1_well = p1[index]
            p1_well = p1_well.reshape(p1_well.shape[0], 2)
            X = p0_well[:, 0]
            Y = p0_well[:, 1]
            X_ = p1_well[:, 0]
            Y_ = p1_well[:, 1]
            lambda_x = (np.mean(X_ * X) - np.mean(X_) * np.mean(X)) / (np.mean(X**2) - np.mean(X)**2)
            lambda_y = (np.mean(Y_ * Y) - np.mean(Y_) * np.mean(Y)) / (np.mean(Y**2) - np.mean(Y)**2)
            d_x = np.mean(X_) - lambda_x * np.mean(X)
            d_y = np.mean(Y_) - lambda_y * np.mean(Y)
            # print("lambda_x: ", lambda_x, "lambda_y: ", lambda_y, "d_x: ", d_x, "d_y: ", d_y)
            # print("displacement: ", displacement)
            detections.append((label, confidence, (x*lambda_x + d_x, y*lambda_y + d_y, w*lambda_x, h*lambda_y)))
            
    return detections, p0_list, p1_list, index_list

def image_callback(msg):
    global last_detections, last_detection_time, last_image
    # rospy.loginfo("Received image")
    # Do something with the image
    image_time = msg.header.stamp.to_sec()
    cv_image = CvBridge()
    if msg.encoding == '8UC1':
        img = Image()
        img.header = msg.header
        img.height = msg.height
        img.width = msg.width
        img.is_bigendian = msg.is_bigendian
        img.step = msg.step
        img.data = msg.data
        img.encoding = "mono8"
        # copy to cv2
        try:
            cv_image = bridge.imgmsg_to_cv2(img, "mono8")
        except CvBridgeError as e:
            print(e)
    else:
        cv_image = bridge.imgmsg_to_cv2(msg, "mono8")

    # convert to rgb
    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
    
    # if yolo frequency < yolo_freq, use yolo detection
    if last_detections == None or rospy.get_time() - last_detection_time > 1/yolo_freq:
        # yolo
        last_detection_time = rospy.get_time()
        yolo_tic = time.time()
        yolo_image, detections = image_detection2(cv_image, network, class_names, class_colors, thresh)
        last_detections = detections
        last_image = cv_image
        # darknet.print_detections(detections, True)
        yolo_toc = time.time()
        yolo_time.append(yolo_toc - yolo_tic)
        if show_time:
            print("Yolo time: ", yolo_toc - yolo_tic)
        # pub yolo
        yolo_msg = bridge.cv2_to_imgmsg(yolo_image, "bgr8")
        yolo_msg.header.stamp = rospy.Time.from_sec(image_time)
        if yolo_show:
            try:
                pub_yolo.publish(yolo_msg)
            except CvBridgeError as e:
                print(e)
    # else use last detection to predict detection
    else:
        tic = time.time()
        detections, p0_list, p1_list, index_list = feature_track(last_image, cv_image, last_detections)
        toc = time.time()
        yolo_predict_time.append(toc - tic)
        if show_time:
            print("Yolo Predict time: ", toc - tic)
        last_detections = detections
        last_image = cv_image
        # darknet.print_detections(detections, True)
        
    # print("yolo/yolo_predict", len(yolo_time)/(len(yolo_time) + len(yolo_predict_time)), '/', len(yolo_predict_time)/(len(yolo_time) + len(yolo_predict_time)))
    # print("mean yolo time/mean yolo predict time", np.mean(yolo_time), '/', np.mean(yolo_predict_time))

    yolo_mask = np.ones(cv_image.shape[:2], dtype=np.uint8) *255
    for detection in detections:
        label, confidence = detection[0], detection[1]
        margin = 20
        if label in dynamic_objects:
            try:
                x, y, w, h = detection[2]
                left = max(int(x - w / 2) - margin, 0)
                top = max(int(y - h / 2) - margin, 0)
                right = min(int(x + w / 2) + margin, cv_image.shape[1])
                bottom = min(int(y + h / 2) + margin, cv_image.shape[0])
                yolo_mask[top:bottom, left:right] = 0
            except Exception as e:
                continue
    
    # publish yolo mask
    yolo_mask_msg = bridge.cv2_to_imgmsg(yolo_mask, "mono8")
    yolo_mask_msg.header.stamp = rospy.Time.from_sec(image_time)
    try:
        pub_yolo_mask.publish(yolo_mask_msg)
    except CvBridgeError as e:
        print(e)
    


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please provide a config file")
        exit()
    
    # load yaml file
    config_file = sys.argv[1]
    with open(config_file, 'r') as file:
        file.readline()
        # read yaml file 1-22 line
        lines = [file.readline() for i in range(22)]
        config = yaml.load(''.join(lines), Loader=yaml.FullLoader)
        image_topic = config["image0_topic"]
        yolo_topic = config["yolo_topic"]
        mask_topic = config["mask_topic"]
        yolo_freq = config["yolo_freq"]
        thresh = config["yolo_threshold"]
        yolo_show = config["yolo_show"]
    file.close()
    
    bridge = CvBridge()
    # publisher
    pub_yolo = rospy.Publisher(yolo_topic, Image, queue_size=100)
    pub_yolo_mask = rospy.Publisher(mask_topic, Image, queue_size=100)
    
    rospy.init_node('YoloSegNode')
    rospy.loginfo("YoloSeg Node Started")
    rospy.loginfo("Image Topic for Yolo: " + image_topic)
    rospy.loginfo("Yolo Topic: " + yolo_topic)
    rospy.loginfo("Mask Topic: " + mask_topic)
    rospy.loginfo("Yolo Frequency: " + str(yolo_freq))
    rospy.loginfo("Yolo Threshold: " + str(thresh))
    rospy.loginfo("Yolo Show: " + str(yolo_show))
    
    # subscriber
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=100)
    
    rospy.spin()