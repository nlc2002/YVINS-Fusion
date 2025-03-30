#! /usr/bin/env python3
import numpy as np
import rospy
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import NavSatFix
from kaist_align_node import pose_align, align
import pymap3d as pm

# The local coordinate origin
lat0 = 0 # deg
lon0 = 0  # deg
h0 = 0     # meters

# global variables
# gnss_local_offset = 18.0
gnss_local_offset = 0
is_aligned = False
vins_path = []
vins_path_aligned = Path()
vins_seg_path = []
vins_seg_path_aligned = Path()
gt = Path()
gt_offset = np.zeros(3)
t = np.zeros(3)
t_seg = np.zeros(3)
yaw = 0
yaw_seg = 0
align_length = 500
align_length_start = 200

def lla2ENU(lat, lon, alt):
    global lat0, lon0, h0
    if len(gt.poses) == 0:
        lat0 = lat
        lon0 = lon
        h0 = alt
        return np.zeros(3)
    else:
        return np.array(pm.geodetic2enu(lat, lon, alt, lat0, lon0, h0))

def lla_callback(data: NavSatFix):
    global t, yaw, is_aligned, yaw_seg, t_seg, gt_offset
    gt.header = data.header
    gt.header.frame_id = "world"
    gt.header.stamp = data.header.stamp - rospy.Duration(gnss_local_offset)
    print("Received GNSS", gt.header.stamp.to_sec())
    gt_pose = PoseStamped()
    gt_pose.header = gt.header
    latitude = data.latitude
    longitude = data.longitude
    altitude = data.altitude
    # print(f"Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}")
    enu = lla2ENU(latitude, longitude, altitude)
    if len(gt.poses) == 0:
        gt_offset[0] = enu[0]
        gt_offset[1] = enu[1]
        gt_offset[2] = enu[2]
    gt_pose.pose.position.x = enu[0] - gt_offset[0]
    gt_pose.pose.position.y = enu[1] - gt_offset[1]
    gt_pose.pose.position.z = enu[2] - gt_offset[2]
    # print("ENU: ", gt_pose.pose.position.x, gt_pose.pose.position.y, gt_pose.pose.position.z)
    gt_pose.pose.orientation.w = 1
    gt.poses.append(gt_pose)
    pub_aligned_gt.publish(gt)
    with open(gt_file, 'a') as f:
        f.write(str(data.header.stamp.to_sec()) + ',' + str(gt_pose.pose.position.x) + ',' + str(gt_pose.pose.position.y) + ',' + str(gt_pose.pose.position.z) + ',' + str(gt_pose.pose.orientation.x) + ',' + str(gt_pose.pose.orientation.y) + ',' + str(gt_pose.pose.orientation.z) + ',' + str(gt_pose.pose.orientation.w) + '\n')
    if is_aligned:
        return
    else:
        if len(gt.poses) > align_length:
            yaw, t = align(gt, vins_path, align_length_start)
            print("Align VINS to GT: yaw: ", yaw, "t: ", t)
            yaw_seg, t_seg = align(gt, vins_seg_path, align_length_start)
            print("Align VINS_SEG to GT: yaw: ", yaw_seg, "t: ", t_seg)
            is_aligned = True
    return

def vins_path_callback(data: Path):
    rospy.logdebug("Received VINS Path")
    print("Received VINS Path", data.header.stamp.to_sec())
    if is_aligned:
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
        if len(vins_path_aligned.poses) == 0:
            with open(vins_aligned_file, 'a') as f:
                for pose in data.poses:
                    pose_aligned = pose_align(pose, R, t)
                    vins_path_aligned.poses.append(pose_aligned)
                    f.write(str(pose_aligned.header.stamp.to_sec()) + ',' + str(pose_aligned.pose.position.x) + ',' + str(pose_aligned.pose.position.y) + ',' + str(pose_aligned.pose.position.z) + ',' + str(pose_aligned.pose.orientation.x) + ',' + str(pose_aligned.pose.orientation.y) + ',' + str(pose_aligned.pose.orientation.z) + ',' + str(pose_aligned.pose.orientation.w) + '\n')
        else:
            pose = data.poses[-1]
            pose_aligned = pose_align(pose, R, t)
            vins_path_aligned.poses.append(pose_aligned)
            with open(vins_aligned_file, 'a') as f:
                f.write(str(pose_aligned.header.stamp.to_sec()) + ',' + str(pose_aligned.pose.position.x) + ',' + str(pose_aligned.pose.position.y) + ',' + str(pose_aligned.pose.position.z) + ',' + str(pose_aligned.pose.orientation.x) + ',' + str(pose_aligned.pose.orientation.y) + ',' + str(pose_aligned.pose.orientation.z) + ',' + str(pose_aligned.pose.orientation.w) + '\n')
        vins_path_aligned.header = data.header
        pub_aligned_path.publish(vins_path_aligned)
                
    else:
        # nav_msgs/Path:
        # print('poses length: ', len(data.poses))
        # print('poses[0]: ', data.poses[0])
        stamp = [data.header.stamp.to_sec(), 
                data.poses[-1].pose.position.x, 
                data.poses[-1].pose.position.y, 
                data.poses[-1].pose.position.z]
        vins_path.append(stamp)
        # print("VINS: ", stamp)
    
def vins_seg_path_callback(data):
    rospy.logdebug("Received VINS_SEG Path")
    print("Received VINS_SEG Path", data.header.stamp.to_sec())
    if is_aligned:
        R = np.array([[np.cos(yaw_seg), -np.sin(yaw_seg), 0],
                    [np.sin(yaw_seg), np.cos(yaw_seg), 0],
                    [0, 0, 1]])
        if len(vins_seg_path_aligned.poses) == 0:
            with open(vins_seg_aligned_file, 'a') as f:
                for pose in data.poses:
                    pose_aligned = pose_align(pose, R, t_seg)
                    vins_seg_path_aligned.poses.append(pose_aligned)
                    # print euler angles
                    # q = [pose_aligned.pose.orientation.x, pose_aligned.pose.orientation.y, pose_aligned.pose.orientation.z, pose_aligned.pose.orientation.w]
                    # R_ = q2R(q)
                    # euler = Rotation.from_matrix(R_).as_euler('zyx')
                    # print("VINS_SEG Euler: ", euler)
                    f.write(str(pose_aligned.header.stamp.to_sec()) + ',' + str(pose_aligned.pose.position.x) + ',' + str(pose_aligned.pose.position.y) + ',' + str(pose_aligned.pose.position.z) + ',' + str(pose_aligned.pose.orientation.x) + ',' + str(pose_aligned.pose.orientation.y) + ',' + str(pose_aligned.pose.orientation.z) + ',' + str(pose_aligned.pose.orientation.w) + '\n')
        else:
            pose = data.poses[-1]
            pose_aligned = pose_align(pose, R, t_seg)
            vins_seg_path_aligned.poses.append(pose_aligned)
            # q = [pose_aligned.pose.orientation.x, pose_aligned.pose.orientation.y, pose_aligned.pose.orientation.z, pose_aligned.pose.orientation.w]
            # R_ = q2R(q)
            # euler = Rotation.from_matrix(R_).as_euler('zyx')
            # print("VINS_SEG Euler: ", euler)
            with open(vins_seg_aligned_file, 'a') as f:
                f.write(str(pose_aligned.header.stamp.to_sec()) + ',' + str(pose_aligned.pose.position.x) + ',' + str(pose_aligned.pose.position.y) + ',' + str(pose_aligned.pose.position.z) + ',' + str(pose_aligned.pose.orientation.x) + ',' + str(pose_aligned.pose.orientation.y) + ',' + str(pose_aligned.pose.orientation.z) + ',' + str(pose_aligned.pose.orientation.w) + '\n')
        vins_seg_path_aligned.header = data.header
        pub_aligned_path_seg.publish(vins_seg_path_aligned)
    else:
        # nav_msgs/Path:
        stamp = [data.header.stamp.to_sec(), 
                data.poses[-1].pose.position.x, 
                data.poses[-1].pose.position.y, 
                data.poses[-1].pose.position.z,]
        vins_seg_path.append(stamp)
        # print("VINS_SEG: ", stamp)
        



if __name__ == '__main__':
    lla_topic = "/ublox_driver/receiver_lla"
    vins_path_topic = "/vins_estimator/path"
    vins_seg_path_topic = "/vins_estimator_seg/path"
    rospy.init_node('align', log_level=rospy.INFO)
    rospy.loginfo("Align Node Started")
    
    # save aligned path to file
    dir = os.path.dirname(os.path.realpath(__file__))
    gt_file = dir + '/zed2i_gt_aligned.csv'
    vins_aligned_file = dir + '/zed2i_vins_aligned.csv'
    vins_seg_aligned_file = dir + '/zed2i_vins_seg_aligned.csv'
    if os.path.exists(gt_file):
        os.remove(gt_file)
    if os.path.exists(vins_aligned_file):
        os.remove(vins_aligned_file)
    if os.path.exists(vins_seg_aligned_file):
        os.remove(vins_seg_aligned_file)
    
    # subscriber
    rospy.Subscriber(lla_topic, NavSatFix, lla_callback, queue_size=100)
    rospy.Subscriber(vins_path_topic, Path, vins_path_callback, queue_size=100)
    rospy.Subscriber(vins_seg_path_topic, Path, vins_seg_path_callback, queue_size=100)
    
    # publish aligned path
    pub_aligned_gt = rospy.Publisher("/gt_path", Path, queue_size=100)
    pub_aligned_path = rospy.Publisher("/aligned_path", Path, queue_size=100)
    pub_aligned_path_seg = rospy.Publisher("/aligned_path_seg", Path, queue_size=100)

    rospy.spin()
