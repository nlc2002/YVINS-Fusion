#! /usr/bin/env python3
import numpy as np
import rospy
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.optimize import leastsq
from draw import R2q, q2R
from scipy.spatial.transform import Rotation

# global variables
is_aligned = False
vins_path = []
vins_path_aligned = Path()
vins_seg_path = []
vins_seg_path_aligned = Path()
gt = Path()
t = np.zeros(3)
t_seg = np.zeros(3)
yaw = 0
yaw_seg = 0
align_length = 300
align_length_start = 100

def pose_align(pose_stamp, R, t):
    pose_aligned = PoseStamped()
    pose_aligned.header = pose_stamp.header
    x = pose_stamp.pose.position.x
    y = pose_stamp.pose.position.y
    z = pose_stamp.pose.position.z
    qx = pose_stamp.pose.orientation.x
    qy = pose_stamp.pose.orientation.y
    qz = pose_stamp.pose.orientation.z
    qw = pose_stamp.pose.orientation.w
    q = np.array([qx, qy, qz, qw])
    R0 = q2R(q)
    R_new = np.dot(R, R0)
    q_new = R2q(R_new)
    [x_new, y_new, z_new] = np.dot(R, np.array([x, y, z])) + t
    pose_aligned.pose.position.x = x_new
    pose_aligned.pose.position.y = y_new
    pose_aligned.pose.position.z = z_new
    # pose_aligned.pose.orientation = pose_stamp.pose.orientation
    pose_aligned.pose.orientation.x = q_new[0]
    pose_aligned.pose.orientation.y = q_new[1]
    pose_aligned.pose.orientation.z = q_new[2]
    pose_aligned.pose.orientation.w = q_new[3]
    return pose_aligned

def align(gt_path:Path, path_buffer, align_length_start=100):
    # align path to gt
    # gt_path
    # path_buffer: [[time, x, y, z], ...]
    # return: yaw, t
    gt_buffer = []
    for pose in gt_path.poses:
        gt_buffer.append([pose.header.stamp.to_sec(),
                          pose.pose.position.x,
                          pose.pose.position.y,
                          pose.pose.position.z])
    time_start = max(gt_buffer[align_length_start][0], path_buffer[align_length_start][0])
    time_end = min(gt_buffer[-1][0], path_buffer[-1][0])    
    gt = np.array([entry for entry in gt_buffer if time_start <= entry[0] <= time_end])
    path = np.array([entry for entry in path_buffer if time_start <= entry[0] <= time_end])
    # Interpolation to align time, path aligned to gt
    path_interp = np.zeros((len(gt), 3))
    
    for i in range(3):
        path_interp[:, i] = np.interp(gt[:, 0], path[:, 0], path[:, i+1])
    
    # print("path_interp: ", path_interp)
    # find R，t to align path to gt use SVD
    q = gt[:, 1:4]
    p = path_interp
    x = p - np.mean(p, axis=0)
    y = q - np.mean(q, axis=0)
    S = np.dot(x.T, y)
    U, _, Vt = np.linalg.svd(S)
    R = np.dot(np.dot(Vt.T, 
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(np.dot(Vt.T, U.T))]]).T), 
                U.T)
    t = np.mean(q, axis=0) - np.dot(R, np.mean(p, axis=0))
    # to eular angles
    R_eular = Rotation.from_matrix(R).as_euler('zyx')
    return R_eular[0], t
    
    
def gt_callback(data: PoseStamped):
    rospy.logdebug("Received GT Pose")
    global is_aligned, yaw, t, yaw_seg, t_seg
    gt.header = data.header
    gt_pose = PoseStamped()
    gt_pose.header = data.header
    gt_pose.pose = data.pose
    gt.poses.append(gt_pose)
    pub_aligned_gt.publish(gt)
    # print euler angles
    # q = [gt_pose.pose.orientation.x, gt_pose.pose.orientation.y, gt_pose.pose.orientation.z, gt_pose.pose.orientation.w]
    # R_ = q2R(q)
    # euler = Rotation.from_matrix(R_).as_euler('zyx')
    # print("GT Euler: ", euler)
    with open(gt_file, 'a') as f:
        f.write(str(gt_pose.header.stamp.to_sec()) + ',' + str(gt_pose.pose.position.x) + ',' + str(gt_pose.pose.position.y) + ',' + str(gt_pose.pose.position.z) + ',' + str(gt_pose.pose.orientation.x) + ',' + str(gt_pose.pose.orientation.y) + ',' + str(gt_pose.pose.orientation.z) + ',' + str(gt_pose.pose.orientation.w) + '\n')
    if is_aligned:
        return
    else:
        if len(gt.poses) > align_length: 
            try:
                yaw, t = align(gt, vins_path, align_length_start)
                print("Align VINS to GT: yaw: ", yaw, "t: ", t)
                yaw_seg, t_seg = align(gt, vins_seg_path, align_length_start)
                print("Align VINS_SEG to GT: yaw: ", yaw_seg, "t: ", t_seg)
                is_aligned = True
            except:
                print("Align failed")
                is_aligned = False
    return

def vins_path_callback(data: Path):
    rospy.logdebug("Received VINS Path")
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
    gt_topic = "/gt_pose"
    vins_path_topic = "/vins_estimator/path"
    vins_seg_path_topic = "/vins_estimator_seg/path"
    rospy.init_node('align', log_level=rospy.INFO)
    rospy.loginfo("Align Node Started")
    rospy.loginfo("Ground Truth Topic: " + gt_topic)
    
    # save aligned path to file
    dir = os.path.dirname(os.path.realpath(__file__))
    gt_file = dir + '/gt.csv'
    vins_aligned_file = dir + '/vins_aligned.csv'
    vins_seg_aligned_file = dir + '/vins_seg_aligned.csv'
    if os.path.exists(gt_file):
        os.remove(gt_file)
    if os.path.exists(vins_aligned_file):
        os.remove(vins_aligned_file)
    if os.path.exists(vins_seg_aligned_file):
        os.remove(vins_seg_aligned_file)
    
    # subscriber
    rospy.Subscriber(gt_topic, PoseStamped, gt_callback, queue_size=100)
    rospy.Subscriber(vins_path_topic, Path, vins_path_callback, queue_size=100)
    rospy.Subscriber(vins_seg_path_topic, Path, vins_seg_path_callback, queue_size=100)
    
    # publish aligned path
    pub_aligned_gt = rospy.Publisher("/gt_path", Path, queue_size=100)
    pub_aligned_path = rospy.Publisher("/aligned_path", Path, queue_size=100)
    pub_aligned_path_seg = rospy.Publisher("/aligned_path_seg", Path, queue_size=100)
    

    
    

    rospy.spin()


    
    

    