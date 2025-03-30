import csv
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import os
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R

# set gt_msg frequency to 10Hz
gf_freq = 10

def R2q(R_matrix):
    """
    Converts a rotational matrix to a unit quaternion.
    """
    return R.from_matrix(R_matrix).as_quat()

def q2R(q):
    """
    Converts a unit quaternion to a rotational matrix.
    """
    return R.from_quat(q).as_matrix()

def csv2rosbag(csv_file, bag_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    print('data length: ' + str(len(data)))
    bag = rosbag.Bag(bag_file, 'w')
    offset = np.zeros(3)
    last_time = 0
    for row in data:
        timestamp = float(row[0])/1e9
        if timestamp - last_time < 1.0/gf_freq:
            continue
        last_time = timestamp
        # print('timestamp: ' + str(timestamp))
        R = np.zeros((3,3))
        R[0,0] = float(row[1])
        R[0,1] = float(row[2])
        R[0,2] = float(row[3])
        R[1,0] = float(row[5])
        R[1,1] = float(row[6])
        R[1,2] = float(row[7])
        R[2,0] = float(row[9])
        R[2,1] = float(row[10])
        R[2,2] = float(row[11])
        t = np.zeros(3)
        if offset[0] == 0:
            offset[0] = float(row[4])
            offset[1] = float(row[8])
            offset[2] = float(row[12])
        t[0] = float(row[4]) - offset[0]
        t[1] = float(row[8]) - offset[1]
        t[2] = float(row[12]) - offset[2]
        q = R2q(R)
        gt_pose = PoseStamped()
        gt_pose.header.stamp = rospy.Time.from_sec(timestamp)
        gt_pose.header.frame_id = 'world'
        gt_pose.pose.position.x = t[0]
        gt_pose.pose.position.y = t[1]
        gt_pose.pose.position.z = t[2]
        gt_pose.pose.orientation.x = q[0]
        gt_pose.pose.orientation.y = q[1]
        gt_pose.pose.orientation.z = q[2]
        gt_pose.pose.orientation.w = q[3]
        bag.write('/gt_pose', gt_pose, gt_pose.header.stamp)
    bag.close()
    print('Saved ' + bag_file)
        

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    datasets = os.listdir(dir_path)
    # datasets = os.listdir('/home/nlc20/datas/KAIST')
    datasets = [dataset for dataset in datasets if 'urban' in dataset ]
    p = Pool(8)
    for dataset in datasets:
        gt_file = os.path.join(dir_path, dataset, 'global_pose.csv')
        # gt_file = os.path.join('/home/nlc20/datas/KAIST', dataset, 'global_pose.csv')
        bag_file = os.path.join(dir_path, dataset, 'global_pose.bag')
        # csv2rosbag(gt_file, bag_file)
        # if '39' in dataset:
        p.apply_async(csv2rosbag, args=(gt_file, bag_file))
    p.close()
    p.join()
    print('Done')
        