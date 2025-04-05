import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import scienceplots
from cycler import cycler

calculate_distance = 100000
distance_interval = 1000
#plot
plt.style.use(['science','ieee','no-latex','grid','vibrant'])
plt.rcParams.update({"font.size":4})
# set linewidth
plt.rcParams['lines.linewidth'] = 0.5
# set background line width
plt.rcParams['grid.linewidth'] = 0.3
# set color cycle
default_cycler = (cycler(linestyle=['-', '--', ':', '-.','-','--',':'])+cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']))
plt.rc('axes', prop_cycle=default_cycler)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#1f77b4: blue, ff7f0e: orange, 2ca02c: green, d62728: red, 9467bd: purple, 8c564b: brown, e377c2: pink
# set boxplot style
plt.rcParams['boxplot.boxprops.linewidth'] = 0.3
plt.rcParams['boxplot.whiskerprops.linewidth'] = 0.3
plt.rcParams['boxplot.flierprops.linewidth'] = 0.1
plt.rcParams['boxplot.capprops.linewidth'] = 0.3
plt.rcParams['boxplot.medianprops.linewidth'] = 0.3
plt.rcParams['boxplot.meanprops.linewidth'] = 0.3
box_width = 200

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

def draw_trajectory(gt, path, path_seg, ax):
    """
    Draw the trajectory of the ground truth and the VINS.
    gt: list of ground truth poses, each pose is a list of [timestamp, x, y, z, qx, qy, qz, qw]
    path and path_seg format is the same as gt
    fig: matplotlib
    """
    gt = np.array(gt)
    path = np.array(path)
    path_seg = np.array(path_seg)
    # plot 2d trajectory, x-y plane
    if len(gt) > 0:
        ax.plot(gt[:, 1], gt[:, 2], label='Ground Truth', c=colors[1])
    if len(path) > 0:
        ax.plot(path[:, 1], path[:, 2], label='VINS', c=colors[0])
    if len(path_seg) > 0:
        ax.plot(path_seg[:, 1], path_seg[:, 2],  label='VINS (Yolo-Based)')
    ax.set_xlabel('East[m]')
    ax.set_ylabel('North[m]')
    ax.legend(framealpha=0.5, loc='upper left')
    print("Trajectory plot completed.")
    
def draw_TranslationError(gt, path, path_seg, ax):
    """
    Draw the translation error of the VINS.
    gt: list of ground truth poses, each pose is a list of [timestamp, x, y, z, qx, qy, qz, qw]
    path and path_seg format is the same as gt
    fig: matplotlib Box diagram
    """
    gt = np.array(gt)[:,0:4]
    path = np.array(path)[:,0:4]
    path_seg = np.array(path_seg)[:,0:4]
    distance = np.zeros((len(gt)))
    for i in range(len(gt)-1):
        distance[i+1] = distance[i] + np.linalg.norm(gt[i+1,1:4] - gt[i,1:4])
    # Interpolation of the path and path_seg use np.interp
    path_interp = np.zeros((len(gt), 4))
    path_seg_interp = np.zeros((len(gt), 4))
    path_interp[:,0] = gt[:,0]
    path_seg_interp[:,0] = gt[:,0]
    for i in range(1,4):
        path_interp[:,i] = np.interp(gt[:,0], path[:,0], path[:,i])
        path_seg_interp[:,i] = np.interp(gt[:,0], path_seg[:,0], path_seg[:,i])
    # Calculate the translation error
    error = np.linalg.norm(path_interp[:,1:4] - gt[:,1:4], axis=1)
    error_seg = np.linalg.norm(path_seg_interp[:,1:4] - gt[:,1:4], axis=1)
    # Plot the translation error as Box diagram, xlabel is the distance
    # divide the distance into 10 parts
    distance_all = min(distance[-1], calculate_distance)
    distance_label = np.arange(0, distance_all, distance_interval) + distance_interval
    # to int 
    distance_label = distance_label.astype(int)
    error_interval = []
    error_seg_interval = []
    for i in range(len(distance_label)):
        if i == 0:
            index = np.where(distance < distance_label[i])
        else:   
            index = np.where((distance >= distance_label[i-1]) & (distance < distance_label[i]))
        error_interval.append(error[index])
        error_seg_interval.append(error_seg[index])
    flierprops_set = dict(marker='+', markersize=0.3, markerfacecolor='red')
    ax.boxplot(error_seg_interval, positions=distance_label, widths=box_width, patch_artist=True, boxprops=dict(facecolor=colors[2]), sym = 'r.', flierprops=flierprops_set)
    ax.boxplot(error_interval, positions=distance_label+box_width, widths=box_width, patch_artist=True, boxprops=dict(facecolor=colors[0]), sym = 'r.', flierprops=flierprops_set)
    ax.set_xlabel('Distance[m]')
    ax.set_xticks([])
    ax.set_xticks(distance_label+box_width/2)
    ax.set_ylabel('Translation Error[m]')
    # set legend
    ax.plot([], c = colors[0], label='VINS', linestyle='-')
    ax.plot([], c = colors[2], label='VINS (Yolo-Based)', linestyle='-')
    ax.plot([], c='red', label='Fliers', marker='.', markersize=0.3, linestyle='None')
    ax.legend(loc='upper left', framealpha=0.5)
    print("Translation error plot completed.")
    
def draw_YawError(gt, path, path_seg, ax):
    """
    Draw the yaw error of the VINS.
    gt: list of ground truth poses, each pose is a list of [timestamp, x, y, z, qx, qy, qz, qw]
    path and path_seg format is the same as gt
    fig: matplotlib Box diagram
    """
    gt = np.array(gt)
    path = np.array(path)
    path_seg = np.array(path_seg)
    distance = np.zeros((len(gt)))
    for i in range(len(gt)-1):
        distance[i+1] = distance[i] + np.linalg.norm(gt[i+1,1:4] - gt[i,1:4])
    # Interpolation of the path and path_seg use np.interp
    path_interp = np.zeros((len(gt), 5))
    path_seg_interp = np.zeros((len(gt), 5))
    path_interp[:,0] = gt[:,0]
    path_seg_interp[:,0] = gt[:,0]
    for i in range(1,5):
        path_interp[:,i] = np.interp(gt[:,0], path[:,0], path[:,i+3])
        path_seg_interp[:,i] = np.interp(gt[:,0], path_seg[:,0], path_seg[:,i+3])
    # Calculate the yaw error
    error = np.zeros((len(gt)))
    error_seg = np.zeros((len(gt)))
    for i in range(len(gt)):
        q_gt = gt[i,4:8]
        q_path = path_interp[i,1:5]
        q_path_seg = path_seg_interp[i,1:5]
        R_gt = q2R(q_gt)
        R_path = q2R(q_path)
        R_path_seg = q2R(q_path_seg)
        eular_gt = R.from_matrix(R_gt).as_euler('zyx', degrees=True)
        eular_path = R.from_matrix(R_path).as_euler('zyx', degrees=True)
        eular_path_seg = R.from_matrix(R_path_seg).as_euler('zyx', degrees=True)
        # yaw error
        error[i] = eular_path[0] - eular_gt[0]
        error_seg[i] = eular_path_seg[0] - eular_gt[0]
        error[i] = min(abs(error[i]), 360 - abs(error[i]))
        error_seg[i] = min(abs(error_seg[i]), 360 - abs(error_seg[i]))
    # divide the distance
    distance_all = min(distance[-1], calculate_distance)
    distance_label = np.arange(0, distance_all, distance_interval) + distance_interval
    # to int
    distance_label = distance_label.astype(int)
    error_interval = []
    error_seg_interval = []
    for i in range(len(distance_label)):
        if i == 0:
            index = np.where(distance < distance_label[i])
        else:
            index = np.where((distance >= distance_label[i-1]) & (distance < distance_label[i]))
        error_interval.append(error[index])
        error_seg_interval.append(error_seg[index])
    flierprops_set = dict(marker='+', markersize=0.3, markerfacecolor='red')
    ax.boxplot(error_seg_interval, positions=distance_label, widths=box_width, patch_artist=True, boxprops=dict(facecolor=colors[2]), sym = 'r.', flierprops=flierprops_set)
    ax.boxplot(error_interval, positions=distance_label+box_width, widths=box_width, patch_artist=True, boxprops=dict(facecolor=colors[0]), sym = 'r.', flierprops=flierprops_set)
    ax.set_xlabel('Distance[m]')
    ax.set_xticks(distance_label+box_width/2)
    ax.set_ylabel('Yaw Error[deg]')
    ax.set_ylim([0, 15])
    # set legend
    ax.plot([], c = colors[0], label='VINS', linestyle='-')
    ax.plot([], c = colors[2], label='VINS (Yolo-Based)', linestyle='-')
    ax.plot([], c='red', label='Fliers', marker='.', markersize=0.3, linestyle='None')
    ax.legend(loc='upper left', framealpha=0.5)
    print("Yaw error plot completed.")
    
def draw_RotationError(gt, path, path_seg, ax):
    """
    Draw the rotation error of the VINS.
    gt: list of ground truth poses, each pose is a list of [timestamp, x, y, z, qx, qy, qz, qw]
    path and path_seg format is the same as gt
    fig: matplotlib Box diagram
    """
    gt = np.array(gt)
    path = np.array(path)
    path_seg = np.array(path_seg)
    distance = np.zeros((len(gt)))
    for i in range(len(gt)-1):
        distance[i+1] = distance[i] + np.linalg.norm(gt[i+1,1:4] - gt[i,1:4])
    # Interpolation of the path and path_seg use np.interp
    path_interp = np.zeros((len(gt), 5))
    path_seg_interp = np.zeros((len(gt), 5))
    path_interp[:,0] = gt[:,0]
    path_seg_interp[:,0] = gt[:,0]
    for i in range(1,5):
        path_interp[:,i] = np.interp(gt[:,0], path[:,0], path[:,i])
        path_seg_interp[:,i] = np.interp(gt[:,0], path_seg[:,0], path_seg[:,i])
    # Calculate the rotation error
    error = np.zeros((len(gt)))
    error_seg = np.zeros((len(gt)))
    for i in range(len(gt)):
        q_gt = gt[i,4:8]
        q_path = path_interp[i,1:5]
        q_path_seg = path_seg_interp[i,1:5]
        R_gt = q2R(q_gt)
        R_path = q2R(q_path)
        R_path_seg = q2R(q_path_seg)
        # rotation error
        error[i] = np.arccos(0.5*(np.trace(np.dot(R_gt.T, R_path))-1))*180/np.pi 
        error_seg[i] = np.arccos(0.5*(np.trace(np.dot(R_gt.T, R_path_seg))-1))*180/np.pi
        error[i] = min(abs(error[i]), 180 - abs(error[i]))
        error_seg[i] = min(abs(error_seg[i]), 180 - abs(error_seg[i]))
    # Plot the rotation error as Box diagram, xlabel is the distance
    # divide the distance
    distance_all = min(distance[-1], calculate_distance)
    distance_label = np.arange(0, distance_all, distance_interval) + distance_interval
    # to int
    distance_label = distance_label.astype(int)
    error_interval = []
    error_seg_interval = []
    for i in range(len(distance_label)):    
        if i == 0:
            index = np.where(distance < distance_label[i])
        else:
            index = np.where((distance >= distance_label[i-1]) & (distance < distance_label[i]))
        error_interval.append(error[index])
        error_seg_interval.append(error_seg[index])
    flierprops_set = dict(marker='+', markersize=0.3, markerfacecolor='red')
    ax.boxplot(error_seg_interval, positions=distance_label, widths=box_width, patch_artist=True, boxprops=dict(facecolor=colors[2]), sym = 'r.', flierprops=flierprops_set)
    ax.boxplot(error_interval, positions=distance_label+box_width, widths=box_width, patch_artist=True, boxprops=dict(facecolor=colors[0]), sym = 'r.', flierprops=flierprops_set)
    ax.set_xlabel('Distance[m]')
    ax.set_xticks(distance_label+box_width/2)
    ax.set_ylabel('Rotation Error[deg]')
    # set legend at the left top
    ax.plot([], c = colors[0], label='VINS', linestyle='-')
    ax.plot([], c = colors[2], label='VINS (Yolo-Based)', linestyle='-')
    ax.plot([], c='red', label='Fliers', marker='.', markersize=0.3, linestyle='None')
    ax.legend(loc='upper left', framealpha=0.5)
    print("Rotation error plot completed.")
    
    
    
def read_csv(file):
    """
    Read csv file and return the content as a list of lists.
    format: [[timestamp, x, y, z, qx, qy, qz, qw], ...]
    """
    with open(file, 'r') as f:
        lines = f.readlines()
        content = []
        for line in lines:
            content.append([float(x) for x in line.split(',')])
    return content

def draw(gt_file, vins_aligned_file, vins_seg_aligned_file):
    """
    Draw the trajectory of the ground truth and the VINS.
    gt_file: ground truth file
    vins_aligned_file: VINS file
    vins_seg_aligned_file: VINS file (segmented)
    """
    gt = read_csv(gt_file)
    vins_aligned = read_csv(vins_aligned_file)
    vins_seg_aligned = read_csv(vins_seg_aligned_file)
    fig1 = plt.figure(figsize=(2, 2))
    ax = fig1.add_subplot(111)
    ax.set_position([0.15, 0.1, 0.8, 0.85])
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.tick_params(axis='both', pad=0.1)
    ax.set_aspect('equal', adjustable='datalim')
    draw_trajectory(gt, vins_aligned, vins_seg_aligned, ax)
    fig2 = plt.figure(figsize=(2, 2))
    ax1, ax2, ax3 = fig2.subplots(3, 1)
    ax1.set_position([0.15, 0.7, 0.8, 0.25])
    ax2.set_position([0.15, 0.4, 0.8, 0.25])
    ax3.set_position([0.15, 0.1, 0.8, 0.25])
    ax1.xaxis.labelpad = 0
    ax2.xaxis.labelpad = 0
    ax3.xaxis.labelpad = 0
    ax1.yaxis.labelpad = 0
    ax2.yaxis.labelpad = 0
    ax3.yaxis.labelpad = 0
    ax1.tick_params(axis='both', pad=0.1)
    ax2.tick_params(axis='both', pad=0.1)
    ax3.tick_params(axis='both', pad=0.1)
    draw_TranslationError(gt, vins_aligned, vins_seg_aligned, ax1)
    draw_YawError(gt, vins_aligned, vins_seg_aligned, ax2)
    draw_RotationError(gt, vins_aligned, vins_seg_aligned, ax3)
    # save the plot as pdf
    dir = os.path.dirname(os.path.realpath(__file__))
    fig1.savefig(dir + '/trajectory.pdf', bbox_inches='tight')
    fig2.savefig(dir + '/error.pdf', bbox_inches='tight')
    plt.show()
    
def draw_zed2i(gt_file, vins_aligned_file, vins_seg_aligned_file):
    """
    Draw the trajectory of the ground truth and the VINS.
    gt_file: ground truth file
    vins_aligned_file: VINS file
    vins_seg_aligned_file: VINS file (segmented)
    """
    global distance_interval, box_width
    box_width = 40
    distance_interval = 300
    gt = read_csv(gt_file)
    vins_aligned = read_csv(vins_aligned_file)
    vins_seg_aligned = read_csv(vins_seg_aligned_file)
    fig1 = plt.figure(figsize=(2, 3))
    ax1, ax2 = fig1.subplots(2, 1)
    # set ax1 and ax2 size
    ax1.xaxis.labelpad = 0
    ax2.xaxis.labelpad = 0
    ax1.yaxis.labelpad = 0
    ax2.yaxis.labelpad = 0
    ax1.tick_params(axis='both', pad=0.1)
    ax2.tick_params(axis='both', pad=0.1)
    ax1.set_position([0.15, 0.4, 0.8, 0.6])
    ax2.set_position([0.15, 0.1, 0.8, 0.25])
    ax1.set_aspect('equal', adjustable='datalim')
    draw_trajectory(gt, vins_aligned, vins_seg_aligned, ax1)
    draw_TranslationError(gt, vins_aligned, vins_seg_aligned, ax2)
    # save the plot as pdf
    dir = os.path.dirname(os.path.realpath(__file__))
    fig1.savefig(dir + '/zed2i_trajectory.pdf', bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath(__file__))
    gt_file = dir + '/gt.csv'
    vins_aligned_file = dir + '/vins_aligned.csv'
    vins_seg_aligned_file = dir + '/vins_seg_aligned.csv'
    zed2i_gt_aligned_file = dir + '/zed2i_gt_aligned.csv'
    zed2i_vins_aligned_file = dir + '/zed2i_vins_aligned.csv'
    zed2i_vins_seg_aligned_file = dir + '/zed2i_vins_seg_aligned.csv'
    # draw(gt_file, vins_aligned_file, vins_seg_aligned_file)
    draw_zed2i(zed2i_gt_aligned_file, zed2i_vins_aligned_file, zed2i_vins_seg_aligned_file)