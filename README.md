# YVINS-Fusion

## An Fast YOLO-based Visual Pipeline for [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)

## 1. Prerequisites

### 1.1 **Ubuntu** and **ROS**

Ubuntu 64-bit 20.04
ROS Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**

Follow [Ceres Installation](http://ceres-solver.org/installation.html).

## 2. Build YVINS-Fusion

Clone the repository and catkin_make:

```bash
cd ~/catkin_ws/src
git clone https://github.com/nlc2002/YVINS-Fusion.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)

### 3.1 Campus dataset 

[download](https://cloud.tsinghua.edu.cn/f/c93294680f2544b0a094/?dl=1)

```bash
roslaunch vins vins_zed2i_mono.launch 
rosbag play campus.bag
```

### 3.2 [KAIST](https://sites.google.com/view/complex-urban-dataset)

*/datas/global_pose.py* transfer the global pose.csv to rosbag. Then you can merge the global_pose.bag with needed bag files.

```bash
roslaunch vins vins_kaist39_mono.launch 
rosbag play kaist.bag
```

### 6.1 Configuration file

Write a config file for your device. You can take config files of KAIST as the example. 

## 8. Acknowledgements

We use [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/).

We use [Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/v.1.1.0) as Yolo detector.

## 9. License

The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
