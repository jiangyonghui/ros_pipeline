step 1: Set up Jetson TX2
1) Download JetPack from Nvidia 
2) Run JetPack*.run
3) Video tutorial: https://www.youtube.com/watch?v=D7lkth34rgM
4) Flash Jetson TX2 and install the downloaded libraries

--------------------------
step 2: Install ROS
1) Install ROS distribution(kinect) following org tutorial
2) Create catkin workspace and use catkin build tool as ROS package building system (just as a recommendation)

--------------------------
step 3: Install OpenCV
1) Refer to https://github.com/jetsonhacks/buildOpenCVTX2
2) Modify the build bash file according to your need
* I checked out v3.4.0 and modify cuda version to 9.0

--------------------------
step 4: Install OpenPose and Caffe as 3rd party lib(that is what i did)
1) Refer to https://github.com/jiangyonghui/openpose
2) Modify Makefile.config.Ubuntu16_cuda8_JetsonTX2 both in openpose/ubuntu and caffe
* Uncomment OPENCV_VERSION := 3, and change lib dir as you like

--------------------------
step 5: Install TensorFlow and Keras
1) TensorFlow installation refers to https://github.com/openzeka/Tensorflow-for-Jetson-TX2 

[issue] 
Description: can't uninstall enum34
Solution: maunally remove enum34 and its info file at /usr/lib/python2.7/dist-packages and run again

2) Keras installation refers to https://keras.io/#installation

---------------------------
step 6: Install Pytorch
1) refer to https://github.com/andrewadare/jetson-tx2-pytorch

[issue 1]
The following error appears if pytorch version is higher than v0.3.1
[Error] ‘BatchNorm2d’ object has no attribute ‘track_running_stats’
solution: install v0.3.1
