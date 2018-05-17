step 1: Set up Jetson TX2
1) Download JetPack from Nvidia 
2) Run JetPack*.run
3) Video tutorial: https://www.youtube.com/watch?v=D7lkth34rgM
4) Flash Jetson TX2 and install the downloaded libraries

[issue 1] 
Description: After flashing, the installation is stuck at "Determinating Target IP Address"
Why: Host and Target are not connected through the same Ethernet
Solution: on Host run command: $ ifconfig, checkout the port with the internet ip of 10.34.219.x(Bosch Internet)
using this port as ssh connection option during JetPack installation

[issue 2]
Description: cuda toolkit and some other libraries are not installed on Jetson TX2, which should be done by JetPack
Solution: first install cuda dirver and then install cuda toolkit and other libraries
exsample commands: 
$ sudo apt-get install cuda-driver-dev-9-0  
$ sudo apt-get install cuda-toolkit-9-0
$ sudo apt-get install lib*balbla* 
       
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
