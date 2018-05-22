cmake_minimum_required(VERSION 2.8.3)
project(data_manager)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Set include dirs
set(OPENPOSE_INCLUDE_DIRS 
	/home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/include)
	
set(CAFFE_INCLUDE_DIRS
    /home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/3rdparty/caffe/distribute/include)

set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)

# for gflag, glog, eigen3 ...
set(OTHER_INCLUDE_DIRS /usr/include
                       /usr/include/eigen3)
	

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
   	std_msgs
	sensor_msgs
	roscpp
	cv_bridge
	message_repository
	roslib
	image_transport)

find_package(HDF5 COMPONENTS CXX HL)
find_package(Boost COMPONENTS iostreams filesystem system thread)

catkin_package(
  CATKIN_DEPENDS 
  std_msgs 
  roscpp
  cv_bridge
  message_repository  	
  roslib
  image_transport
)

## add all the lib include dirs into include_directories
include_directories(
	include
  	${catkin_INCLUDE_DIRS}
  	${OPENPOSE_INCLUDE_DIRS}
  	${CAFFE_INCLUDE_DIRS}
  	${CUDA_INCLUDE_DIRS}
  	${OTHER_INCLUDE_DIRS}
  	${HDF5_INCLUDE_DIRS}
  	${Boost_INCLUDE_DIRS})

## find the corresponding libs 
find_library(OPENPOSE_LIBRARY openpose /home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/build/lib)
find_library(CAFFE_LIBRARY caffe /home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/3rdparty/caffe/distribute/lib)
find_library(CUDA_LIBRARY cuda /usr/local/cuda/lib64)
find_library(GFLAGS_LIBRARY gflags /usr/lib/aarch64-linux-gnu)
find_library(GLOG_LIBRARY glog /usr/lib/aarch64-linux-gnu)


add_executable(data_manager src/data_manager.cpp src/dataManager.cpp src/openposeKpSub.cpp)
target_link_libraries(data_manager	${catkin_LIBRARIES} 
				   					${OPENPOSE_LIBRARY} 
				   					${CAFFE_LIBRARY} 
				   					${CUDA_LIBRARY} 
				   					${GFLAGS_LIBRARY} 
				   					${GLOG_LIBRARY}
				   					${HDF5_LIBRARIES}
				   					${Boost_LIBRARIES})
add_dependencies(data_manager  ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS scripts/action_classifier
					  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})




