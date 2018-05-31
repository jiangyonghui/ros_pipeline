cmake_minimum_required(VERSION 2.8.3)
project(openpose_ros)

add_compile_options(-std=c++11)
add_definitions(-DUSE_CAFFE)

## Set include dirs
set(OPENPOSE_INCLUDE_DIRS 
	/home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/include)
	
set(CAFFE_INCLUDE_DIRS
    /home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/3rdparty/caffe/distribute/include)

set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)

# for gflag, glog ...
set(OTHER_INCLUDE_DIRS /usr/include)

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
	image_transport
)

## catkin specific configuration 
catkin_package(
  CATKIN_DEPENDS 
  std_msgs 
  roscpp
  cv_bridge
  message_repository  	
  roslib
  image_transport)			 

find_package(OpenCV COMPONENTS highgui imgcodecs)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OPENPOSE_INCLUDE_DIRS}
  ${CAFFE_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

find_library(OPENPOSE_LIBRARY openpose /home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/build/lib)
find_library(CAFFE_LIBRARY caffe /home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/3rdparty/caffe/distribute/lib)
find_library(CUDA_LIBRARY cuda /usr/local/cuda/lib64)
find_library(GFLAGS_LIBRARY gflags /usr/lib/aarch64-linux-gnu)
find_library(GLOG_LIBRARY glog /usr/lib/aarch64-linux-gnu)
find_library(PTHREAD_LIBRARY pthread /usr/lib/aarch64-linux-gnu)

set(OPENPOSE_ROS_SRC_LIST
    src/openposeROS.cpp
    src/rosImageSubscriber.cpp)


add_executable(openpose_ros src/openpose_ros.cpp ${OPENPOSE_ROS_SRC_LIST})
target_link_libraries(openpose_ros 	${catkin_LIBRARIES} 
				   					${OPENPOSE_LIBRARY} 
				   					${CAFFE_LIBRARY} 
				   					${CUDA_LIBRARY} 
				   					${GFLAGS_LIBRARY} 
				   					${GLOG_LIBRARY}
				   					${OpenCV_LIBRARIES})
add_dependencies(openpose_ros ${catkin_EXPORTED_TARGETS})

add_executable(openpose src/openpose.cpp)
target_link_libraries(openpose   	${catkin_LIBRARIES} 
				   					${OPENPOSE_LIBRARY} 
				   					${CAFFE_LIBRARY} 
				   					${CUDA_LIBRARY} 
				   					${GFLAGS_LIBRARY} 
				   					${GLOG_LIBRARY}
				   					${OpenCV_LIBRARIES})

add_executable(image_seq_reader src/imageSeqReader.cpp)
target_link_libraries(image_seq_reader	${catkin_LIBRARIES}   
				   						${GFLAGS_LIBRARY} 
				   						${GLOG_LIBRARY}
				   						${OpenCV_LIBRARIES}
				   						${PTHREAD_LIBRARY})
add_dependencies(image_seq_reader	${catkin_EXPORTED_TARGETS})





