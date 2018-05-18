#ifndef OPENPOSEROS_HPP
#define OPENPOSEROS_HPP

#include <openpose/core/array.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_repository/PersonDetection.h>


// bodypart detection
message_repository::BodypartDetection getBodyPartDetectionFromArrayAndIndex(const op::Array<float> &array, size_t idx);


// initialize bodypart
message_repository::BodypartDetection getNANBodypart();


// retrieve pose and publish keypoints
bool retrievePoseInfo(const op::Array<float>& poseKeypoints, ros::Publisher& keypoints_pub, std::map<unsigned int, std::string>& bodypartMap);



#endif
