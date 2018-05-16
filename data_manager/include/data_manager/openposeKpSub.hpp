#ifndef OPENPOSEKPSUB_HPP
#define OPENPOSEKPSUB_HPP

#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <vector>
#include <map>
#include <string>
#include <openpose/pose/enumClasses.hpp>
#include <message_repository/PersonDetection.h>

// Subscribe openpose keypoints
class OpenposeKpSub
{
private:
    ros::NodeHandle nh_;
 	ros::Subscriber kp_sub_;
 	const std::string pose_topic_;
  	const std::vector<int> node_seq_;
  	const op::PoseModel pose_model_;
  	std::vector<float> nodeKp_;
    std::map<unsigned int, std::string> bodypart_map_;

public:
    OpenposeKpSub(ros::NodeHandle& nh, const std::string &pose_topic, 
                  const std::vector<int> &node_seq, const op::PoseModel& pose_model);
         
    ~OpenposeKpSub();
    const std::map<unsigned int, std::string>& getBodyPartMapFromPoseModel();
    void initializeSubscriber();
    void subscriberCallback(const message_repository::PersonDetectionConstPtr& kp_msg);
    std::vector<float> getNodeKeypoints();
    void resetNodeKeypoints();   
};

#endif












