#include "data_manager/openposeKpSub.hpp"
#include "data_manager/dataManager.hpp"
#include <ros/console.h>
#include <openpose/pose/poseParameters.hpp>

OpenposeKpSub::OpenposeKpSub(ros::NodeHandle& nh, const std::string& pose_topic, 
                             const std::vector<int>& node_seq, const op::PoseModel& pose_model):
                             nh_(nh), pose_topic_(pose_topic), node_seq_(node_seq), pose_model_(pose_model)
{
    ROS_INFO("Constructing Openpose Subscriber ...");
    bodypart_map_ = getBodyPartMapFromPoseModel();
    initializeSubscriber();
}


OpenposeKpSub::~OpenposeKpSub()
{
     ROS_INFO("Subscription terminated!");
}


const std::map<unsigned int, std::string>& OpenposeKpSub::getBodyPartMapFromPoseModel()
{
    if (pose_model_ == op::PoseModel::COCO_18) 
	{
  		ROS_INFO("Got Pose Model: COCO");
  		// different pose kp than OG Openpose
  		return data_manager::POSE_COCO_BODY_PARTS;
	} 
	else if (pose_model_ == op::PoseModel::MPI_15 || pose_model_ == op::PoseModel::MPI_15_4) 
	{
  		ROS_INFO("Got Pose Model: MPI");
  		return op::getPoseBodyPartMapping(pose_model_);
	} 
	else 
	{
   		ROS_FATAL("Invalid pose model, not map present");
  		exit(1);
	}
}


void OpenposeKpSub::initializeSubscriber()
{
    ROS_INFO("Initializing Openpose Subscriber ...");
    kp_sub_ = nh_.subscribe(pose_topic_, 1, &OpenposeKpSub::subscriberCallback, this);
    return;
}


void OpenposeKpSub::subscriberCallback(const message_repository::PersonDetectionConstPtr& kp_msg)
{
    // extract node keypoints from kp_msg and store them in node_kp_
	ROS_INFO("Calling node keypoints extraction ...");
	std::vector<float> node_;
	
	for (auto bodypart_idx : node_seq_) 
	{
		std::string body_part_string = bodypart_map_[bodypart_idx];

	  	if (body_part_string == "Nose") 
	  	{
			node_.push_back(kp_msg->nose.x);
			node_.push_back(kp_msg->nose.y);
	  	} 
	  	else if (body_part_string == "Neck") 
	  	{
			node_.push_back(kp_msg->neck.x);
			node_.push_back(kp_msg->neck.y);
	  	} 
	  	else if (body_part_string == "RShoulder") 
	  	{	
			node_.push_back(kp_msg->right_shoulder.x);
			node_.push_back(kp_msg->right_shoulder.y);
	  	} 
	  	else if (body_part_string == "RElbow") 
	  	{
			node_.push_back(kp_msg->right_elbow.x);
			node_.push_back(kp_msg->right_elbow.y);
	  	} 
	  	else if (body_part_string == "RWrist") 
	  	{	
			node_.push_back(kp_msg->right_wrist.x);
			node_.push_back(kp_msg->right_wrist.y);
	 	} 
	 	else if (body_part_string == "LShoulder") 
	 	{
			node_.push_back(kp_msg->left_shoulder.x);
			node_.push_back(kp_msg->left_shoulder.y);
	  	} 
	  	else if (body_part_string == "LElbow") 
	  	{	
			node_.push_back(kp_msg->left_elbow.x);
			node_.push_back(kp_msg->left_elbow.y);
	  	} 
	  	else if (body_part_string == "LWrist") 
	  	{			
			node_.push_back(kp_msg->left_wrist.x);
			node_.push_back(kp_msg->left_wrist.y);
	  	} 
	  	else if (body_part_string == "RHip") 
	  	{			
			node_.push_back(kp_msg->right_hip.x);
			node_.push_back(kp_msg->right_hip.y);
	  	} 
	  	else if (body_part_string == "RKnee") 
	  	{	
			node_.push_back(kp_msg->right_knee.x);
			node_.push_back(kp_msg->right_knee.y);
	  	} 
	  	else if (body_part_string == "RAnkle") 
	  	{			
			node_.push_back(kp_msg->right_ankle.x);
			node_.push_back(kp_msg->right_ankle.y);
	  	} 
	  	else if (body_part_string == "LHip") 
	  	{	
			node_.push_back(kp_msg->left_hip.x);
			node_.push_back(kp_msg->left_hip.y);
	  	} 
	  	else if (body_part_string == "LKnee") 
	  	{
			node_.push_back(kp_msg->left_knee.x);
			node_.push_back(kp_msg->left_knee.y);
	  	} 
	  	else if (body_part_string == "LAnkle") 
	  	{	
			node_.push_back(kp_msg->left_ankle.x);
			node_.push_back(kp_msg->left_ankle.y);
	  	} 
	  	else if (body_part_string == "REye") 
	  	{
			node_.push_back(kp_msg->right_eye.x);
			node_.push_back(kp_msg->right_eye.y);
	  	} 
	  	else if (body_part_string == "LEye") 
	  	{
			node_.push_back(kp_msg->left_eye.x);
			node_.push_back(kp_msg->left_eye.y);
	  	} 
	  	else if (body_part_string == "REar") 
	  	{	
			node_.push_back(kp_msg->right_ear.x);
			node_.push_back(kp_msg->right_ear.y);
	  	} 
	  	else if (body_part_string == "LEar") 
	  	{	
			node_.push_back(kp_msg->left_ear.x);
			node_.push_back(kp_msg->left_ear.y);
	  	} 
	  	else if (body_part_string == "Bkg") 
	  	{
			ROS_INFO("Background! No Effective Info!");
	  	}
	  	else
	  	{
			ROS_ERROR("Unknown bodypart %s, this should never happen!", body_part_string.c_str());
	  	}
	}

	nodeKp_ = node_; 
	
	ROS_INFO("Calling node keypoints successfully");
	return;
}


std::vector<float> OpenposeKpSub::getNodeKeypoints()
{
    return nodeKp_;
}


void OpenposeKpSub::resetNodeKeypoints()
{
    nodeKp_.clear();
    return;
}
















