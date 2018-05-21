#ifndef DATA_MANAGER_FUNC_HPP
#define DATA_MANAGER_FUNC_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <string>
#include <std_msgs/Float32MultiArray.h>


// define bodypart nodes
namespace data_manager
{
	const std::map<unsigned int, std::string> POSE_COCO_BODY_PARTS
	{
		{0,  "LAnkle"},
        {1,  "LEar"},
        {2,  "LElbow"},
        {3,  "LEye"},
        {4,  "LHip"},
        {5,  "LKnee"},
        {6,  "LShoulder"},
        {7,  "LWrist"},
        {8,  "Neck"},
        {9,  "Nose"},
        {10, "RAnkle"},
        {11, "REar"},
        {12, "RElbow"},
        {13, "REye"},
        {14, "RHip"},
        {15, "RKnee"},
        {16, "RShoulder"},
        {17, "RWrist"},
        {18, "Bkg"}
	};
}

// define some data structures here
struct ActionessHistogram
{
	int frame_id;
	int label;
};


// add tensor to repository
void AddTensor(Eigen::Tensor<float, 3>& repo, std::vector<float>& node_keypoints, const int& index);

// pose keypoints interpolation
bool poseInterpolator(Eigen::Tensor<float, 3>& repo);

// get proposal tensor
Eigen::Tensor<float, 3> GetProposalTensor(const Eigen::Tensor<float, 3>& repo, const int& tensor_id, const int& swindow_len);

// retrieve grouped action tensor
Eigen::Tensor<float, 3> GetActionTensor(const Eigen::Tensor<float, 3>& tensorRepo, std::vector<int>& action_group);

// convert eigen tensor to std_msgs::Float32MultiArray msg
void EigenTensorToMsg(const Eigen::Tensor<float, 3>& tensor, std_msgs::Float32MultiArray& msg);

// resample action group to size of swindow_len
void ResampleActionGroup(std::vector<int>& action_group, const int& swindow_len, const int& swindow_str);

// write tensor to h5 file
void WriteTenforRepo(const Eigen::Tensor<float, 3>& tensorRepo, std::vector<int>& tensor_shape, std::string& file_name);

// optional: plot actionness proposal --> TODO
// void PlotActionnessProposal(...)

#endif
