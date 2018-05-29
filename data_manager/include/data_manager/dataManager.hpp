#ifndef DATA_MANAGER_FUNC_HPP
#define DATA_MANAGER_FUNC_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <string>
#include <std_msgs/Float64MultiArray.h>
#include <Armadillo/armadillo>
#include <openpose/headers.hpp>


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


// pose keypoints interpolation
void poseInterpolator(arma::mat& mat);

// normalization
void normTensor(arma::mat& node_mat, const op::Point<int>& imageSize, const int id_neck, const int id_rhip);

// calc 3d tensor
void calcTensor(std::shared_ptr<arma::cube> sWindow);

// convert tensor to msg
void EigenTensorToMsg(std::shared_ptr<arma::cube> tensorPtr, std_msgs::Float64MultiArray& msg);

// write tensor to h5 file
//void WriteTenforRepo(const Eigen::Tensor<float, 3>& tensorRepo, std::vector<int>& tensor_shape, std::string& file_name);

// optional: plot actionness proposal --> TODO
// void PlotActionnessProposal(...)

#endif
