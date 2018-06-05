#include "data_manager/dataManager.hpp"
#include "data_manager/spline.h"
#include <ros/console.h>
#include <std_msgs/Float32MultiArray.h>
#include <H5Cpp.h>
#include <algorithm>
#include <vector>
#include <cmath>


// pose keypoints interpolation
void poseInterpolator(arma::mat& mat)
{
    ROS_INFO("Starting Pose Interpolation ...");
    
    const auto n_rows = mat.n_rows;
    const auto n_cols = mat.n_cols;
    std::vector<double> t;
    std::vector<double> val;

    for (auto col = 0; col < n_cols; ++col)
    {
        tk::spline s;
        t.clear();
        val.clear();

        arma::uvec t_spline = arma::find_finite(mat.col(col));
        arma::vec val_spline = static_cast<arma::vec>(mat.col(col)).elem(t_spline);

        for (arma::uvec::iterator it = t_spline.begin(); it != t_spline.end(); ++it)
        {
            t.push_back(*it);
        }

        for (arma::vec::iterator it = val_spline.begin(); it != val_spline.end(); ++it)
        {
            val.push_back(*it);
        }

        s.set_points(t, val);

        for (auto row = 0; row < n_rows; ++row)
        {
            static_cast<arma::vec>(mat.col(col)).at(row) = s(row);
        }
    }
    
    ROS_INFO("Pose Interpolation Done!");
    
    return;
}


// normalization
void normTensor(arma::mat& node_mat, const op::Point<int>& imageSize, const int id_neck, const int id_rhip)
{
    ROS_INFO("Starting Pose Normalization ...");
    
    // step 1: normalized with respect to image coordinate system
    const auto width = imageSize.x;
    const auto height = imageSize.y;
    const auto n_cols = node_mat.n_cols;
    const auto n_rows = node_mat.n_rows;
        
    for (auto col = 0; col < n_cols; ++col)
    {
        if (!(col%2))
        {
            node_mat.col(col) = 2*node_mat.col(col)/width-1;
        }
        else
        {
            node_mat.col(col) = 2*node_mat.col(col)/height-1;
        }
    }
    
    // step 2: normalized with respect to torso length  
    for (auto row = 0; row < n_rows; ++row)
    {
        auto& x_neck = node_mat.at(row, 2*id_neck);
        auto& y_neck = node_mat.at(row, 2*id_neck+1);
        auto& x_rhip = node_mat.at(row, 2*id_rhip);
        auto& y_rhip = node_mat.at(row, 2*id_rhip+1); 
        const auto torso_len = std::sqrt(std::pow(x_neck-x_rhip,2) + std::pow(y_neck-y_rhip,2));
        node_mat.row(row) /= torso_len;
        const auto torso_x = (x_neck+x_rhip)/2;
        const auto torso_y = (y_neck+y_rhip)/2;
        
        for (auto col = 0; col < n_cols; ++col)
        {
            if (!(col%2))
            {
                node_mat.at(row, col) -= torso_x;
            }
            else
            {
                node_mat.at(row, col) -= torso_y;
            }
        }
    }
    
    ROS_INFO("Pose Normalization Done!");
    
    return;
}


// calc 3d tensor
void calcTensor(std::shared_ptr<arma::cube> sWindow)
{
    ROS_INFO("Starting Tensor Calculation ...");
    
    const auto n_rows = sWindow->n_rows;
    const auto n_slices = sWindow->n_slices;

    for (auto slice_id = 1; slice_id < n_slices; ++slice_id)
    {
        for (auto row_id = 1; row_id < n_rows; ++row_id)
        {
            sWindow->slice(slice_id).row(row_id) =
            sWindow->slice(slice_id-1).row(row_id) -
            sWindow->slice(slice_id-1).row(row_id-1);
        }
    }
    
    ROS_INFO("Tensor Calculation Done!");
    
    return;
}


// convert tensor to msg
void EigenTensorToMsg(std::shared_ptr<arma::cube> tensorPtr, std_msgs::Float64MultiArray& msg)
{
	ROS_INFO("Converting Tensor to ROS Message ...");
	
	if (msg.layout.dim.size() != 3)
	{
		msg.layout.dim.resize(3);
	}
    
    const auto n_frames = tensorPtr->n_rows;
    const auto n_nodes = tensorPtr->n_cols;
    const auto n_channels = tensorPtr->n_slices;
    const auto tensor_size = tensorPtr->n_elem;
    
	msg.layout.dim[0].label = "frame";
	msg.layout.dim[0].size = n_frames;

	msg.layout.dim[1].label = "node";
	msg.layout.dim[1].size = n_nodes;

	msg.layout.dim[2].label = "channel";
	msg.layout.dim[2].size = n_channels;

	msg.layout.data_offset = 0;

	// transfer data to msg
	msg.data.resize(tensor_size);
	msg.data.clear();

	for(auto frame = 0; frame < n_frames; ++frame)
		for(auto node = 0; node < n_nodes; ++node)
			for(auto channel = 0; channel < n_channels; ++channel)
				msg.data.push_back(tensorPtr->at(frame,node,channel));
    
    ROS_INFO("ROS Message Conversion Done!");
    
	return;
}


// write tensor to h5 file
//void WriteTenforRepo(const Eigen::Tensor<double, 3>& tensorRepo, std::vector<int>& tensor_shape, std::string& file_name)
//{
//	if(!tensorRepo.repo.empty())
//	{
//		/*t_4d(dim_0, dim_1, dim_2, dim_3)
//		* dim_0: number of frames
//		* dim_1: number of node keypoints, e.g. 17x2
//		* dim_2: rank, (x,v,a)
//		*/
//		auto dim_0 = tensor_shape.at(0);
//		auto dim_1 = tensor_shape.at(1);
//		auto dim_2 = tensor_shape.at(2);
//
//		int tensor_3d[dim_0][dim_1][dim_2];
//		ROS_INFO("Tensor Shape: (%d, %d, %d)", dim_0, dim_1, dim_2);
//
//		/*--- get 3d tensor elements ---*/
//		for(auto frame = 0; frame < dim_0; ++frame)
//		{
//			//ROS_INFO("Enter Frame %d:", frame);
//			for(auto tensor = 0; tensor < dim_2; ++tensor)
//			{
//				//ROS_INFO("Enter Tensor %d:", tensor);
//				for(auto point = 0; point < dim_1; point += 2)
//				{
//					auto node = point/2;
//					//ROS_INFO("Enter Node %d:", node);

//					auto x = tensorRepo.repo.at(frame).tensor.at(tensor).node.at(node).x;
//					auto y = tensorRepo.repo.at(frame).tensor.at(tensor).node.at(node).y;
//					tensor_3d[frame][point][tensor] = x;
//					tensor_3d[frame][point+1][tensor] = y;
//					//ROS_INFO("Point: (%d, %d) has been pushed back to buffer", x, y);
//				}
//			}
//		}
//
//
//		ROS_INFO("Tensors have been written to buffer");
//
//		/*--- write data to h5 file ---*/
//		const H5std_string FILE_NAME(file_name);
//		const H5std_string DATASET_NAME("Pose Tensor Repository");
//		const int RANK = 3;
//
//		try
//		{
//			/*
//			* Turn off the auto-printing when failure occurs so that we can
//			* handle the errors appropriately
//			*/
//			H5::Exception::dontPrint();
//
//			// Create a new file using H5F_ACC_TRUNC access
//			H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
//
//			// Define the size of the array and create the data space for fixed size dataset
//			hsize_t dimsf[3];
//			dimsf[0] = dim_0;
//			dimsf[1] = dim_1;
//			dimsf[2] = dim_2;
//
//			H5::DataSpace dataspace(RANK, dimsf);
//
//			// Define datatype for the data in the file
//			H5::IntType datatype(H5::PredType::NATIVE_INT);
//			datatype.setOrder(H5T_ORDER_LE);
//
//			/*
//			* Create a new dataset within the file using defined dataspace and
//			* datatype and default dataset creation properties.
//			*/
//			H5::DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
//
//			// Write the data to the dataset
//			dataset.write(tensor_3d, H5::PredType::NATIVE_INT);
//			ROS_INFO("Tensor outputing succeeded!");
//		}
//
//		// catch failure caused by the H5File operations
//		catch(H5::FileIException error)
//		{
//		  error.printError();
//		  return -1;
//		}
//		// catch failure caused by the DataSet operations
//		catch(H5::DataSetIException error)
//		{
//		  error.printError();
//		  return -1;
//		}
//		// catch failure caused by the DataSpace operations
//		catch(H5::DataSpaceIException error)
//		{
//		  error.printError();
//		  return -1;
//		}
//		// catch failure caused by the DataType operations
//		catch(H5::DataTypeIException error)
//		{
//		  error.printError();
//		  return -1;
//		}
//	}
//	else
//	{
//		ROS_ERROR("The Tensor Repository is empty, outputing failed!");
//		return -1;
//	}

//	return;
//}
