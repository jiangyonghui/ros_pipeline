#include "data_manager/dataManager.hpp"
#include "data_manager/spline.h"
#include <ros/console.h>
#include <std_msgs/Float32MultiArray.h>
#include <H5Cpp.h>


// pose keypoints interpolation
bool poseInterpolator(arma::mat& mat)
{
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

    return true;
}


// calc 3d tensor 
void calcTensor(std::shared_ptr<arma::cube> sWindow)
{
    // TODO

    return;
} 

void normTensor(std::shared_ptr<arma::cube> sWindow)
{
    // TODO

    return;
}


// convert eigen tensor to std_msgs::Int32MultiArray msg
void EigenTensorToMsg(const Eigen::Tensor<float, 3>& tensor, std_msgs::Float64MultiArray& msg)
{

	if (msg.layout.dim.size() != 3)
	{
		msg.layout.dim.resize(3);
	}

	msg.layout.dim[0].label = "frame";
	msg.layout.dim[0].size = tensor.dimension(0);

	msg.layout.dim[1].label = "node";
	msg.layout.dim[1].size = tensor.dimension(1);

	msg.layout.dim[2].label = "channel";
	msg.layout.dim[2].size = tensor.dimension(2);

	msg.layout.data_offset = 0;

	// transfer data to msg
	msg.data.resize(tensor.size());
	msg.data.clear();

	for(int i = 0; i < tensor.dimension(0); ++i)
		for(int j = 0; j < tensor.dimension(1); ++j)
			for(int k = 0; k < tensor.dimension(2); ++k)
				msg.data.push_back(tensor(i,j,k));

	return;
}


// resample action group to size of swindow_len
void ResampleActionGroup(std::vector<int>& action_group, const int swindow_len, const int swindow_str)
{
	std::vector<int> resampled_tensor;
	int resample_str = std::floor((action_group.size()+swindow_len-1)/swindow_len)*swindow_str;
	ROS_INFO("Resampling Stride: %d", resample_str);
	ROS_INFO("Resampling Tensor ID: ");

	for(auto tick = 0; tick < swindow_len; ++tick)
	{
		auto tensor_id = action_group.front() + tick*resample_str;
		std::cout << tensor_id << " ";
		resampled_tensor.push_back(tensor_id);
	}

	std::cout << std::endl;
	action_group.clear();
	action_group = resampled_tensor;

	return;
}


// write tensor to h5 file
void WriteTenforRepo(const Eigen::Tensor<double, 3>& tensorRepo, std::vector<int>& tensor_shape, std::string& file_name)
{
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

	return;
}
