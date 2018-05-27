/* This is the node for data management, e.g. data convertion, data flow,  management, data preprocess, etc. */
/*
 * subscribe:
 * /openpose_ros/detected_poses_keypoints
 *
 * servic_1: actionness_proposal
 * server: pose_net
 * client: action_proposal_client
 * request: sliding window
 * response: actioness label
 *
 * service_2: action_classifier
 * server: pose_net
 * client: action_classifier_client
 * request: action tensor
 * response: action label
*/

#include <openpose/headers.hpp>

#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/service_server.h>
#include <ros/init.h>
#include <ros/console.h> // ROS::DEBUG
#include <std_srvs/Empty.h>

// ros custom messages
#include <message_repository/PersonDetection.h>
#include <message_repository/ActionnessProposal.h>
#include <message_repository/ActionClassifier.h>

// data_manager functions
#include <data_manager/header.hpp>

#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h>  // google::InitGoogleLogging

//#include <boost/thread.hpp>
#include <thread>
#include <mutex>

// 3d tensor: cube
#include <Armadillo/armadillo>


DEFINE_string(pose_topic,				"/openpose_ros/detected_poses_keypoints", 	"Subscribe to pose topic that OpenPose publishes.");
DEFINE_string(model_pose, 				"COCO", 								  	"Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_int32(logging_level,				4, 											"The logging level. Integer in the range [0, 255]. 0 will output "
             																		"any log() message, while"
             																		"255 will not output any. Current OpenPose library messages are "
             																		"in the range 0-4: 1 for"
             																		"low priority messages and 4 for important ones.");
DEFINE_double(actioness_thres_value, 	2.0,										"Threshold value for action proposal");
DEFINE_double(action_grouping_rate, 	0.5, 										"Threshold rate for action grouping");
DEFINE_int32(tensor_repository_length, 	100, 										"Frames for action classfication");
DEFINE_int32(sliding_window_length, 	10, 										"Frames of sliding window");
DEFINE_int32(pose_tensor_stride, 		1, 											"Stride of pose tensor");
DEFINE_int32(sliding_window_stride, 	1, 											"Stride of sliding window");
DEFINE_int32(tensor_offset, 			0, 											"Starting frame for tensor calculation");
DEFINE_int32(sliding_window_offset, 	0, 											"Starting frame for sliding window calculation");
DEFINE_bool(output_tensor_repository, 	false, 										"Flag for tensor repository output");
DEFINE_string(h5_file_folder, 			"/home/nvidia/data/h5/", 					"Location of output h5 file");


typedef std::vector<std::vector<double>> nodeVec;

std::mutex mutex;

// node keypoints map
std::shared_ptr<nodeVec> nodeKpWritePtr = std::make_shared<nodeVec>();
std::shared_ptr<nodeVec> nodeKpReadPtr(nodeKpWritePtr);

// setting default parameter values
const auto swindow_str = FLAGS_sliding_window_stride;
const auto tensor_str = FLAGS_pose_tensor_stride;
const auto repo_len = FLAGS_tensor_repository_length;
const auto tensor_offset = FLAGS_tensor_offset;
const auto swindow_offset = FLAGS_sliding_window_offset;

const std::vector<int> node_seq{8, 9, 8, 16, 12, 17, 12, 16, 8, 14, 15, 10, 15,
							        14, 8, 4, 5, 0, 5, 4, 8, 6, 2, 7, 2, 6, 8};
const op::PoseModel poseModel = op::flagsToPoseModel(FLAGS_model_pose);



// action proposal and classfication
void nodeKpSubscriber()
{
	// logging
	op::log(" ");
	op::log("Data Manager -- Action Proposal and Classification Client");
	op::log("---------------------------------------------------------");

	// declare action classfication service
	ros::NodeHandle nh;
	ros::ServiceClient action_classifier_client =
		nh.serviceClient<message_repository::ActionClassifier>("action_classifier");
	message_repository::ActionClassifier action_classifier_srv;

	// Initialize OpenposeKpSub
	OpenposeKpSub openposeKpSubscriber(nh, static_cast<const std::string>(FLAGS_pose_topic), node_seq, poseModel);

	int frame_id = -1;
	int tensor_id = -1;

	// get node extract callback ready
	ros::spinOnce();

	// processing frame by frame
	while (ros::ok)
	{
		// TODO: [Issue] if some log goes on here, it turns out weird

		// get body nodes from openpose
		std::vector<double> nodeKeypoints(openposeKpSubscriber.getNodeKeypoints());
		openposeKpSubscriber.resetNodeKeypoints();

		// check if the node keypoints are correctly delivered
		if (!nodeKeypoints.empty())
		{
			++frame_id;
			if ((frame_id >= tensor_offset) && !((frame_id - tensor_offset) % tensor_str))
			{
				// set block mutex to secure data writing to shared map
				std::lock_guard<std::mutex> lock(mutex);

				++tensor_id;
				// add tensor to repo
				ROS_INFO("node keypoints size: %lu", nodeKeypoints.size());
				nodeKpWritePtr->push_back(nodeKeypoints);
				//AddTensor(tensorRepo, nodeKeypoints, tensor_id);
				ROS_INFO_STREAM("Current Tensor Repository Size: " << (tensor_id+1));



//				if ((tensor_id+1) >= swindow_len)
//				{
//					++swindow_id;

//					if ((swindow_id >= swindow_offset) && !((swindow_id - swindow_offset) % swindow_str))
//					{
//						// feed sliding window into action proposal node
//						op::log(" ");
//						op::log("Actionness Proposal");
//						op::log("---------------");
//						std_msgs::Float32MultiArray tensor_msg;
// 						Eigen::Tensor<float, 3> sWindow = GetProposalTensor(tensorRepo, tensor_id, swindow_len);
//						EigenTensorToMsg(sWindow, tensor_msg);
//						actionness_proposal_srv.request.tensor = tensor_msg;

//						// call action proposal service
//						if (actionness_proposal_client.call(actionness_proposal_srv))
//						{
//							ROS_INFO("Calling Action Proposal Service ... ");

//						  	ActionessHistogram histAction;

//						  	// histAction.frame_id is the starting tensor_id of the sliding window
//						  	histAction.frame_id = tensor_id - (swindow_len-1) * swindow_str;
//						  	histAction.label = (actionness_proposal_srv.response.label)%2;
//						  	actionness_hist.push_back(histAction);

//						  	ROS_INFO_STREAM("Sliding Window Head Tensor ID: "
//										  	<< histAction.frame_id
//										  	<< ", Actioness Class: "
//										  	<< histAction.label);

//						  	// online action grouping
//						  	if (!histAction.label)
//								++num_zero;
//						  	else
//								++num_one;
//						  	actionness_rate = float(num_one) / float(num_one + num_zero);
//						  	ROS_INFO("Current Actioness Rate: %f", actionness_rate);

//						  	if (actionness_rate >= (float)thres_rate)
//						  	{
//								action_group.push_back(histAction.frame_id);
//						  	}
//						  	else
//						  	{
//						  		if (!action_group.empty())
//						  		{
//								  	int head_id = action_group.front() * tensor_str + tensor_offset;
//								  	int end_id = (action_group.back() + swindow_len - 1) + tensor_str + tensor_offset;
//								  	ROS_INFO("Action detected between frame no.%d and no.%d; Action size: %lu!",
//									  		head_id, end_id, (action_group.size() + swindow_len - 1));

//								  	op::log(" ");
//								  	op::log("Action Classification");
//								  	op::log("---------------------");

//								  	// store the action group info (head_id, group_size) for writing tensor to h5 file
//								  	action_group_hist.push_back(std::make_pair(action_group.front(), action_group.size()));

//								  	// resampling action tensor
//								  	ROS_INFO("Starting action tensor resampling ...");
//								  	ResampleActionGroup(action_group, swindow_len, swindow_str);
//								  	ROS_INFO("Action tensor resampling done!");

//								  	ROS_INFO("Retrieving action tensor ...");
//								  	Eigen::Tensor<float, 3> action_tensor(swindow_shape);
//								  	action_tensor = GetActionTensor(tensorRepo, action_group);
//								  	ROS_INFO("Action tensor retrieved!");

//								  	std_msgs::Float32MultiArray tensor_msg;
//								  	EigenTensorToMsg(action_tensor, tensor_msg);
//								  	action_classifier_srv.request.tensor = tensor_msg;

//								  	if (action_classifier_client.call(action_classifier_srv))
//								  	{
//										ROS_INFO("Calling Action Classification Service ...");

//										auto action_label = action_classifier_srv.response.label;

//										ROS_INFO("The action label is: %d", action_label);
//										ROS_INFO("Calling Action Classification Service Successfully!");
//								  	}
//								  	else
//								  	{
//										ROS_INFO("Calling Action Classification Service Failed!");
//								  	}

//								  	num_zero = 0;
//								  	num_one = 0;
//								  	action_group.clear();
//								}
//								else
//								{
//									ROS_INFO("No action detected so far!");
//								}
//					  		}

//					 		ROS_INFO("Calling Action Proposal Service Successfully!");
//						}
//						else
//						{
//					  		ROS_ERROR("Calling Action Proposal Service Failed!");
//						}
//				  	}
//				  	else
//				  	{
//				  		ROS_INFO("No action proposal! Skipping tensor id: %d", tensor_id);
//				  	} // end action proposal and classfication
//				}
//			  	else
//				{
//					ROS_INFO("No action proposal!");
//					ROS_INFO("Tensor repository size is less than slinding window size, waiting %d more tensors.",
//							(swindow_len - (tensor_id+1)));
//				} // end checking tensor repo size

			  	// when tensor repository is full, handle tensor repo accordingly
//			  	if ((tensor_id+1) == repo_len)
//			  	{
//					ROS_INFO("Tensor repository is full!");

//					// check if any action has been detected ->TODO: further optimization is expected
//					if (action_group_hist.empty() && !action_group.empty())
//					{
//				  		ROS_INFO("Action detected for the entire tensor repository!");
//				  		op::log(" ");
//				 		op::log("Action Classification");
//				  		op::log("---------------------");

//				  		ROS_INFO("Resampling action tensor ...");

//				  		// resampling action tensor
//					  	ROS_INFO("Starting action tensor resampling ...");
//					  	ResampleActionGroup(action_group, swindow_len, swindow_str);
//					  	ROS_INFO("Action tensor resampling done!");

//					  	ROS_INFO("Retrieving action tensor ...");
//					  	Eigen::Tensor<float, 3> action_tensor(swindow_shape);
//					  	action_tensor = GetActionTensor(tensorRepo, action_group);
//					  	ROS_INFO("Action tensor retrieved!");

//					  	std_msgs::Float32MultiArray tensor_msg;
//						EigenTensorToMsg(action_tensor, tensor_msg);
//						action_classifier_srv.request.tensor = tensor_msg;

//				  		if (action_classifier_client.call(action_classifier_srv))
//				  		{
//							ROS_INFO("Calling Action Classification Service ...");

//							auto action_label = action_classifier_srv.response.label;
//							ROS_INFO_STREAM("The action label is: " << action_label);
//							ROS_INFO("Calling Action Classification Service Succeeded!");
//				  		}
//				  		else
//				  		{
//							ROS_INFO("Calling Action Classification Service Failed!");
//				  		}
//					}
//					else
//					{
//				  		ROS_INFO("No action detected in the entire tensor repository!");
//					}

//					// option for outputing tensor repository for offline processing
//					// TODO: output tensor every 5/10 frames
////					if (FLAGS_output_tensor_repository)
////					{
////				  		ROS_INFO("Ready for outputing tensor repository");
////				  		// write to file
////				  		std::vector<int> tensor_shape{repo_len, node_seq_len * 2, 3};
////				  		std::string file_name = FLAGS_h5_file_folder + "test_" + static_cast<std::string>(file_id) + ".h5";
////				  		WriteTenforRepo(tensorRepo, tensor_shape, file_name);
////				  		++file_id;
////					}

//					// release the repo block, processing the next block of frames
//					ROS_INFO("Releasing tensor repository, sliding window and actioness proposal ...");
//					tensor_id = 0;
//					swindow_id = 0;
//					num_zero = 0;
//					num_one = 0;
//					action_group.clear();
//					action_group_hist.clear();
//					actionness_hist.clear();
//					tensorRepo.setZero();
//					ROS_INFO("All Reset done!");
//					ROS_INFO("Waiting for next action detection and recognition ...");
//					op::log(" ");
//			  	}
//			  	else
//			  	{
//					ROS_INFO("There are still %d places in tensor repository", repo_len - (tensor_id+1));
//					op::log(" ");
//			  	} // end checking repo size and update tensor id
//			}
			else
			{
				ROS_INFO("Skipping frame id: %d ", frame_id);
			} // end adding tensor to repo, online action proposal and classfication
		}

		ros::spinOnce();
	} // end adding node keypoints to repo

	return;
}


// pose tensor preparation and do action classification
void actionClassifier()
{
    auto tensor_id = 0;

    // swindow shape
    const auto n_rows = FLAGS_sliding_window_length;
    const auto n_cols = node_seq.size();
    const auto n_slices = 3;

    // declare sliding window
    std::shared_ptr<arma::cube> sWindow(new arma::cube(n_rows, n_cols, n_slices, arma::fill::zeros));
    arma::mat& node_mat = sWindow->slice(0);

	ROS_INFO_STREAM("Initialize sliding window with size: [" << n_rows "x" << n_cols << "x" << n_slices << "]");

	// do the job
    while(ros::ok)
    {
        // make sure there is enough node data in the repository
        if (nodeKpReadPtr->size() >= tensor_id+1)
        {
            // for frame 0-8
            if ((tensor_id+1) < n_rows)
            {
                std::lock_guard<std::mutex> lock(mutex);
                ROS_INFO_STREAM("Retrieving node keypoints from frame nr." << tensor_id);
                node_mat.row(tensor_id) = arma::rowvec(nodeKpReadPtr->at(tensor_id));
                ++tensor_id;
                break;
            }
            // for frame >= 9
            else
            {
                // update frame mat: shed the first row, append new coming frame
                if ((tensor_id+1) > n_rows)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    node_mat.shed_row(0);
                    mat.insert_rows(n_rows-1, arma::rowvec(nodeKpReadPtr->at(tensor_id)));
                }

                // do smoothing
                poseInterpolator(node_mat);

                // calc 3d tensor
                calcTensor(sWindow);

                // normalization
                normTensor(sWindow);
                
                // feed to classifier
                // log feedback



                ++tensor_id;
            }

        }
        // empty node repo
        else if (nodeKpReadPtr->empty())
        {
            ROS_INFO("Empty node keypoints repository, waiting for data ...");
            break;
        }
        // not updated repo
        else
        {
            ROS_INFO("No more new added data, waiting for data ...");
            break;
        }
    }

    return;
}



int main(int argc, char **argv)
{
	google::InitGoogleLogging("data manager");
  	gflags::ParseCommandLineFlags(&argc, &argv, true);
  	ros::init(argc, argv, "data_manager");

  	std::thread t_nodeKpSub(nodeKpSubscriber());
  	std::thread t_actionClassifier(actionClassifier());

  	t_nodeKpSub.join();
  	t_actionClassifier.join();

  	return 0;
}
