/* This is the node for data management, e.g. data convertion, data flow,  management, data preprocess, etc. */
/*
 * subscribe to:
 * /openpose_ros/detected_poses_keypoints
 *
 * service: action_classifier
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
#include <ros/console.h>
#include <std_srvs/Empty.h>

// custom ros msgs
#include <message_repository/PersonDetection.h>
#include <message_repository/ActionnessProposal.h>
#include <message_repository/ActionClassifier.h>

// custom funcs 
#include <data_manager/header.hpp>

#include <gflags/gflags.h> // DEFINE_*
#include <glog/logging.h>  // google::InitGoogleLogging

#include <thread> 
#include <mutex> 
#include <chrono>

#include <Armadillo/armadillo> // arma::cube, arma::mat, arma::vec


DEFINE_string(pose_topic,				"/openpose_ros/detected_poses_keypoints", 	"Subscribe to pose topic that OpenPose publishes.");
DEFINE_string(model_pose, 				"COCO", 								  	"Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_int32(logging_level,				4, 											"The logging level. Integer in the range [0, 255]. 0 will output "
             																		"any log() message, while"
             																		"255 will not output any. Current OpenPose library messages are "
             																		"in the range 0-4: 1 for"
             																		"low priority messages and 4 for important ones.");
DEFINE_double(actioness_thres_value, 	2.0,										"Threshold value for action proposal");
DEFINE_double(action_grouping_rate, 	0.5, 										"Threshold rate for action grouping");
DEFINE_int32(tensor_repo_len,        	100, 										"Frames for action classfication");
DEFINE_int32(sliding_window_length, 	10, 										"Frames of sliding window");
DEFINE_int32(tensor_stride, 		    1, 											"Stride of pose tensor");
DEFINE_int32(sliding_window_stride, 	1, 											"Stride of sliding window");
DEFINE_int32(tensor_offset, 			0, 											"Starting frame for tensor calculation");
DEFINE_int32(sliding_window_offset, 	0, 											"Starting frame for sliding window calculation");
DEFINE_bool(output_tensor_repository, 	false, 										"Flag for tensor repository output");
DEFINE_string(h5_file_folder, 			"/home/nvidia/data/h5/", 					"Location of output h5 file");


// setting default parameter values
const std::vector<int> nodeSeq{8, 9, 8, 16, 12, 17, 12, 16, 8, 14, 15, 10, 15,
							        14, 8, 4, 5, 0, 5, 4, 8, 6, 2, 7, 2, 6, 8};

// node keypoints map
std::shared_ptr<arma::mat> nodeKpWritePtr(new arma::mat);
std::shared_ptr<arma::mat> nodeKpReadPtr(nodeKpWritePtr);

std::mutex mutex;

// action proposal and classfication
// fps: ~10000
void nodeKpSubscriber()
{
	op::log("Starting Thread -- Node Keypoints Subscriber");
	op::log("--------------------------------------------");
	op::log(" ");

	// Initialize params and OpenposeKpSub
	ros::NodeHandle nh;
	ros::NodeHandle nh_p("~");
	
	std::string pose_topic;
	std::string pose_model;
	int tensor_offset;
	int tensor_str;
	
	nh_p.param<std::string>("pose_topic", pose_topic, FLAGS_pose_topic);
	nh_p.param<std::string>("pose_model", pose_model, FLAGS_model_pose);
	nh_p.param<int>("tensor_offset", tensor_offset, FLAGS_tensor_offset);
	nh_p.param<int>("tensor_str", tensor_str, FLAGS_tensor_stride);
	
	OpenposeKpSub openposeKpSubscriber(nh, pose_topic, node_seq, op::flagsToPoseModel(pose_model));
	openposeKpSubscriber.launchSubscriber();
	
	auto frame_id = -1;

	// get node extract callback ready
	ros::spinOnce();

	// processing frame by frame
	while (ros::ok())
	{
		// get body nodes from openpose
		arma::rowvec nodeKeypoints(openposeKpSubscriber.getNodeKeypoints());
		openposeKpSubscriber.resetNodeKeypoints();
        
		// check if the node keypoints are correctly delivered
		if (!nodeKeypoints.is_empty())
		{   
			++frame_id;
			if ((frame_id >= tensor_offset) && !((frame_id - tensor_offset) % tensor_str))
			{
				// set block mutex to secure data writing to shared map
				std::lock_guard<std::mutex> lock(mutex);

				// add tensor to repo
				ROS_INFO_STREAM("node keypoints size: " << nodeKeypoints.n_elem);
				nodeKpWritePtr->insert_rows(nodeKpWritePtr->n_rows, nodeKeypoints);
				ROS_INFO_STREAM("Current Tensor Repository Size: " << nodeKpWritePtr->n_rows);
			}
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
	op::log(" ");
	op::log("Starting Thread -- Action Classifier");
	op::log("------------------------------------");
	op::log(" ");

    // declare action classification service
    ros::NodeHandle nh;
    ros::NodeHandle nh_p("~");
    
    int tensor_repo_len = FLAGS_tensor_repo_len;
    nh_p.getParam("tensor_repo_len", tensor_repo_len);
    
	ros::ServiceClient action_classifier_client =
		nh.serviceClient<message_repository::ActionClassifier>("/pose_net/action_classifier");
	message_repository::ActionClassifier action_classifier_srv;

    auto tensor_id = 0;

    // swindow shape
    const auto n_frames = FLAGS_sliding_window_length;
    const auto n_nodes = 2*nodeSeq.size();
    const auto n_channels = 3;
    
    const op::Point<int> imageSize(1280,720);
    
    // get the position of neck and rhip in the node sequence for normalization
    std::vector<int>::const_iterator it_neck = std::find(nodeSeq.begin(), nodeSeq.end(), 8);
    std::vector<int>::const_iterator it_rhip = std::find(nodeSeq.begin(), nodeSeq.end(), 14);
    const auto id_neck = it_neck - nodeSeq.begin();
    const auto id_rhip = it_rhip - nodeSeq.end();

    // declare sliding window
    std::shared_ptr<arma::cube> sWindow(new arma::cube(n_frames, n_nodes, n_channels, arma::fill::zeros));
    std::shared_ptr<arma::mat> nodePtr(new arma::mat);
    
	ROS_INFO_STREAM("Initialize sliding window with size: [" << n_frames << "x" << n_nodes << "x" << n_channels << "]");
	ROS_INFO("Ready for action classification ...");

	// do the job
	// fps: 4~5
    while(ros::ok())
    {
        // make sure there is enough node data in the repository
        if (nodeKpReadPtr->n_rows >= tensor_id+1)
        {
            ROS_INFO_STREAM("Tensor ID: " << tensor_id);
            
            // for frame 0-8
            if ((tensor_id+1) < n_frames)
            {
                nodePtr->insert_rows(tensor_id, nodeKpReadPtr->row(tensor_id));
            }
            // for frame >= 9
            else
            {
                // update frame mat: shed the first row, append new coming frame
                if ((tensor_id+1) > n_frames)
                {
                    ROS_INFO("Updating Sliding Window ...");
                    nodePtr->shed_row(0);
                    nodePtr->insert_rows(n_frames-1, nodeKpReadPtr->row(tensor_id));
                }
                else
                {
                    nodePtr->insert_rows(tensor_id, nodeKpReadPtr->row(tensor_id));
                    ROS_INFO("Got first Sliding Window, ready for classification ...");
                }
                
                // when node repo reaches the maximal size
                if (nodeKpReadPtr->n_rows == tensor_repo_len)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    ROS_INFO_STREAM("Node Repository reaches size: " << tensor_repo_len);
                    ROS_INFO_STREAM("Releasing data between Frame Nr.0 and Frame Nr." << tensor_id-n_frames);  
                    nodeKpReadPtr->shed_rows(0, tensor_id-n_frames); 
                    tensor_id = n_frames - 1;    
                }
                
                // do smoothing
                // t: ~0.0025
                poseInterpolator(*nodePtr);

                // normalization
                // t: ~0.0002
                normTensor(*nodePtr, imageSize, id_neck, id_rhip);
                
                // calc 3d tensor
                // t: ~0.0001
                sWindow->slice(0) = *nodePtr;
                calcTensor(sWindow);
              
                // transform tensor to msg
                // t: ~0.0002
                std_msgs::Float64MultiArray tensor_msg;
				EigenTensorToMsg(sWindow, tensor_msg);
                action_classifier_srv.request.tensor = tensor_msg;
                
                op::log(" ");
			  	op::log("Action Classification");
			  	op::log(" ");
                
                // feed to classifier
                // t: ~0.2+
                if (action_classifier_client.call(action_classifier_srv))
			  	{
					ROS_INFO("Calling Action Classification Service ...");

					auto action_label = action_classifier_srv.response.label;

					ROS_INFO("The action label is: %d", action_label);
					ROS_INFO("Calling Action Classification Service Successfully!");
			  	}
			  	else
			  	{
					ROS_WARN("Calling Action Classification Service Failed!");
			  	}
            }
            
            op::log("----------------------------------------------------");
		    op::log(" ");
            
            ++tensor_id;
        }
    }
    
    return;
}

int main(int argc, char **argv)
{
  	google::InitGoogleLogging("data_manager");
  	gflags::ParseCommandLineFlags(&argc, &argv, true);
  	ros::init(argc, argv, "data_manager");
    
  	std::thread t_nodeKpSub(nodeKpSubscriber);
  	std::this_thread::sleep_for(std::chrono::milliseconds(100));
  	std::thread t_actionClassifier(actionClassifier);
 	
  	t_nodeKpSub.join();
  	t_actionClassifier.join();

  	return 0;
}



