/*------------------------------------------------- OpenPose ROS Node ---------------------------------------------------/
** [Publisher]: 
** input_image_pub      topic: /openpose_ros/input_image					msg_type: sensor_msgs::Image                    
** image_skeleton_pub   topic: /openpose_ros/detected_poses_image			msg_type: sensor_msgs::Image
** keypoints_pub        topic: /openpose_ros/detected_poses_keypoints		msg_type: message_repository::PersonDetection
**
** [Subscriber]:
** rosImgSubscriber     topic: /cv_camera/image_raw							msg_type: sensor_msgs::Image
**																					  sensor_msgs::ImageConstPtr
*/																				



// OpenPose dependencies
#include <openpose/headers.hpp>

// ros dependencies
#include <ros/ros.h>
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_repository/PersonDetection.h>

// std dependencies
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds

// opencv dependencies
#include <opencv2/highgui/highgui.hpp> //cv::imshow

// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging

// local defined classes and funcs
#include "openpose_ros/header.hpp"

// OpenPose
std::string model_folder_location = "/home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/models/";
std::string package_path = ros::package::getPath("openpose_ros");
std::string keypoints_folder_location = package_path + "/result/keypoints/";

DEFINE_int32(logging_level,			4,									"The logging level. Integer in the range [0, 255]. 0 will output any log() message,"
                                                						"while 255 will not output any. Current OpenPose library messages are in the range 0-4:"
                                                						"1 for low priority messages and 4 for important ones.");
DEFINE_string(camera_topic,         "/cv_camera/image_raw",		     	"Image topic that OpenPose will process.");
DEFINE_string(model_folder, 		model_folder_location,				"Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(keypoints_folder, 	keypoints_folder_location,			"Folder path where the output keypoints are located");
DEFINE_bool(save_json_keypoints, 	false,								"Set this flas to True to enable keypoints output in json file");
DEFINE_string(net_resolution, 		"656x368", 							"Multiples of 16. If it is increased, the accuracy usually increases."
                                         								"If it is decreased, the speed increases.");

DEFINE_string(output_resolution, 	"1280x720", 						"The image resolution (display and output). Use \"-1x-1\" to force the program to use"
                                           			  					"the default images resolution.");

DEFINE_string(model_pose, 			"COCO",								"Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_int32(num_gpu_start, 		0, 									"GPU device start number.");
DEFINE_double(scale_gap, 			0.3, 								"Scale gap between scales. No effect unless "
                              											"num_scales>1. Initial scale is always 1. If you"
                              											"want to change the initial scale, you actually "
                              											"want to multiply the `net_resolution` by"
                              											"your desired initial scale.");
DEFINE_int32(num_scales, 			1, 									"Number of scales to average.");

// OpenPose Rendering
DEFINE_bool(disable_blending, 		false, 								"If blending is enabled, it will merge "
                                     									"the results with the original frame. If "
                                     									"disabled, it will only display the results.");
DEFINE_double(alpha_pose, 			0.6,								"Blending factor (range 0-1) for the body part "
                               											"rendering. 1 will show it completely, 0 will"
                               											"hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap, 		0.7,								"Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
    																	"heatmap, 0 will only show the frame. Only valid for GPU rendering.");
DEFINE_double(render_threshold, 	0.05,
              															"Only estimated keypoints whose score confidences are higher "
              															"than this threshold will be rendered. Generally, a high threshold (> 0.5) will only"
              															"render very clear body parts; while small thresholds (~0.1) will also output guessed"
              															"and occluded keypoints, but also more false positives (i.e. wrong detections).");

int openPoseDetection()
{
	op::log("OpenPose ROS Client Node", op::Priority::High);
	
	ROS_INFO("Initialize OpenPose ...");
	// ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    
    
    // Step 2 - Read Google flags (user defined configuration)
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "1280x720");
	const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
	
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
    {
    	op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    }

    if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
    {
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or num_scales = 1.", 
        	__LINE__, __FUNCTION__, __FILE__);
    }

    
    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput;
    op::CvMatToOpOutput cvMatToOpOutput;
    op::OpOutputToCvMat opOutputToCvMat;
    
    std::map<unsigned int, std::string> bodypartMap = op::getPoseBodyPartMapping(poseModel);
    
    const bool enableGoogleLogging = true;
    
    std::shared_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor(new op::ScaleAndSizeExtractor(
    	netInputSize, outputSize, FLAGS_num_scales, FLAGS_scale_gap));

    std::shared_ptr<op::PoseExtractorCaffe> poseExtractorCaffe(new op::PoseExtractorCaffe(
    	poseModel, FLAGS_model_folder, FLAGS_num_gpu_start, {}, op::ScaleMode::ZeroToOne, enableGoogleLogging));

    std::shared_ptr<op::PoseGpuRenderer> poseGpuRenderer(new op::PoseGpuRenderer(
    	poseModel, poseExtractorCaffe, (float)FLAGS_render_threshold, !FLAGS_disable_blending, 
    	(float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap)); 
    std::shared_ptr<op::FrameDisplayer> frameDisplayer(new op::FrameDisplayer("OpenPose ROS Node", outputSize)); 


    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorCaffe->initializationOnThread();
    poseGpuRenderer->initializationOnThread();
    
    
	// Step 5 - Declare Publisher
    ros::NodeHandle nh;
    ros::Publisher input_image_pub = nh.advertise<sensor_msgs::Image>("/openpose_ros/input_image", 0); 
    ros::Publisher image_skeleton_pub = nh.advertise<sensor_msgs::Image>("/openpose_ros/detected_poses_image", 0);
  	ros::Publisher keypoints_pub = nh.advertise<message_repository::PersonDetection>("/openpose_ros/detected_poses_keypoints", 0);


    // Initialize the image subscriber
    RosImgSub rosImgSubscriber(nh, FLAGS_camera_topic);

    int frame_id = 0;
	
	ROS_INFO_STREAM("initial net_input_size: " << FLAGS_net_resolution);
  	ROS_INFO_STREAM("initial output_size: " << FLAGS_output_resolution);
  	ROS_INFO_STREAM("initial pose_model: " << FLAGS_model_pose);
	
	ROS_INFO("Initialization Done!");
	
    ros::spinOnce();
    
    const std::chrono::high_resolution_clock::time_point timerBegin = std::chrono::high_resolution_clock::now();

    while (ros::ok())
    {
        // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
        // Step 1 - Get cv_image ptr 
        cv_bridge::CvImagePtr cvImagePtr = rosImgSubscriber.getCvImagePtr();                     
        
        if(cvImagePtr != nullptr)
        {	
        	op::log(" ");
        	ROS_INFO_STREAM("Frame ID: " << frame_id);
            cv::Mat inputImage = cvImagePtr->image;
            
            // publish input image
            sensor_msgs::Image input_ros_image = *(cvImagePtr->toImageMsg());
            input_image_pub.publish(input_ros_image);
            
            
            // Step 2 - Format input image to OpenPose input and output formats
            const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
			double scaleInputToOutput;
            std::vector<double> scaleInputToNetInputs;
            std::vector<op::Point<int>> netInputSizes;
            op::Point<int> outputResolution;
            
            std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
                = scaleAndSizeExtractor->extract(imageSize);
            
            // t: ~0.03+
            auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
            auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
            
    
            // Step 3 - Estimate poseKeypoints
            // t: 1.5 ~ 2.+ 
            ROS_INFO("Performing Forward Pass ....");

            // Estimate poseKeypoints
            poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
            ROS_INFO("Forward Pass Done");

            const op::Array<float> poseKeypoints = poseExtractorCaffe->getPoseKeypoints();
            ROS_INFO("Keypoints Got");
            

            // Step 4 - Render poseKeypoints
            // t: ~0.02+
            poseGpuRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
            ROS_INFO("Pose Rendering Done"); 
            op::log(" ");
   

            // Step 5 - OpenPose output format to cv::Mat and publish detected pose image
            // t: ~0.02+
            auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);
            sensor_msgs::Image output_ros_image;
  			cvImagePtr->image = outputImage;
 	  		output_ros_image = *(cvImagePtr->toImageMsg());
  			image_skeleton_pub.publish(output_ros_image);
  			
  		
            // ------------------------- SHOWING RESULT AND CLOSING -------------------------
            //cv::imshow("OpenPose ROS Result", outputImage);
            //cv::waitKey(10);
            //frameDisplayer->displayFrame(outputImage, 10); // alternative
            
  			// -------------------------- RETRIEVE POSE INFO -----------------------------
  			// Validate poseKeypoints
  			// t: ~0.002+
  			retrievePoseInfo(poseKeypoints, keypoints_pub, bodypartMap);
  			op::log("------------------------------------------------------------------------------");
  			
  			if(FLAGS_save_json_keypoints)
  			{
  				std::shared_ptr<op::KeypointSaver> keypointJsonSaver(new op::KeypointSaver(FLAGS_keypoints_folder, op::DataFormat::Json));
  				
  				std::string fileName = "keypoints_" + op::toFixedLengthString(frame_id, 3u);
  				std::string keypointName = "pose_keypoints";
  				std::vector<op::Array<float>> keypointVector{poseKeypoints};
  				
  				keypointJsonSaver->saveKeypoints(keypointVector, fileName, keypointName);
  			}
            
            ++frame_id;
        }

        ros::spinOnce();
    }


    // Measuring total time
    const double totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
                              	(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
                              
    const std::string message = "Real-time pose estimation demo successfully finished. Total time: " 
                                + std::to_string(totalTimeSec) + " seconds. " + std::to_string(frame_id)
                                + " frames processed. Average fps is " + std::to_string(frame_id/totalTimeSec) + ".";
                                
    op::log(message, op::Priority::Max);

	return 0;
}



int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ros::init(argc, argv, "openpose_ros");

    return openPoseDetection();
}











































