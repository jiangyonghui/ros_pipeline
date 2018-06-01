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
#include <opencv2/imgcodecs/imgcodecs.hpp> // cv::imwrite

// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging

// local defined classes and funcs
#include <openpose_ros/header.hpp>

#include <algorithm> // std::generate

#include "openpose_ros/spline.h"

// OpenPose
std::string model_folder_location = "/home/nvidia/MasterThesis/code/ros_pipeline/3rdPartyLib/openpose/models/";
std::string package_path = ros::package::getPath("openpose_ros");
std::string keypoints_folder_location = package_path + "/result/keypoints/";
std::string image_folder_location = package_path + "/result/image/";

DEFINE_int32(logging_level,			4,									"The logging level. Integer in the range [0, 255]. 0 will output any log() message,"
                                                						"while 255 will not output any. Current OpenPose library messages are in the range 0-4:"
                                                						"1 for low priority messages and 4 for important ones.");
DEFINE_string(image_topic,         "/cv_camera/image_raw",		     	"Image topic that OpenPose will process.");
DEFINE_string(model_folder, 		model_folder_location,				"Folder (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(keypoints_folder, 	keypoints_folder_location,			"Folder where the output keypoints are located");
DEFINE_string(image_folder,         image_folder_location,              "Folder where the output skeleton-rendered images are located");
DEFINE_bool(save_json_keypoints, 	false,								"Set true to enable keypoints output in json file");
DEFINE_bool(save_skeleton_images,   false,                              "Set true to save skeleton-rendered images");
DEFINE_bool(display_image,          false,                              "Set true to display skeleton-renderd image");
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
    
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "1280x720");
	const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

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
    
    std::shared_ptr<std::vector<op::Array<float>>> poseKpPtr(new std::vector<op::Array<float>>);
    std::shared_ptr<std::vector<op::Array<float>>> outputArrayPtr(new std::vector<op::Array<float>>);
    std::shared_ptr<std::vector<double>> scaleInputToOutputPtr(new std::vector<double>);

    // Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorCaffe->initializationOnThread();
    poseGpuRenderer->initializationOnThread();
    
    ros::NodeHandle nh;
    RosImgSub rosImgSubscriber(nh, FLAGS_image_topic);

    // Initialize the image subscriber
    int frame_id = 0;
	
    ros::spinOnce();
    
    const std::chrono::high_resolution_clock::time_point timerBegin = std::chrono::high_resolution_clock::now();
    
    ROS_INFO("Initialization Done, waiting for image msg ...");
    
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
            
            
            scaleInputToOutputPtr->push_back(scaleInputToOutput);
            outputArrayPtr->push_back(outputArray);
            
    
            // Step 3 - Estimate poseKeypoints
            // t: 1.5 ~ 2.+ 
            ROS_INFO("Performing Forward Pass ....");

            // Estimate poseKeypoints
            // pose processing ->CPU vs GPU
            poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
            ROS_INFO("Forward Pass Done");

            const op::Array<float> poseKeypoints = poseExtractorCaffe->getPoseKeypoints();
            ROS_INFO("Keypoints Got");
            
            poseKpPtr->push_back(poseKeypoints);
            
            rosImgSubscriber.resetCvImagePtr();
            ++frame_id;
        }
        
        if(poseKpPtr->size() == 24)
        {
            // do smoothing
            ROS_INFO("Pose Smoothing ...");
            
            const auto num_people = poseKpPtr->at(0).getSize(0);
            const auto num_nodes = poseKpPtr->at(0).getSize(1);
            const auto num_channel = poseKpPtr->at(0).getSize(2);
            const auto num_frame = poseKpPtr->size();
            
            ROS_INFO_STREAM("num people: " << num_people);
            ROS_INFO_STREAM("num nodes: " << num_nodes);
            ROS_INFO_STREAM("num channel: " << num_channel);
            ROS_INFO_STREAM("num frame: " << num_frame);
         
            tk::spline s_x, s_y;
            std::vector<double> t(num_frame), val_x, val_y; 
            std::generate(t.begin(), t.end(), [n=0.]() mutable {return ++n;}); 
            
            for (auto node = 0; node < num_nodes; ++node)
            {
                for (auto frame = 0; frame < num_frame; ++frame)
                {
                    val_x.push_back(poseKpPtr->at(frame).at(3*node));
                    val_y.push_back(poseKpPtr->at(frame).at(3*node+1));
                }

                s_x.set_points(t, val_x);
                s_y.set_points(t, val_y);
                
                for (auto frame = 0; frame < num_frame; ++frame)
                {
                    poseKpPtr->at(frame).at(3*node) = s_x(static_cast<double>(frame));
                    poseKpPtr->at(frame).at(3*node+1) = s_y(static_cast<double>(frame));
                }
                val_x.clear();
                val_y.clear();  
            }
                    
                   
            // Render poseKeypoints
            for (auto frame = 0; frame < num_frame; ++frame)
            {
                // t: ~0.02+
                poseGpuRenderer->renderPose(outputArrayPtr->at(frame), poseKpPtr->at(frame), scaleInputToOutputPtr->at(frame));
                ROS_INFO("Pose Rendering Done"); 
                op::log(" ");

                auto outputImage = opOutputToCvMat.formatToCvMat(outputArrayPtr->at(frame));
                      			
	
	            // save keypoints in json file
	            if  (FLAGS_save_json_keypoints)
	            {
		            std::shared_ptr<op::KeypointSaver> keypointJsonSaver(new op::KeypointSaver(FLAGS_keypoints_folder, op::DataFormat::Json));
		
		            std::string fileName = "keypoints_" + op::toFixedLengthString(frame, 3u);
		            std::string keypointName = "pose_keypoints";
		            std::vector<op::Array<float>> keypointVector{poseKpPtr->at(frame)};
		
		            keypointJsonSaver->saveKeypoints(keypointVector, fileName, keypointName);
	            }
	
	            // save skeleton-rendered image
	            if  (FLAGS_save_skeleton_images)
	            {
	                std::string imageName = FLAGS_image_folder + "skeleton_" + op::toFixedLengthString(frame, 3u) + ".png";
	                cv::imwrite(imageName, outputImage);
	            }
            
            }
            
            ros::shutdown();
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
    ros::init(argc, argv, "pose_smoothing");

    return openPoseDetection();
}











































