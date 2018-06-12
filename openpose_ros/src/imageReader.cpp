#include "openpose_ros/imageReader.hpp"

#include <opencv2/opencv.hpp>
#include <ros/rate.h>
#include <cv_bridge/cv_bridge.h>


 
ImageReader::ImageReader(ros::NodeHandle& nh, const std::string& image_path, 
                         const double pub_rate = 30., const std::string& cv_window = "Image Window") 
    : nh_(nh), it_(nh), image_path_(image_path), pub_rate_(pub_rate), OPENCV_WINDOW_{cv_window}
{
    image_pub_ = it_.advertise("/image_reader/image_raw", 1);
//    cv::namedWindow(OPENCV_WINDOW_);
}


ImageReader::~ImageReader()
{
//    cv::destroyWindow(OPENCV_WINDOW_);
}


// publish image msg
int ImageReader::pubImageMsg()
{
    auto count = 0;
    
    // get images from path
    std::vector<cv::String> filenames;
    cv::glob(image_path_, filenames);
    auto num_images = filenames.size();
    ros::Rate loop_rate(pub_rate_);
    
    while (ros::ok())
    {
        if(count < num_images)
 		{
 			ROS_INFO_STREAM("Frame ID: " << count);
 			cv::Mat image = cv::imread(filenames.at(count));
 			
 			if(!image.empty())
 			{	
//			    cv::imshow(OPENCV_WINDOW_, image);
//			    cv::waitKey(10);
 				
 				image_msg_ = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
 				image_pub_.publish(image_msg_);
 
 				++count;
 				ROS_INFO("CV Image to ROS Msg Conversion Done!");
 			}
 			else
 			{
 				ROS_ERROR("Invalid Image, no message is transported!");
 				return -1;
 			}
 			
 			ros::spinOnce();
 			loop_rate.sleep();
 		}
 		else
 		{
 		    ROS_WARN("No more available images!");
 		    ros::shutdown();
 		}
 		
    }
    
    return 0;
}



