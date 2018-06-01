/*----- Image Sequence Publisher -----
* this node reads image sequence from a folder
* and then publishes it to /data_manager/image_seq_reader/image_seq
*/
#include "openpose_ros/imageReader.hpp"
#include <ros/package.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

std::string package_path = ros::package::getPath("openpose_ros");
std::string image_folder = package_path + "/data/ScanPassenger/*.png";

DEFINE_string(image_path,    image_folder,     "The location of image sequence");



// int ImageSeqReader()
// {
// 	// declare ros nodehandler and publisher
// 	ros::NodeHandle nh;
// 	image_transport::ImageTransport it(nh);
// 	image_transport::Publisher image_seq_pub = it.advertise("/image_reader/image_raw", 1);
// 	
//	// find image pattern and store the name in filenames
// 	std::vector<cv::String> filenames;
// 	cv::glob(FLAGS_image_seq_folder, filenames);
// 	int image_num = filenames.size();
// 	int count = 0;
// 	
// 	ros::Rate loop_rate(30);
// 	
// 	while(ros::ok())
// 	{
// 		if(count < image_num)
// 		{
// 			ROS_INFO_STREAM("Frame ID: " << count);
// 			cv::Mat image = cv::imread(filenames.at(count));
// 			
// 			if(!image.empty())
// 			{	
// 				if (FLAGS_display_image)
// 				{
// 				    cv::namedWindow("Display Window");
// 				    cv::imshow("Display Window", image);
// 				    cv::waitKey(10);
// 				}
// 				
// 				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
// 				image_seq_pub.publish(msg);
// 
// 				++count;
// 				ROS_INFO("CV Image Converted to ROS Msg Succeed!");
// 			}
// 			else
// 			{
// 				ROS_ERROR("Invalid Image, no message is transferred!");
// 				return -1;
// 			}
// 			
// 			ros::spinOnce();
// 			loop_rate.sleep();
// 		}
// 	}
// 	
// 	return 0;
// }

int main(int argc, char** argv)
{
    google::InitGoogleLogging("Image Sequence Reader");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ros::init(argc, argv, "image_reader");
    
    ros::NodeHandle nh;
    const double pub_rate = 0.5;
    ImageReader imageReader(nh, image_folder, pub_rate);

    return imageReader.pubImageMsg();
}









