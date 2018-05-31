/*----- Image Sequence Publisher -----
* this node reads image sequence from a folder
* and then publishes it to /data_manager/image_seq_reader/image_seq
*/

 #include <ros/ros.h>
 #include <image_transport/image_transport.h>
 #include <cv_bridge/cv_bridge.h>
 
 #include <opencv2/highgui/highgui.hpp>
 #include "opencv2/imgcodecs.hpp"
 
 #include <gflags/gflags.h>
 #include <glog/logging.h>
 
 DEFINE_string(image_seq_folder, 		"/home/nvidia/MasterThesis/code/ros_pipeline/rp_ws/src/openpose_ros/data/ScanPassenger/*.png",	"The location of image sequence");
 
 int ImageSeqReader()
 {
 	// declare ros nodehandler and publisher
 	ros::NodeHandle nh;
 	image_transport::ImageTransport it(nh);
 	image_transport::Publisher image_seq_pub = it.advertise("/image_seq_reader/image_seq", 5);
 	
	// find image pattern and store the name in filenames
 	std::vector<cv::String> filenames;
 	cv::glob(FLAGS_image_seq_folder, filenames);
 	int image_num = filenames.size();
 	int count = 0;
 	
 	ros::Rate loop_rate(0.5);
 	
 	while(ros::ok())
 	{
 		if(count < image_num)
 		{
 			ROS_INFO_STREAM("Frame ID: " << count);
 			cv::Mat image = cv::imread(filenames.at(count));
 			
 			if(!image.empty())
 			{	
 				cv::namedWindow("Display Window");
 				cv::imshow("Display Window", image);
 				cv::waitKey(10);
 				
 				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
 				image_seq_pub.publish(msg);
 
 				++count;
 				ROS_INFO("CV Image Converted to ROS Msg Succeed!");
 			}
 			else
 			{
 				ROS_ERROR("Invalid Image, no message is transferred!");
 				return -1;
 			}
 			
 			ros::spinOnce();
 			loop_rate.sleep();
 		}
 		else
 		{
 			ROS_ERROR("No more valid images!");
 			ROS_INFO("Shutting down window ...");
 			cv::destroyWindow("Display Window");
 			return 0;
 		}
 	}
 }
 
 int main(int argc, char** argv)
 {
 	google::InitGoogleLogging("Image Sequence Reader");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ros::init(argc, argv, "image_seq_reader");
    
    return ImageSeqReader();
 }
