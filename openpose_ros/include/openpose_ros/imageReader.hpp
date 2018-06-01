#ifndef IMAGEREADER_HPP
#define IMAGEREADER_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <string>
#include <vector>


class ImageReader
{
    private: 
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Publisher image_pub_;
        sensor_msgs::ImagePtr image_msg_;
        std::string image_path_;
        double pub_rate_;
        std::string OPENCV_WINDOW_;
        
    public:
        ImageReader(ros::NodeHandle& nh, const std::string& image_path, const double pub_rate, const std::string& cv_window);
        ~ImageReader();
        int pubImageMsg();
};

#endif


