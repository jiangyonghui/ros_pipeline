#ifndef ROSIMAGESUBSCRIBER_HPP
#define ROSIMAGESUBSCRIBER_HPP

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_repository/PersonDetection.h>

// ROS Image Subscriber to camera_topic
class RosImgSub
{
    private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Subscriber image_sub_;
        cv_bridge::CvImagePtr cv_img_ptr_;
        std::string image_topic_;

    public:
        RosImgSub(ros::NodeHandle& nh, const std::string& image_topic);
        
        ~RosImgSub();

        void imageCallback(const sensor_msgs::ImageConstPtr& msg);

        cv_bridge::CvImagePtr& getCvImagePtr();
};

#endif

