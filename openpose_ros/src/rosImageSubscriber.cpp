#include "openpose_ros/rosImageSubscriber.hpp"



// ROS Image Subscriber to camera_topic
RosImgSub::RosImgSub(ros::NodeHandle& nh, const std::string& image_topic): nh_(nh), it_(nh), image_topic_(image_topic)
{
    image_sub_ = it_.subscribe(image_topic_, 1, &RosImgSub::imageCallback, this);
}
 

RosImgSub::~RosImgSub()
{
    ROS_INFO("Subscription to image message terminated.");
}


void RosImgSub::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_img_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    return;
}
 
cv_bridge::CvImagePtr& RosImgSub::getCvImagePtr()
{
    return cv_img_ptr_;
}

void RosImgSub::resetCvImagePtr()
{
    cv_img_ptr_.reset();
    return;
}
 


