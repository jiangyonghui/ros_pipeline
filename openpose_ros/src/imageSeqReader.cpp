/*----- Image Sequence Publisher -----
* this node reads image sequence from a folder
* and then publishes it to /data_manager/image_seq_reader/image_seq
*/
#include "openpose_ros/imageReader.hpp"
#include <ros/package.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

std::string package_path = ros::package::getPath("openpose_ros");
std::string data_folder = package_path + "/data/";

DEFINE_string(image_folder,    "ScanPassenger/*.png",     "The location of image sequence");
DEFINE_double(read_rate,       0.5,                        "image read rate");


int main(int argc, char** argv)
{
    google::InitGoogleLogging("Image Sequence Reader");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ros::init(argc, argv, "image_reader");
    
    ros::NodeHandle nh;
    ros::NodeHandle nh_p("~");
    double read_rate(FLAGS_read_rate);
    std::string image_path = data_folder + FLAGS_image_folder;
    nh_p.getParam("image_path", image_path);
    nh_p.getParam("read_rate", read_rate);
    ImageReader imageReader(nh, image_path, read_rate, "Image Window");

    return imageReader.pubImageMsg();
}









