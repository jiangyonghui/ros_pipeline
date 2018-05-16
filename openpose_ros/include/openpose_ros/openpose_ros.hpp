#ifndef OPENPOSE_ROS_HPP
#define OPENPOSE_ROS_HPP

#include <openpose/headers.hpp>

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

    public:
        RosImgSub(const std::string& image_topic): it_(nh_)
        {
            // Subscribe to input video feed and publish output video feed
            image_sub_ = it_.subscribe(image_topic, 1, &RosImgSub::convertImage, this);
            cv_img_ptr_ = nullptr;
        }
        
        ~RosImgSub()
        {
        	ROS_INFO("Stop Image Subscription.");
        }

        void convertImage(const sensor_msgs::ImageConstPtr& msg)
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
        }

        cv_bridge::CvImagePtr& getCvImagePtr()
        {
            return cv_img_ptr_;
        }
        
        void resetCvImgPtr()
        {
        	cv_img_ptr_ = nullptr;
        	return;
        }
};


// get bodypart map
//std::map<unsigned int, std::string> getBodyPartMapFromPoseModel(const op::PoseModel &pose_model) 
//{
//  	if (pose_model == op::PoseModel::COCO_18) 
//  	{
//    	return op::POSE_COCO_BODY_PARTS;
//  	}	 
//  	else if (pose_model == op::PoseModel::MPI_15 || pose_model == op::PoseModel::MPI_15_4) 
//    {
//    	return op::POSE_MPI_BODY_PARTS;
//  	} 
//  	else 
//  	{
//    	ROS_FATAL("Invalid pose model, not map present");
//    	exit(1);
//  	}
//}


// bodypart detection
message_repository::BodypartDetection getBodyPartDetectionFromArrayAndIndex(const op::Array<float> &array, size_t idx) 
{
  	message_repository::BodypartDetection bodypart;

  	bodypart.x = array[idx];
  	bodypart.y = array[idx + 1];
  	bodypart.confidence = array[idx + 2];
  	
  	return bodypart;
}


// initialize bodypart
message_repository::BodypartDetection getNANBodypart() 
{
  	message_repository::BodypartDetection bodypart;
  	bodypart.confidence = NAN;
  	
  	return bodypart;
}


// retrieve pose and publish keypoints
bool retrievePoseInfo(const op::Array<float>& poseKeypoints, ros::Publisher& keypoints_pub, std::map<unsigned int, std::string>& bodypartMap)
{
	// -------------------------- RETRIEVE POSE INFO -----------------------------
  	// Validate poseKeypoints
  	if (!poseKeypoints.empty() && poseKeypoints.getNumberDimensions() != 3) 
  	{
    	ROS_ERROR("pose.getNumberDimensions(): %d != 3", (int)poseKeypoints.getNumberDimensions());
   	 	return false;
  	}

  	int num_people = poseKeypoints.getSize(0);
  	int num_bodyparts = poseKeypoints.getSize(1);

  	ROS_INFO("Detected Person(s): %d", num_people);
  	op::log("---------------------------");

  	for (size_t person_idx = 0; person_idx < num_people; person_idx++) 
  	{
    	op::log(" ");
    	ROS_INFO("Person ID: %zu", person_idx);
    	op::log(" ");

    	message_repository::PersonDetection person_msg;

    	// add number of people detected
    	person_msg.num_people_detected = num_people;

    	// add person ID
    	person_msg.person_ID = person_idx;

		// Initialize all bodyparts with nan
		person_msg.nose = getNANBodypart();
		person_msg.neck = getNANBodypart();
		person_msg.right_shoulder = getNANBodypart();
		person_msg.right_elbow = getNANBodypart();
		person_msg.right_wrist = getNANBodypart();
		person_msg.left_shoulder = getNANBodypart();
		person_msg.left_elbow = getNANBodypart();
		person_msg.left_wrist = getNANBodypart();
		person_msg.right_hip = getNANBodypart();
		person_msg.right_knee = getNANBodypart();
		person_msg.right_ankle = getNANBodypart();
		person_msg.left_hip = getNANBodypart();
		person_msg.left_knee = getNANBodypart();
		person_msg.left_ankle = getNANBodypart();
		person_msg.right_eye = getNANBodypart();
		person_msg.left_eye = getNANBodypart();
		person_msg.right_ear = getNANBodypart();
		person_msg.left_ear = getNANBodypart();
		person_msg.background = getNANBodypart();

    	// create bodyparts map
   	 	for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++) 
   	 	{
      		size_t final_idx = 3 * (person_idx * num_bodyparts + bodypart_idx);
      		std::string body_part_string = bodypartMap[bodypart_idx];
      		message_repository::BodypartDetection bodypart_detection =
          		getBodyPartDetectionFromArrayAndIndex(poseKeypoints, final_idx);

		  	if (body_part_string == "Nose")
				person_msg.nose = bodypart_detection;
		  	else if (body_part_string == "Neck")
				person_msg.neck = bodypart_detection;
		  	else if (body_part_string == "RShoulder")
				person_msg.right_shoulder = bodypart_detection;
		  	else if (body_part_string == "RElbow")
				person_msg.right_elbow = bodypart_detection;
		  	else if (body_part_string == "RWrist")
				person_msg.right_wrist = bodypart_detection;
		  	else if (body_part_string == "LShoulder")
				person_msg.left_shoulder = bodypart_detection;
		  	else if (body_part_string == "LElbow")
				person_msg.left_elbow = bodypart_detection;
		  	else if (body_part_string == "LWrist")
				person_msg.left_wrist = bodypart_detection;
		  	else if (body_part_string == "RHip")
				person_msg.right_hip = bodypart_detection;
		  	else if (body_part_string == "RKnee")
				person_msg.right_knee = bodypart_detection;
		  	else if (body_part_string == "RAnkle")
				person_msg.right_ankle = bodypart_detection;
		  	else if (body_part_string == "LHip")
				person_msg.left_hip = bodypart_detection;
		  	else if (body_part_string == "LKnee")
				person_msg.left_knee = bodypart_detection;
		  	else if (body_part_string == "LAnkle")
				person_msg.left_ankle = bodypart_detection;
		  	else if (body_part_string == "REye")
				person_msg.right_eye = bodypart_detection;
		  	else if (body_part_string == "LEye")
				person_msg.left_eye = bodypart_detection;
		  	else if (body_part_string == "REar")
				person_msg.right_ear = bodypart_detection;
		  	else if (body_part_string == "LEar")
				person_msg.left_ear = bodypart_detection;
		  	else if (body_part_string == "Background")
		  	{
		  		person_msg.background.x = 0.;
		  		person_msg.background.y = 0.;
		  		person_msg.background.confidence = 1.0;
		  	}
		  	else 
		  	{
				ROS_ERROR("Unknown bodypart %s, this should never happen!",body_part_string.c_str());
		  	}

      		ROS_INFO("body part: %s", body_part_string.c_str());
     		ROS_INFO("(x, y, confidence): %f, %f, %f", bodypart_detection.x, bodypart_detection.y, bodypart_detection.confidence);
    	}

    	// publish keypoints data of person_msg
    	keypoints_pub.publish(person_msg);
  	}

  	op::log(" ");
	
	return true;
}

//bool saveJsonKeypoints(const op::Array<float>& poseKeypoints, const std::string& keypointsFolder, const int& frame_id)
//{
//	std::shared_ptr<op::KeypointJsonSaver> keypointJsonSaver(new op::KeypointJsonSaver(keypointsFolder));	
//	
//	const bool humanReadable = true;
//	const auto stringLength = 12u;

//	// extendible for hand and face keypoints
//	const std::vector<std::pair<op::Array<float>, std::string>> keypointVector
//	{
//    	std::make_pair(poseKeypoints, "pose_keypoints")
//    	// std::make_pair(handKeypoints, "hand_keypoints"),
//    	// std::make_pair(faceKeypoints, "face_keypoints")
//	};

//	std::string fileName = "keypoints_" + op::toFixedLengthString(frame_id, stringLength);
//	
//	ROS_INFO("Writing keypoints to file ...");
//	keypointJsonSaver->save(keypointVector, fileName, humanReadable);

//	ROS_INFO("Keypoints saved");
//	
//	return true;
//}



#endif
