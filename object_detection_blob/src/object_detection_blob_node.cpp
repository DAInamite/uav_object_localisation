#include "BlobDetection.h"
#include <string>

using namespace std;

int main(int argc, char **argv) {
	ros::init(argc, argv, "object_detection_blob_node");
	ros::NodeHandle n("~");
	bool publishMasks = false;
	string imageSubscriptionTopic = "/usb_cam/image_raw";
	string detectedObjectsTopic = "object_detection_blob_node/detected_objects";
    string detectedWallTopic = "object_detection_blob_node/detected_walls";
	string correctionMode = "naive";
	n.getParam("correction", correctionMode);
	n.getParam("publishMasks", publishMasks);
	n.getParam("imageTopic", imageSubscriptionTopic);
	n.getParam("objectsTopic", detectedObjectsTopic);
    n.getParam("wallTopic", detectedWallTopic);


	auto mode = LightCorrection::None;

	if(correctionMode == "naive")
		mode = LightCorrection::Naive;
	if(correctionMode == "local")
		mode = LightCorrection::Local;
	if(correctionMode == "global")
		mode = LightCorrection::Global;

	BlobDetection detection(n,
							imageSubscriptionTopic,
							detectedObjectsTopic,
							detectedWallTopic,
							mode,
							cv::Scalar(100,140,30), cv::Scalar(120,255,255),  // cup
							cv::Scalar(10,160,130), cv::Scalar(30,255,255),	 // battery
							cv::Scalar(170,150,150), cv::Scalar(10,255,255), // base
							publishMasks); // publish masks
	ROS_INFO("Ready");
	ros::spin();

	return 0;
}
