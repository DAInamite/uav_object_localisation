#include "Estimator.h"


using namespace std;


int main(int argc, char **argv)
{
	ros::init(argc, argv, "object_estimator");

    string  topic_cnn("/detected_objects"),
            topic_blob("/object_detection_blob_node/object_detection_blob_node/detected_objects"),
            topic_wall("/object_detection_blob_node/object_detection_blob_node/detected_walls"),
            topic_publish_objects("/localized_objects"),
            topic_publish_walls("/localized_walls"),
			cameraFrame("usbcam"),
			worldFrame("world"),
			copterFrame("base_link");
    double FOV_v = 45.0;
    double FOV_h = 58.0;
    double sampleRate = 15.0;
    double cutoffFreqeuncy = 1.0;
    double publishInterval = 0.1;

    ros::NodeHandle n("~");

    n.getParam("fov_h", FOV_h);
    n.getParam("fov_v", FOV_v);
    n.getParam("topic_cnn", topic_cnn);
    n.getParam("topic_blob", topic_blob);
    n.getParam("topic_wall", topic_wall);
    n.getParam("topic_publish_objects", topic_publish_objects);
    n.getParam("topic_publish_walls", topic_publish_walls);
    n.getParam("cameraFrame", cameraFrame);
    n.getParam("worldFrame", worldFrame);
    n.getParam("copterFrame", copterFrame);
    n.getParam("filterSampleRate", sampleRate);
    n.getParam("filterCutoffFrequency", cutoffFreqeuncy);
    n.getParam("publishInterval", publishInterval);

    ROS_INFO("[CNN Objects] subscribing to %s", topic_cnn.c_str());
    ROS_INFO("[Blob Objects] subscribing to %s", topic_blob.c_str());
    ROS_INFO("[Walls] subscribing to %s", topic_wall.c_str());
    ROS_INFO("[FOW h] subscribing to %f", FOV_h);
    ROS_INFO("[FOW_v] subscribing to %f", FOV_v);
    ROS_INFO("[TF] using frame %s as camera frame", cameraFrame.c_str());
    ROS_INFO("[TF] using frame %s as world frame", worldFrame.c_str());
    ROS_INFO("[TF] using frame %s as copter frame", copterFrame.c_str());
    ROS_INFO("-------------------------------------");
    ROS_INFO("[Objects] publishing to %s", topic_publish_objects.c_str());
    ROS_INFO("[Walls] publishing to %s", topic_publish_walls.c_str());



    Estimator estimator(n, topic_cnn, topic_blob, topic_wall, topic_publish_objects, topic_publish_walls, cameraFrame, worldFrame, copterFrame, FOV_h, FOV_v, sampleRate, cutoffFreqeuncy, publishInterval);

    ros::AsyncSpinner spinner(1);
    spinner.start();

    ROS_INFO("Ready");

    ros::waitForShutdown();

	return 0;
}
