#include <ros/ros.h>

#include <opencv2/opencv.hpp>

#include "Tester.h"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "object_detection_blob_test");
	ros::NodeHandle n("~");
	std::string path(".");
	int threshold = 84; // twice the answer to everything!!!
        bool waitaftererror = false;
	n.getParam("path", path);
	n.getParam("threshold", threshold);
    n.getParam("waiterror", waitaftererror);

	//cv::namedWindow("image", cv::WINDOW_NORMAL);
	//cv::startWindowThread();

    Tester t(path + "/", threshold, waitaftererror);
	t.test(n);

	//cv::destroyWindow("image");

	return 0;
}
