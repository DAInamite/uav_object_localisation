//============================================================================
// Name        : tagger.cpp
// Author      : Stephan Wypler
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <ros/ros.h>
#include "Tester.h"
#ifndef HEADLESS
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif
using namespace std;
using namespace ros;

int main(int argc, char** argv) {
    ros::init(argc, argv, "localization_tester");
    NodeHandle n("~");
    string path(".");
    int threshold = 84; // twice the answer to everything!!!
    bool interactive = true;
    n.getParam("path", path);
    n.getParam("threshold", threshold);
    n.getParam("interactive", interactive);
#ifndef HEADLESS
	namedWindow("image", WINDOW_NORMAL);
	cv::startWindowThread();
#endif
	Tester t(path, threshold);
	t.test(interactive);
#ifndef HEADLESS
	destroyWindow("image");
#endif
	return 0;
}
