#include "CNN.h"
#include <ros/ros.h>
#include <sstream>
#include <string>
#include <vector>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "object_recognition_cnn/Object.h"
#include "object_recognition_cnn/LoadWeights.h"
#ifdef CONVERSIONTESTS
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "loader.h"
#endif

using namespace tiny_cnn;
using namespace std;
using namespace cv;
using namespace ros;

// that surrounds almost anything with quotes (even other macro's contents)
#define STRX(...) # __VA_ARGS__
#define STR(...) STRX(__VA_ARGS__)

int main(int argc, char** argv) {
#ifdef CONVERSIONTESTS

    namedWindow("debug", WINDOW_NORMAL);
	Mat img = imread("/home/stephan/Dokumente/Master/spacebot2014ws/testImages/battery/b1.png");
	vec_t in, out;
	auto printminmax = [](vec_t& vector){
        auto minmax = minmax_element(vector.begin(), vector.end());
		cout << "size: " << vector.size() << " min: " << *minmax.first << " max: " << *minmax.second << endl;
	};

	cv::Mat conv_image;
	vector<cv::Mat> split_image;
	cv::cvtColor(img, conv_image, CV_BGR2HSV); // note that this gives you
    cv::split(conv_image, split_image);
	for(auto& channel : split_image){
	    double min, max;
	    cv::minMaxLoc(channel, &min, &max);
        cout << "min: " << min << " max: " << max << endl;
    }
    in = mat2vec_t<hsv>(img);
    cout << "HSV: ";
    printminmax(in);
	imshow("debug", vec_t2bgrMat<hsv>(in));
	waitKey(0);
	in = mat2vec_t<yuv>(img);
	cout << "YUV: ";
	printminmax(in);
    imshow("debug", vec_t2bgrMat<yuv>(in));
	waitKey(0);
	in = mat2vec_t<ycrcb>(img);
	cout << "YCrCb: ";
	printminmax(in);
    imshow("debug", vec_t2bgrMat<ycrcb>(in));
	waitKey(0);
	in = mat2vec_t<bgr>(img);
	cout << "BGR: ";
	printminmax(in);
    imshow("debug", vec_t2bgrMat<bgr>(in));
    waitKey(0);
    // this is how you convert and display grayscale
	in = mat2vec_t<ycrcb>(img, {0});
	cout << "gray: ";
	printminmax(in);
    cv::split(vec_t2bgrMat<bgr>(in, {0}), split_image);
    imshow("debug", split_image[0]);
	waitKey(0);

    destroyWindow("debug");
	return 0;
#endif

	ROS_INFO("Network configured as follows:\n"
			"Use TBB: %s\n"
			"Use SSE: %s\n"
			"Use AVX: %s\n"
			"Fully Connected: %s\n"
			"BGR Color Space: %s\n"
			"YCrCb Color Space: %s\n"
			"HSV Color Space: %s\n"
			"YUV Color Space: %s\n"
			"Channels %s\n",
#if defined(CNN_USE_TBB)
			"yes",
#else
			"no",
#endif
#if defined(CNN_USE_SSE)
			"yes",
#else
			"no",
#endif
#if defined(CNN_USE_AVX)
			"yes",
#else
			"no",
#endif
#if defined(FULLY_CONNECTED)
			"yes",
#else
			"no",
#endif
#if !defined(HSV_COLOR_SPACE) && !defined(YUV_COLOR_SPACE) && !defined(YCRCB_COLOR_SPACE)
			"yes",
#else
			"no",
#endif
#if defined(YCRCB_COLOR_SPACE)
			"yes",
#else
			"no",
#endif
#if defined(HSV_COLOR_SPACE)
			"yes",
#else
			"no",
#endif

#if defined(YUV_COLOR_SPACE)
			"yes",
#else
			"no",
#endif
#if defined(CHANNELS)
			STR(CHANNELS)
#else
            "{0, 1, 2}"
#endif
	);
	/**
	 * The ros::init() function needs to see argc and argv so that it can perform
	 * any ROS arguments and name remapping that were provided at the command line. For programmatic
	 * remappings you can use a different version of init() which takes remappings
	 * directly, but for most command-line programs, passing argc and argv is the easiest
	 * way to do it.  The third argument to init() is the name of the node.
	 *
	 * You must call one of the versions of ros::init() before using any other
	 * part of the ROS system.
	 */
	ros::init(argc, argv, "object_recognition_cnn");

	/**
	 * NodeHandle is the main access point to communications with the ROS system.
	 * The first NodeHandle constructed will fully initialize this node, and the last
	 * NodeHandle destructed will close down the node.
	 */
	NodeHandle nh;

	/**
	 * The advertise() function is how you tell ROS that you want to
	 * publish on a given topic name. This invokes a call to the ROS
	 * master node, which keeps a registry of who is publishing and who
	 * is subscribing. After this advertise() call is made, the master
	 * node will notify anyone who is trying to subscribe to this topic name,
	 * and they will in turn negotiate a peer-to-peer connection with this
	 * node.  advertise() returns a Publisher object which allows you to
	 * publish messages on that topic through a call to publish().  Once
	 * all copies of the returned Publisher object are destroyed, the topic
	 * will be automatically unadvertised.
	 *
	 * The second parameter to advertise() is the size of the message queue
	 * used for publishing messages.  If messages are published more quickly
	 * than we can send them, the number here specifies how many messages to
	 * buffer up before throwing some away.
	 */
	ros::Publisher object_pub = nh.advertise<object_recognition_cnn::Object>("/detected_objects", 1);

#ifndef HEADLESS
	namedWindow("prediction", WINDOW_NORMAL);
	namedWindow("weight1", WINDOW_NORMAL);
	namedWindow("weight3", WINDOW_NORMAL);
	namedWindow("tile", WINDOW_NORMAL);
	namedWindow("error", WINDOW_NORMAL);
	cv::startWindowThread();
#endif
	NodeHandle n("~");
    string localisationServiceName = "/objRecog/localizeObj";
    n.getParam("localisationService", localisationServiceName);
    string interesting_images_topic = "/interesting_images";
    n.getParam("interesting_images_topic", interesting_images_topic);
    bool useOwnLocalisazion = false;
    n.getParam("use_own_localisazion", useOwnLocalisazion);
	CNN cnn(nh, object_pub, interesting_images_topic, localisationServiceName, useOwnLocalisazion);

	string image_topic = "/robot/Kinect/rgb/image";
	n.getParam("image", image_topic);
	ROS_INFO("subscribing to %s", image_topic.c_str());
    image_transport::ImageTransport it(nh);
	image_transport::Subscriber sub = it.subscribe(image_topic, 1, &CNN::onImageArrive, &cnn);
	ros::ServiceServer tester = nh.advertiseService("test", &CNN::test, &cnn);
	ros::ServiceServer trainer = nh.advertiseService("train", &CNN::train, &cnn);
	ros::ServiceServer clearer = nh.advertiseService("clear", &CNN::clear, &cnn);
	ros::ServiceServer weightLoader = nh.advertiseService("loadWeights", &CNN::loadWeights, &cnn);
	ros::ServiceServer weightStorer = nh.advertiseService("storeWeights", &CNN::storeWeights, &cnn);
	ros::ServiceServer imageLoader = nh.advertiseService("loadImages", &CNN::loadImagesFromFiles, &cnn);
	ros::ServiceServer tileTester = nh.advertiseService("localize", &CNN::localize, &cnn);

	// load initial weights if given
	string weightsPath = "";
	n.getParam("weights", weightsPath);
	if(weightsPath.length() > 0){
		if(cnn.loadWeightsImpl(weightsPath)){
			ROS_INFO("successfully loaded initial weights");
		} else {
	        ROS_ERROR("error loading weights file");
		}
	}

	ROS_INFO("Ready");
    ros::spin();
	/*
	ros::AsyncSpinner spinner(1);
	spinner.start();
    ros::waitForShutdown();
	 */

#ifndef HEADLESS
	destroyWindow("prediction");
	destroyWindow("weight1");
	destroyWindow("weight3");
	destroyWindow("tile");
	destroyWindow("error");
#endif

	return 0;
}
