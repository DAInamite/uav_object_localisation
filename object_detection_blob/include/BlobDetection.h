#ifndef BLOBDETECTION_H
#define BLOBDETECTION_H

#include "DetectionObjects.h"
#include "LightCorrection.h"
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include "object_detection_blob/adjust_ranges.h"

using namespace cv;
using namespace std;

class BlobDetection
{
public:
    /** BlobDetection constructor
        @param nh			 ROS Node Handle.
        @param imageTopic	 The image topic to subscribe to.
        @param objectsTopic  The topic to publish detected objects on.
        @param mode          The light correction mode applied to incoming images.
        @param cupcollow     The lower colour-border of the cup.
        @param cupcolup      The upper colour-border of the cup.
        @param batterycollow The lower colour-border of the battery.
        @param batterycolup  The upper colour-border of the battery.
        @param basecollow    The lower colour-border of the base.
        @param basecolup     The upper colour-border of the base.
        @param publishMasks  Toggle publishing of masks.
     */
	BlobDetection(ros::NodeHandle& nh, const std::string& imageTopic, const std::string& objectsTopic, const std::string& wallTopic, const LightCorrection::CorrectionMode mode,
            const cv::Scalar& cupcollow, const cv::Scalar& cupcolup,
			const cv::Scalar& batterycollow, const cv::Scalar& batterycolup,
			const cv::Scalar& basecollow, const cv::Scalar& basecolup, bool publishMasks);
	~BlobDetection();
	// service callbacks
	bool setRangesBase(object_detection_blob::adjust_ranges::Request &req, object_detection_blob::adjust_ranges::Response &res);
	bool setRangesBattery(object_detection_blob::adjust_ranges::Request &req, object_detection_blob::adjust_ranges::Response &res);
	bool setRangesCup(object_detection_blob::adjust_ranges::Request &req, object_detection_blob::adjust_ranges::Response &res);

	struct Result{
		//! The detected object type
		DetectionObjectTypes sType;
		//! The rect around the object in the image
		cv::RotatedRect sObjectRect;
	};


	std::vector<Result> detectObjects(const cv::Mat& image);

private:
	ros::NodeHandle n;
	//! Executes the blob detection at the given image
	std::pair<bool, Result> detectObject(const cv::Mat& image, DetectionObjectTypes DO_type);
	void setupColourRange(const cv::Scalar& low, const cv::Scalar& up, std::string rangeName);
	cv::Mat generateThreshold(const cv::Mat& image, const std::pair<cv::Scalar, cv::Scalar>& range);

	// subscription callback
	void handleOnImageReceived(const sensor_msgs::ImageConstPtr &msg);

	bool publishMasks;
	LightCorrection *pCorrection;

    //! The colour range of the cup
	std::pair<cv::Scalar, cv::Scalar> mCupColourRange;
	//! The colour range of the battery
	std::pair<cv::Scalar, cv::Scalar> mBatteryColourRange;
	//! The colour range of the base
	std::pair<cv::Scalar, cv::Scalar> mBaseColourRange;

	// Test methods
	void showImage(cv::Mat img, string name, bool isHSV = false) const;

	void detectWalls(cv::Mat& image, const ros::Time& timestamp) const;

	void showHist(cv::Mat& img, std::string name) const;

	// ros stuff
	ros::Publisher objectPublisher;
    ros::Publisher wallPublisher;
	image_transport::ImageTransport it;
	image_transport::Subscriber imageSubscriber;
	image_transport::Publisher baseMaskPublisher;
	image_transport::Publisher batteryMaskPublisher;
	image_transport::Publisher cupMaskPublisher;
	image_transport::Publisher lightCorrectedPublisher;
	ros::ServiceServer serv_base;
	ros::ServiceServer serv_battery;
	ros::ServiceServer serv_cup;

};

#endif // BLOBDETECTION_H

