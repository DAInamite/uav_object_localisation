#include "BlobDetection.h"
#include <opencv2/opencv.hpp>
#include <std_msgs/Header.h>
#include <cv_bridge/cv_bridge.h>
#include <algorithm>
#include "object_detection_blob/BlobObject.h"
#include "object_detection_blob/BlobWall.h"
#include "object_detection_blob/Vector2.h"
#include "LightCorrection.h"

BlobDetection::BlobDetection(ros::NodeHandle& nh, const std::string& imageTopic, const std::string& objectsTopic, const std::string& wallTopic, const LightCorrection::CorrectionMode mode,
		                     const cv::Scalar& cupcollow, const cv::Scalar& cupcolup,
							 const cv::Scalar& batterycollow, const cv::Scalar& batterycolup,
							 const cv::Scalar& basecollow, const cv::Scalar& basecolup, bool pm):
							 publishMasks(pm), pCorrection(nullptr), it(nh) {
	pCorrection = new LightCorrection(mode);
	setupColourRange(cupcollow, cupcolup, "cup");
	setupColourRange(batterycollow, batterycolup, "battery");
	setupColourRange(basecollow, basecolup, "base");
	serv_base = nh.advertiseService("set_ranges_base", &BlobDetection::setRangesBase, this);
	serv_battery = nh.advertiseService("set_ranges_battery", &BlobDetection::setRangesBattery, this);
	serv_cup = nh.advertiseService("set_ranges_cup", &BlobDetection::setRangesCup, this);
	imageSubscriber = it.subscribe(imageTopic, 1, &BlobDetection::handleOnImageReceived, this);
	baseMaskPublisher = it.advertise("object_detection_blob_node/base_mask", 1);
	batteryMaskPublisher = it.advertise("object_detection_blob_node/battery_mask", 1);
	cupMaskPublisher = it.advertise("object_detection_blob_node/cup_mask", 1);
	lightCorrectedPublisher = it.advertise("object_detection_blob_node/lightCorrected", 1);
	objectPublisher = nh.advertise<object_detection_blob::BlobObject>(objectsTopic, 1000);
	wallPublisher = nh.advertise<object_detection_blob::BlobWall>(wallTopic, 1000);
}

BlobDetection::~BlobDetection(){
	if(pCorrection)
		delete pCorrection;
	pCorrection = nullptr;
}

void BlobDetection::setupColourRange(const cv::Scalar& low, const cv::Scalar& up, std::string rangeName) {
	// std::pair<cv::Scalar, cv::Scalar>& range;
	if(rangeName == "base") {
		mBaseColourRange.first = low;

		// HSV in opencv goes from 0 -> 180 in the H channel
		mBaseColourRange.second = cv::Scalar(fmod(up[0], 180.f), up[1], up[2]);
	}else if(rangeName == "battery") {
		mBatteryColourRange.first = low;

		// HSV in opencv goes from 0 -> 180 in the H channel
		mBatteryColourRange.second = cv::Scalar(fmod(up[0], 180.f), up[1], up[2]);
	}else if(rangeName == "cup") {
		mCupColourRange.first = low;

		// HSV in opencv goes from 0 -> 180 in the H channel
		mCupColourRange.second = cv::Scalar(fmod(up[0], 180.f), up[1], up[2]);
	}
}

std::vector<BlobDetection::Result> BlobDetection::detectObjects(const cv::Mat& image) {
	std::vector<Result>	result_vec;

	for (int objIndex = DO_Base; objIndex <= DO_Cup; objIndex++ )	{
		auto obj_result = detectObject(image, static_cast<DetectionObjectTypes>(objIndex));

		if (obj_result.first) {
			result_vec.push_back(obj_result.second);

			// DEBUG
			std::string name;
			switch(objIndex) {
				case DO_Base: name="Base"; break;
				case DO_Cup: name="Cup"; break;
				case DO_Battery: name="Battery"; break;
			}

			ROS_INFO("found %s", name.c_str());
		}
	}

	return result_vec;
}

cv::Mat BlobDetection::generateThreshold(const cv::Mat& image, const std::pair<cv::Scalar, cv::Scalar>& range) {
	cv::Mat threshold;

	if (range.first[0] < range.second[0]) {
		cv::inRange(image, range.first, range.second, threshold);
	} else {
		// we go over the border of 180 and so we need two inRange calls
		cv::Mat threshold_tmp_0, threshold_tmp_1;

		cv::inRange(image, range.first, cv::Scalar(180, range.second[1], range.second[2]), threshold_tmp_0);
		cv::inRange(image, cv::Scalar(0, range.first[1], range.first[2]), range.second, threshold_tmp_1);

		cv::add(threshold_tmp_0, threshold_tmp_1, threshold);
	}

	// reduce noise by eroding / delating		
	int erosion_size = 3;  
	cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size) );

	// Apply erosion or dilation on the image
	erode(threshold, threshold, element); 
	dilate(threshold, threshold, element);

	return threshold;
}

std::pair<bool, BlobDetection::Result> BlobDetection::detectObject(const cv::Mat& image, DetectionObjectTypes DO_type) {
	Mat threshold;
	int minArea = 0;
	int maxArea = 0;
	if(publishMasks){
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "hsv", image).toImageMsg();
		lightCorrectedPublisher.publish(msg);
	}

	switch(DO_type) {
		case DO_Base:
			threshold = generateThreshold(image, mBaseColourRange);
			minArea = 400;
			maxArea = 20000;
			if(publishMasks){
				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", threshold).toImageMsg();
				baseMaskPublisher.publish(msg);
			}
			break;
		case DO_Battery:
			threshold = generateThreshold(image, mBatteryColourRange);
			minArea = 200;
			maxArea = 10000;
			if(publishMasks){
				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", threshold).toImageMsg();
				batteryMaskPublisher.publish(msg);
			}
			break;
		case DO_Cup:
			threshold = generateThreshold(image, mCupColourRange);
			minArea = 200;
			maxArea = 10000;
			if(publishMasks){
				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", threshold).toImageMsg();
				cupMaskPublisher.publish(msg);
			}
			break;
	}

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(threshold, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		
	std::vector<cv::Point> approx;
	std::vector<cv::Point> found_object_points;
	int currArea = 0;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);
		int area = contourArea(cv::Mat(approx));

		if (area < minArea || area > maxArea)
			continue;

		// we add the object if we didn't found one already or we got an object with children in the hierachy or it's bigger
		if (found_object_points.empty() || area > currArea || hierarchy[i][2] != -1)  {
			found_object_points = approx;
			currArea = area;
		}
	}

	if (!found_object_points.empty()) {
		Result result;
		result.sType = DO_type;
		result.sObjectRect = cv::minAreaRect(found_object_points);

		return std::make_pair(true, result);
	}

	return std::make_pair(false, Result());
}

void BlobDetection::showImage(Mat img, string name, bool isHSV) const {
#ifdef DEBUG
    if(isHSV){
        cv::Mat tmp;
        cv::cvtColor(img, tmp, CV_HSV2BGR);
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, tmp);
    } else {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, img);
    }
	cv::waitKey(1);
#endif
}

void BlobDetection::handleOnImageReceived(const sensor_msgs::ImageConstPtr &msg)
{
	auto image = cv_bridge::toCvShare(msg, "bgr8")->image;

	showImage(image, "original");
	cv::Mat lightCorrectedImage;
	pCorrection->correct(image, lightCorrectedImage);
    // showImage(lightCorrectedImage, "corrected", true);

	detectWalls(image, msg->header.stamp);

    auto results = detectObjects(lightCorrectedImage);

    for (auto& result : results) {
		object_detection_blob::BlobObject rmsg;

	    rmsg.header.seq++;
	    rmsg.header.stamp = msg->header.stamp;
	    rmsg.header.frame_id = "image";

		rmsg.object_type = result.sType;

		rmsg.center_x = result.sObjectRect.center.x;
		rmsg.center_y = result.sObjectRect.center.y;

		rmsg.size_x = result.sObjectRect.size.width;
		rmsg.size_y = result.sObjectRect.size.height;

		rmsg.img_width = lightCorrectedImage.cols;
		rmsg.img_height = lightCorrectedImage.rows;

		objectPublisher.publish(rmsg);
	}
}

void BlobDetection::detectWalls(cv::Mat& image, const ros::Time& timestamp) const {
    cv::Mat wall;
    cv::cvtColor(image, wall, CV_BGR2HSV);

    std::vector<cv::Mat> spl;
    cv::split(wall, spl);

    int element_size = 3;
    // cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * element_size + 1, 2 * element_size + 1), cv::Point(element_size, element_size) );
    // cv::morphologyEx(spl[1], spl[1], cv::MORPH_OPEN, element);
    cv::medianBlur(spl[1], spl[1], 13); // removes salt and pepper noise from saturation
    cv::GaussianBlur(spl[1], spl[1],cv::Size(7, 7), 7); // removes gaussian noise (well, weakens it) from saturation
    showImage(spl[1], "saturation");

    double satWeight = .25;
    n.getParam("satWeight", satWeight);

    wall = spl[1] * satWeight; // this can work against reflections
    wall += spl[2] * (1.0 - satWeight);

    cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(2 * element_size + 1, 2 * element_size + 1), cv::Point(element_size, element_size) );
    cv::morphologyEx(wall, wall, cv::MORPH_OPEN, element);
    cv::GaussianBlur(wall, wall, cv::Size(7, 7), 7);
    /*
       double minVal = 0.0, maxVal = 0.0; // maybe they come in handy some time ...
       minMaxLoc(wall, &minVal, &maxVal);
    */

    bool useNormalization = false;
    n.getParam("useNormalization", useNormalization);
    if(useNormalization)
        cv::normalize(wall, wall, 0, 255, NORM_MINMAX, CV_8UC1);

    showImage(wall, "wall");
    showHist(wall, "wall_hist");

    int upperBound = 50;
    n.getParam("upperBound", upperBound);

    Mat mask, maskValue, maskWall;

    cv::morphologyEx(spl[2], spl[2], cv::MORPH_OPEN, element);
    cv::GaussianBlur(spl[2], spl[2], cv::Size(7, 7), 7);
    showImage(spl[2], "value");
    showHist(spl[2], "value_hist");

    cv::inRange(spl[2], 0, upperBound, maskValue);
    //cv::morphologyEx(maskValue, maskValue, cv::MORPH_OPEN, element);
    showImage(maskValue, "value_mask");

    cv::inRange(wall, 0, upperBound, maskWall);
    //cv::morphologyEx(maskWall, maskWall, cv::MORPH_OPEN, element);
    showImage(maskWall, "wall_mask");

    cv::bitwise_or(maskValue, maskWall, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);

    double maxBlacknessScore = 0.0;
    int bestContourIndex = -1, secondBestContourIndex = -1;
    std::vector<std::vector<cv::Point> > contours; // Vector for storing contour

    cv::findContours(mask, contours, cv::noArray(), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
    std::vector<std::vector<cv::Point> > poligons(contours.size()); // Vector for storing the 2 best simplified contour


    double blacknessWeight = 1.0;
    n.getParam("blacknessWeight", blacknessWeight);

    double maxSlope = 0.3;
    n.getParam("maxSlope", maxSlope);

    double minLength = 0.7;
    n.getParam("minLength", minLength);
    cv::Mat tmp;

    for(size_t i = 0; i < contours.size(); i++) {// iterate through each contour.
        tmp = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::drawContours(tmp, contours, i, Scalar(255), CV_FILLED, 8); // draw the area in the contour completely white
        cv::subtract(tmp, spl[2] * blacknessWeight, tmp, tmp); // subtract the value channel from the white contour (the lighter the pixel the lower the score for it)
        double blacknessScore = cv::sum(tmp)[0]; // now sum over the score (the higher the black quality and the larger the area the "better" is the black surface.
        if(blacknessScore > maxBlacknessScore){
            cv::RotatedRect boundingBox = cv::minAreaRect(contours[i]);
            cv::approxPolyDP(contours[i], poligons[i], 20, true);
            bool wallCandidate = false;
            for(size_t j = 0; j < poligons[i].size() - 1; j++) {
            	float len = cv::norm(poligons[i][j] - poligons[i][j + 1]);
            	if(len > minLength * std::max(boundingBox.size.width, boundingBox.size.height)){
                	float m = fabs(static_cast<float>(poligons[i][j].y) - poligons[i][j + 1].y) / fabs(static_cast<float>(poligons[i][j].x) - poligons[i][j + 1].x);
            		if(maxSlope >= 0.0 && m > maxSlope)
            			continue;
#ifdef DEBUG
            		cv::Point2f rect_points[4];
            		boundingBox.points( rect_points );
            		//for( size_t k = 0; k < 4; k++ )
            			//line(tmp, rect_points[k], rect_points[(k + 1) % 4], Scalar(64), 2, 8, 0 );
            		cv::line(tmp, poligons[i][j], poligons[i][j + 1] , cv::Scalar(127), 7, 8, 0 );
#endif
            		wallCandidate = true;
            		break;
            	}
            }
            if(wallCandidate){
				// secondBestContourIndex = bestContourIndex;
				maxBlacknessScore = blacknessScore;
				bestContourIndex = static_cast<int>(i);
#ifdef DEBUG
		        cv::drawContours(tmp, poligons, i, Scalar(64), 2, 8 );
#endif
            }
#ifdef DEBUG
            // drawContours(tmp, outline, 0, Scalar(127), 2, 8 );
            // showImage(tmp, "tmp");
#endif
        }
    }
	showImage(tmp, "tmp");
    cv::Mat img = image.clone();
    if(bestContourIndex >= 0){
        object_detection_blob::BlobWall wallMessage;
        wallMessage.header.seq++;
        wallMessage.header.stamp = timestamp;
        wallMessage.header.frame_id = "image";
        auto contourMoments = cv::moments(poligons[bestContourIndex]);
        wallMessage.center.x = contourMoments.m10 / contourMoments.m00;
        wallMessage.center.y = contourMoments.m01 / contourMoments.m00;
        wallMessage.imgSize.x = img.cols;
        wallMessage.imgSize.y = img.rows;
        wallMessage.bottomCenter.x = wallMessage.center.x;
        wallMessage.bottomCenter.y = 0;
        cv::Point& lastPoint = poligons[bestContourIndex][0];
        for(auto& point : poligons[bestContourIndex]){
        	object_detection_blob::Vector2 element;
        	element.x = point.x;
        	element.y = point.y;
        	wallMessage.contour.push_back(element);
        	if(point.x <= wallMessage.center.x && wallMessage.center.x <= lastPoint.x){ // contour goes right to left and intersects with vertical line through center point
        		int distX = point.x - lastPoint.x;
        		int distY = lastPoint.y - point.y;
        		int bottomY = lastPoint.y + distY * static_cast<float>(point.x - wallMessage.center.x) / distX;
        		if(bottomY > wallMessage.bottomCenter.y)
        			wallMessage.bottomCenter.y = bottomY;
        	}
        	if(lastPoint.x <= wallMessage.center.x && wallMessage.center.x <= point.x){ // contour goes left to right and intersects with vertical line through center point
        		int distX = lastPoint.x - point.x;
        		int distY = point.y - lastPoint.y;
        		int bottomY = point.y + distY * static_cast<float>(point.x - wallMessage.center.x) / distX;
        		if(bottomY > wallMessage.bottomCenter.y)
        			wallMessage.bottomCenter.y = bottomY;
        	}
        	lastPoint = point;
        }
        wallMessage.relativeArea = cv::contourArea(contours[bestContourIndex]) / (img.cols * img.rows);
        if(wallMessage.center.x >= 0 && wallMessage.center.y >= 0) // the center is computed from the simplified polygon. if it is too simple (a line) there is no center of gravity for some reason and the shape is not worth publishing
        	wallPublisher.publish(wallMessage);
#ifdef DEBUG
        cv::drawContours(img, contours, bestContourIndex, Scalar(255, 0, 0), 2, 8 ); // Draw the best contour using previously stored index.
        cv::circle(img, cv::Point(contourMoments.m10 / contourMoments.m00, contourMoments.m01 / contourMoments.m00), 3,  Scalar(0, 255, 0), 2);
        cv::circle(img, cv::Point(wallMessage.bottomCenter.x, wallMessage.bottomCenter.y), 3,  Scalar(0, 0, 255), 2);
#endif
    }
    /*
    if(secondBestContourIndex >= 0){
        object_detection_blob::BlobWall wallMessage;
        wallMessage.header.seq++;
        wallMessage.header.stamp = timestamp;
        wallMessage.header.frame_id = "image";
        auto contourMoments = cv::moments(poligons[secondBestContourIndex]);
        wallMessage.center.x = contourMoments.m10 / contourMoments.m00;
        wallMessage.center.y = contourMoments.m01 / contourMoments.m00;
        wallMessage.imgSize.x = img.cols;
        wallMessage.imgSize.y = img.rows;
        wallMessage.bottomCenter.x = wallMessage.center.x;
        wallMessage.bottomCenter.y = 0;
        cv::Point& lastPoint = poligons[secondBestContourIndex][0];
        for(auto& point : poligons[secondBestContourIndex]){
            object_detection_blob::Vector2 element;
            element.x = point.x;
            element.y = point.y;
            wallMessage.contour.push_back(element);
            if(point.x <= wallMessage.center.x && wallMessage.center.x <= lastPoint.x){ // contour goes right to left and intersects with vertical line through center point
            	int distX = point.x - lastPoint.x;
            	int distY = lastPoint.y - point.y;
            	int bottomY = lastPoint.y + distY * static_cast<float>(point.x - wallMessage.center.x) / distX;
            	if(bottomY > wallMessage.bottomCenter.y)
            		wallMessage.bottomCenter.y = bottomY;
            }
            if(lastPoint.x <= wallMessage.center.x && wallMessage.center.x <= point.x){ // contour goes left to right and intersects with vertical line through center point
            	int distX = lastPoint.x - point.x;
            	int distY = point.y - lastPoint.y;
            	int bottomY = point.y + distY * static_cast<float>(point.x - wallMessage.center.x) / distX;
            	if(bottomY > wallMessage.bottomCenter.y)
            		wallMessage.bottomCenter.y = bottomY;
            }
        	lastPoint = point;
        }
        wallMessage.relativeArea = cv::contourArea(contours[secondBestContourIndex]) / (img.cols * img.rows);
        if(wallMessage.center.x >= 0 && wallMessage.center.y >= 0) // the center is computed from the simplified polygon. if it is too simple (a line) there is no center of gravity for some reason and the shape is not worth publishing
        	wallPublisher.publish(wallMessage);
#ifdef DEBUG
        cv::drawContours(img, contours, secondBestContourIndex, Scalar(255, 0, 0), 2, 8 ); // Draw the second best contour using previously stored index.
        cv::circle(img, cv::Point(contourMoments.m10 / contourMoments.m00, contourMoments.m01 / contourMoments.m00), 3,  Scalar(0, 255, 0), 2);
        cv::circle(img, cv::Point(wallMessage.bottomCenter.x, wallMessage.bottomCenter.y), 3,  Scalar(0, 0, 255), 2);
#endif
    }
    */
    showImage(img, "black");
}

void BlobDetection::showHist(cv::Mat& img, string name) const {
#ifdef DEBUG
/// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat hist;

  /// Compute the histograms:
  calcHist( &img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w / histSize );

  Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw hist
  for( int i = 1; i < histSize; i++ )
      line( histImage, Point( bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1)) ), Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ), Scalar( 255), 2, 8, 0  );

  int upperBound = 42;
  n.getParam("upperBound", upperBound);
  cv::line(histImage, cv::Point( bin_w * upperBound, 0), cv::Point( bin_w * upperBound, hist_h - 1), Scalar(127), 2, 8, 0 );

  /// Display
  namedWindow(name, CV_WINDOW_AUTOSIZE );
  imshow(name, histImage );
#endif
}


bool BlobDetection::setRangesCup(object_detection_blob::adjust_ranges::Request &req, object_detection_blob::adjust_ranges::Response &res) {
	setupColourRange(cv::Scalar(req.hue_min, req.sat_min, req.val_min), cv::Scalar(req.hue_max, req.sat_max, req.val_max), "cup");
	return true;
}

bool BlobDetection::setRangesBattery(object_detection_blob::adjust_ranges::Request &req, object_detection_blob::adjust_ranges::Response &res) {
	setupColourRange(cv::Scalar(req.hue_min, req.sat_min, req.val_min), cv::Scalar(req.hue_max, req.sat_max, req.val_max), "battery");
	return true;
}

bool BlobDetection::setRangesBase(object_detection_blob::adjust_ranges::Request &req, object_detection_blob::adjust_ranges::Response &res) {
	setupColourRange(cv::Scalar(req.hue_min, req.sat_min, req.val_min), cv::Scalar(req.hue_max, req.sat_max, req.val_max), "base");
	return true;
}

