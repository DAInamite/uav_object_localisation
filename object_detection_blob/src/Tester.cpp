#include <fstream>
#include <iostream>
#include <algorithm>
#include "Tester.h"
#include "object_recognition_cnn/Localize.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "BlobDetection.h"
#include "LightCorrection.h"

//using namespace cv;
using namespace std;
using namespace boost;
using namespace ros;
namespace fs = ::boost::filesystem;

Tester::Tester(const fs::path& imagepath, const int threshold, bool waitaftererror, const string& tagfilename) :
    mImagePath(imagepath),
    mTagFileName(tagfilename),
    mThreshold(threshold),
    mWaitAfterError(waitaftererror) {
    ROS_INFO("working in %s", fs::absolute(mImagePath).string().c_str());
}

Tester::~Tester() {

}

void Tester::test(ros::NodeHandle& nh) {
	try {
		// load existing index
        std::string indexPath(mImagePath.string());
        indexPath.append(mTagFileName);
        if (!fs::exists(mImagePath) || !fs::is_directory(mImagePath))
            throw string("Image directory does not exist: " + mImagePath.string());
		if (!fs::exists(indexPath) || !fs::is_regular_file(indexPath)){
			cout << "there is no index yet" << endl;
		} else {
			ifstream ifs(indexPath);
			ifs >> *this;
			ifs.close();
		}

		auto lcorrection = LightCorrection(LightCorrection::Naive);
		auto blob_detection = BlobDetection(nh, "imageTopic", "objectsTopic", "wallTopic", LightCorrection::None, cv::Scalar(100, 100, 100), cv::Scalar(130,255,255),
											cv::Scalar(18,50,50), cv::Scalar(38,255,255),
											cv::Scalar(170,50,50), cv::Scalar(185,255,255), false);

		// test for all the objects
		//const set<DetectionObjectTypes>& types = { DO_Base, DO_Battery, DO_Cup };
		// test for all tagged files
        size_t niceCount = 0, naughtyCount = 0, falsepositive = 0;
		bool quit = false;
		
        for(auto testObject : mTaggedFiles) {
            std::string filePath(mImagePath.string());
			filePath.append(testObject.getFileName());
			fs::path absolutePath = fs::absolute(filePath);

			auto image = cv::imread(absolutePath.string());
			showImage(image);
			cv::Mat corrected_image;
			lcorrection.correct(image, corrected_image);

			auto detected_objects = blob_detection.detectObjects(corrected_image);
			
			for (auto& object : detected_objects) {
				Coordinate actualPosition;

				switch(object.sType) {
					case DO_Base:
						actualPosition = testObject.getBasePos();
						break;
					case DO_Cup:
						actualPosition = testObject.getBeakerPos();
						break;
					case DO_Battery:
						actualPosition = testObject.getBatteryPos();
						break;
				}

				Coordinate detectedPosition(object.sObjectRect.center.x, object.sObjectRect.center.y);

				if(!actualPosition.isSet()) {
                    ++falsepositive;
					ROS_INFO("Found object of type %s which shouldn't exist! Filename: %s", objectNames[object.sType].c_str(), testObject.getFileName().c_str());
					quit = drawImage(testObject.getFileName(), actualPosition, detectedPosition);
					continue;
				}

				double dist = actualPosition.euclidianDistance(detectedPosition);

                ROS_INFO("Seen %s at (%d, %d) which is %s OK", objectNames[object.sType].c_str(), detectedPosition.getX(), detectedPosition.getY(), dist > mThreshold ? "NOT" : "");

                if (dist > mThreshold){
					naughtyCount++;
					quit = drawImage(testObject.getFileName(), actualPosition, detectedPosition);
				}
				else
					niceCount++;

				if (quit)
					break;
			}

			if (quit)
				break;
		}

        ROS_INFO("Finished tests: %zu nice, %zu naughty, %zu false positive.", niceCount, naughtyCount, falsepositive);
	}
	catch(string& s){
	    ROS_ERROR("%s", s.c_str());
	}
	catch(...) {
	    ROS_ERROR("Something really fucked up");
	}
}


bool Tester::drawImage(const string file, const Coordinate& actual, const Coordinate& detected) const {
    fs::path imagePath(mImagePath);
	Mat img = imread(imagePath.string() + file);
	circle(img, Point(actual.getX(), actual.getY()), 20, Scalar(0, 255, 0), 3);
	circle(img, Point(detected.getX(), detected.getY()), 20, Scalar(0, 0, 255), 3);
	return showImage(img);
}

bool Tester::showImage(Mat img) const {
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	imshow("image", img);
	cv::waitKey(0);  

    if (mWaitAfterError)
        return cin.get() == 'q';
    else
        return false;
}
