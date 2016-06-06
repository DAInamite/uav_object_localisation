/*
 * Tagger.cpp
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */
#include <fstream>
#include <iostream>
#include <algorithm>
#include "Tester.h"
#include <ros/ros.h>
#include "object_recognition_cnn/Localize.h"
#ifndef HEADLESS
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif
using namespace std;
using namespace boost;
using namespace ros;
namespace fs = ::boost::filesystem;

Tester::Tester(const fs::path& _path, const int _threshold, const string& _fileName) :
		path(_path),
		fileName(_fileName),
		threshold(_threshold){
	ROS_INFO("working in %s", fs::absolute(path).string().c_str());
}

Tester::~Tester() {

}

void Tester::test(bool interactive) {
	try {
		// load existing index
		std::string indexPath(path.string());
		indexPath.append(fileName);
		if (!fs::exists(path) || !fs::is_directory(path))
			throw string("Image directory does not exist: " + path.string());
		if (!fs::exists(indexPath) || !fs::is_regular_file(indexPath)){
			cout << "there is no index yet at " << indexPath << endl;
		} else {
			ifstream ifs(indexPath);
			ifs >> *this;
			ifs.close();
		}
		// begin the test using a service client
		NodeHandle nh;
		ServiceClient client = nh.serviceClient<object_recognition_cnn::Localize>("localize");
		// test for all the objects
		const set<objectsEnum>& types = {BASE, BATTERY, BEAKER};
		// test for all tagged files
		size_t niceCount = 0, naughtyCount = 0;
		for(auto testObject : taggedFiles){
		    // test for all possible objects
		    bool quit = false;
		    for(auto type: types){
		        Coordinate actualPosition;
		        switch(type){
		        case BASE:
		            actualPosition = testObject.getBasePos();
		            break;
		        case BATTERY:
		            actualPosition = testObject.getBatteryPos();
		            break;
		        case BEAKER:
		            actualPosition = testObject.getBeakerPos();
		            break;
		        }
		        // only continue if the object is actually in the picture
		        if(!actualPosition.isSet())
		            continue;
		        // make a service client
	            object_recognition_cnn::Localize srv;
		        // check whether the network detects the object at the same time
				std::string filePath(path.string());
	            filePath.append(testObject.getFileName());
	            fs::path absolutePath = fs::absolute(filePath);
	            ROS_INFO("Path: %s", absolutePath.string().c_str());
		        srv.request.filename = absolutePath.string();
		        srv.request.object_type = type;
		        srv.request.scale = .6;
		        srv.request.offsetX = .33;
                srv.request.offsetY = .33;
                srv.request.interactive = false;
		        if (client.call(srv)) {
		            Coordinate detectedPosition ((double) srv.response.posX, (double) srv.response.posY);
		            double dist = actualPosition.euclidianDistance(detectedPosition);
		            if (dist > threshold){
		                naughtyCount++;
#ifndef HEADLESS

		                drawImage(testObject.getFileName(), actualPosition, detectedPosition);
		                if(interactive){
		                    if(cin.get() == 'q')
		                        quit = true;
		                }
#endif
		            }
		            else
		                niceCount++;
                    ROS_INFO("Seen %s at (%d, %d) which is %s OK", objectNames[type].c_str(), detectedPosition.getX(), detectedPosition.getY(), dist > threshold ? "NOT" : "");
		        }
		        else {
		            ROS_ERROR("Failed to call service add_two_ints");
		        }
		    }
		    if(quit)
		        break;
		}
		ROS_INFO("Finished tests: %zu nice, %zu naughty", niceCount, naughtyCount);
	}
	catch(string& s){
	    ROS_ERROR("%s", s.c_str());
	}
	catch(...) {
	    ROS_ERROR("Something really fucked up");
	}
}

#ifndef HEADLESS
void Tester::drawImage(const string file, const Coordinate& actual, const Coordinate& detected) const {
	fs::path imagePath(path);
	Mat img = imread(imagePath.string() + file);
	circle(img, Point(actual.getX(), actual.getY()), 20, Scalar(0, 255, 0), 3);
	circle(img, Point(detected.getX(), detected.getY()), 20, Scalar(0, 0, 255), 3);
	imshow("image", img);
}
#endif
