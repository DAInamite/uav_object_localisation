/*
 * Tagger.h
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include "Tag.h"
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

#ifndef TESTER_H_
#define TESTER_H_

using namespace cv;

static const std::string objectNames[] = { "base", "battery", "cup" }; // too lazy for switch case when writing names -> look-up table

class Tester
{
public:
    /**
     * @brief Tester constructor
     * @param imagepath         The path to the tagged images.
     * @param threshold         The threshold until which dinstance the center is rated as correct.
     * @param waitaftererror    If true and after an error is detected the program waits until a key is pressed.
     * @param tagfilename       The name of the file with the tagged image information.
     */
    Tester(const boost::filesystem::path& imagepath, const int threshold = 84,
           bool waitaftererror = false, const std::string& tagfilename = "index.txt");
	~Tester();

	template <typename Char, typename CharTraits>
    friend std::basic_ostream<Char, CharTraits>& operator<< (std::basic_ostream<Char, CharTraits>& os, const Tester& t) {
        for(auto& item : t.mTaggedFiles){
			os << item << std::endl;
		}
		return os;
	}

	template <typename Char, typename CharTraits>
    friend std::basic_istream<Char, CharTraits>& operator>> (std::basic_istream<Char, CharTraits>& is, Tester& t) {
		std::string line;
        while(getline(is, line)) {
            if(line.length() > 0) {
				std::stringstream strstr(line);
				Tag tag;
				strstr >> tag;
                t.mTaggedFiles.insert(tag);
			}
		}
		return is;
	}

    //! Tests all image in the path with the blob detection
	void test(ros::NodeHandle& nh);

private:
    //! Shows an image in which the detected and actual position is drawn
	bool drawImage(const std::string file, const Coordinate& actual, const Coordinate& detected) const;
	//! Shows an image
	bool showImage(Mat img) const;
    //! All tagged images in the image path
    std::set<Tag> mTaggedFiles;
    //! The path to the tagged images
    boost::filesystem::path mImagePath;
    //! The name of the file with the tagged image information.
    std::string mTagFileName;
    //! The threshold until which the center is rated as correct
    int mThreshold;
    //! Wait for user input after wrong detection?
    bool mWaitAfterError;
};

#endif /* TESTER_H_ */
