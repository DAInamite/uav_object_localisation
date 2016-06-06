#pragma once


#include "util.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <type_traits>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tiny_cnn {

using namespace boost;
namespace fs = ::boost::filesystem;

#ifndef NETSIZE_X
#define NETSIZE_X 80
#endif
#ifndef NETSIZE_Y
#define NETSIZE_Y ((NETSIZE_X * 3 / 4))
#endif


typedef enum{
	bgr,
	hsv,
	yuv,
	ycrcb
} colour;



// input type (of pic) is assumed to be BGR
template <colour N>
vec_t mat2vec_t(cv::Mat pic, const std::vector<size_t>& channels = {0, 1, 2}) {
	cv::resize(pic, pic, cv::Size(NETSIZE_X, NETSIZE_Y), 0.0f, 0.0f, cv::INTER_LANCZOS4);
	std::vector<cv::Mat> split_image;
	vec_t vec;
	vec.resize(NETSIZE_X * NETSIZE_Y * channels.size(), 0.0);
	cv::Mat conv_image;
	if(N == hsv)
		cv::cvtColor(pic, conv_image, CV_BGR2HSV);
	else if(N == yuv)
		cv::cvtColor(pic, conv_image, CV_BGR2YUV);
	else if(N == ycrcb)
		cv::cvtColor(pic, conv_image, CV_BGR2YCrCb);
	else
		conv_image = pic;
	cv::split(conv_image, split_image);

	size_t i = 0; // this is the channel's position in vec_t (all channels right after each other even when you only want channel 0 and 2 of the split image)
	for(auto channel : channels){
	    for (int y = 0; y < NETSIZE_Y; ++y){
	        for (int x = 0; x < NETSIZE_X; ++x){
                vec[i * NETSIZE_X * NETSIZE_Y + NETSIZE_X * y + x] = (split_image[channel].at<uint8_t>(y, x) / (float_t) 255.0) * 2.0 - 1.0;
	        }
	    }
	    ++i;
	}
	return vec;
}

// be careful, opencv can't do everything with CV_64FC3 matrices, e.g it can't cvtColor them for some reason.
template <colour N>
cv::Mat vec_t2mat(const vec_t& vec, const std::vector<size_t>& channels = {0, 1, 2}) {
	std::vector<cv::Mat> split_image;
	cv::Mat image(NETSIZE_Y, NETSIZE_X, CV_64FC3);
	for(size_t channel = 0; channel < 3; channel++){
	    auto channelPos = find(channels.begin(), channels.end(), channel);
	    if(channelPos != channels.end()){
	        split_image.push_back(cv::Mat(NETSIZE_Y, NETSIZE_X, CV_64FC1, const_cast<float_t*>(vec.data() + distance(channels.begin(), channelPos) * NETSIZE_X * NETSIZE_Y)).clone());
	        split_image[channel] += 1;  // normalize back to
	        split_image[channel] *= .5; // [0 .. 1] range
	    } else {
	        split_image.push_back(cv::Mat(NETSIZE_Y, NETSIZE_X, CV_64FC1, cv::Scalar(.5)));
	    }
	}
	merge(split_image, image);
	return image;
}

template <colour N>
vec_t load_image(const std::string& image_file, const std::vector<size_t>& channels = {0, 1, 2}) {
	cv::Mat pic = cv::imread(image_file);
	return mat2vec_t<N>(pic, channels);
}

template <colour N>
void load_images(const fs::path& path_to_images, std::vector<vec_t>& images, const size_t max_count = 1000,
        const std::vector<size_t>& channels = {0, 1, 2},
		const std::string& ext = ".png") {
	// complain about stuff
	if (!fs::exists(path_to_images) || !fs::is_directory(path_to_images))
		throw std::string("image directory does not exist: " + path_to_images.string());

	// do work
	fs::directory_iterator it(path_to_images);
	fs::directory_iterator endit;
	std::vector<std::string> filenames; // 2-loop approach for random shuffle
	while(it != endit) {
		if (fs::is_regular_file(*it) and it->path().extension() == ext)
			filenames.push_back(it->path().string());
		++it;
	}
	std::random_shuffle(filenames.begin(), filenames.end());
	size_t count = 0;
	for(auto& filename : filenames){
	    vec_t image1 = load_image<N>(filename, channels);
//	    vec_t image2 = load_image<hsv>(filename, {0});
//	    vec_t image3 = load_image<yuv>(filename, {1, 2});
//	    image1.insert(image1.end(), image2.begin(), image2.end());
//	    image1.insert(image1.end(), image3.begin(), image3.end());
	    images.push_back(image1);
	    if(++count >= max_count)
	        break;
	}
}

// WARNING: templates do not return same cv::Mat type. rgb argument results in 64FC3, the others produce 8UC3. All output BGR color order for imgshow()
template <colour N>
cv::Mat vec_t2bgrMat(const vec_t& vec, const std::vector<size_t>& channels = {0, 1, 2}){
	if(N == bgr)
		return vec_t2mat<N>(vec, channels);
    cv::Mat bgr_UC8;
	cv::Mat conv_64FC3 = vec_t2mat<N>(vec, channels);
	cv::Mat conv_8UC3;
	conv_64FC3.convertTo(conv_8UC3, CV_8U, 255);
	if(N == hsv)
		cv::cvtColor(conv_8UC3, bgr_UC8, CV_HSV2BGR);
	else if(N == yuv)
		cv::cvtColor(conv_8UC3, bgr_UC8, CV_YUV2BGR);
	else // image is YCrCb
		cv::cvtColor(conv_8UC3, bgr_UC8, CV_YCrCb2BGR);
	return bgr_UC8;
}

}
