#include "LightCorrection.h"

LightCorrection::LightCorrection(CorrectionMode mode) :
	mMode(mode)
{
}

void LightCorrection::correct(const cv::Mat &input, cv::Mat &output)
{
	switch(mMode)
	{
		case Naive:
			naiveCorrection(input, output);
			cv::cvtColor(output, output, CV_BGR2HSV);
			break;
		case Local:
			localCorrection(input, output);
			break;
		case Global:
			globalCorrection(input, output);
			break;
		case None:
			cv::cvtColor(input, output, CV_BGR2HSV);
			break;

	}
}

void LightCorrection::naiveCorrection(const cv::Mat &input, cv::Mat &output)
{
	cv::Mat tmp = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

	for(int x = 0; x < input.rows; ++x)
	{
		for(int y = 0; y < input.cols; ++y)
		{
            // sum up all three channels
			unsigned int total = input.data[input.step * x + (y * 3)] + input.data[input.step * x + (y * 3) + 1] + input.data[input.step * x + (y * 3) + 2];
            // the ratio of each channel is used to determine the new value of the channel
			tmp.data[tmp.step * x + (y * 3)] = float(input.data[input.step * x + (y * 3)]) / total * 255.0;
			tmp.data[tmp.step * x + (y * 3) + 1] = float(input.data[input.step * x + (y * 3) + 1]) / total * 255.0;
			tmp.data[tmp.step * x + (y * 3) + 2] = float(input.data[input.step * x + (y * 3) + 2]) / total * 255.0;
		}
	}

	output = tmp;
}

void LightCorrection::localCorrection(const cv::Mat &input, cv::Mat &output)
{
	cv::Mat input_image;
	cv::cvtColor(input, input_image, CV_BGR2HSV);

	// Extract the V channel
	std::vector<cv::Mat> lab_planes(3);
	cv::split(input_image, lab_planes);

	// apply the CLAHE algorithm to the V channel
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat dstc;
	clahe->apply(lab_planes[2], dstc);

	// Merge the the color planes back into an Lab image
	dstc.copyTo(lab_planes[2]);
	cv::merge(lab_planes, input_image);

	output = input_image;
}

void LightCorrection::globalCorrection(const cv::Mat &input, cv::Mat &output)
{
	cv::Mat input_image;
	cv::cvtColor(input, input_image, CV_BGR2HSV);

	// Extract the V channel
	std::vector<cv::Mat> lab_planes(3);
	cv::split(input_image, lab_planes);

	// apply the CLAHE algorithm to the V channel
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat dstc;
	equalizeHist(lab_planes[2], dstc);

	// Merge the the color planes back into an Lab image
	dstc.copyTo(lab_planes[2]);
	cv::merge(lab_planes, input_image);

	output = input_image;
}


void LightCorrection::showImage(cv::Mat img, std::string name) const {
	cv::namedWindow(name, cv::WINDOW_NORMAL);
	cv::imshow(name, img);
	cv::waitKey(1);   
}
