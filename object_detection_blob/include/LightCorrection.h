#ifndef LIGHTCORRECTION_H_
#define LIGHTCORRECTION_H_

#include "opencv2/opencv.hpp"

class LightCorrection
{
public:
	enum CorrectionMode
	{
		Naive,
		Local,
		Global,
		None
	};

	LightCorrection(CorrectionMode mode = Naive);

	void setMode(CorrectionMode mode) { mMode = mode; }

    //! Corrects the given image with the current correction mode and returns the result in the output image.
	void correct(const cv::Mat &input, cv::Mat &output);

private:
	void naiveCorrection(const cv::Mat &input, cv::Mat &output);
	void localCorrection(const cv::Mat &input, cv::Mat &output);
	void globalCorrection(const cv::Mat &input, cv::Mat &output);
    //! The active correction mode
	CorrectionMode mMode;
	// TEST
	void showImage(cv::Mat img, std::string name) const;
};

#endif //#ifndef LIGHTCORRECTION_H_
