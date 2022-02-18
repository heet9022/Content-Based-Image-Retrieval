#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class filter
{
};

Mat sobelX3x3(cv::Mat& src);

Mat sobelY3x3(cv::Mat& src);

Mat magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& src);

Mat orientation(cv::Mat& sx, cv::Mat& sy, cv::Mat& src);

Mat gaborFilter(cv::Mat outImg);