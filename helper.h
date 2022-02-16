#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <iostream>
#include <filesystem>
#include<cmath>

#include "csv_util.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class helper
{
};

class Distance {

public:
	char* filename;
	float dist;
};

int saveFeatures(string dir, string featureName);
vector<float> generateFeatures(Mat& image, string featureName);
int readFeatures(string filename, vector<char*>& fileNames, vector<vector<float>>& imageData);
vector<Distance> computeDistance(vector<char*> fileNames, vector<vector<float>> featuresData, vector<float> featureTarget, string featureName);
float sumOfSquaredDifference(vector<float>& A, vector<float>& B);
float histogramIntersection(vector<float>& A, vector<float>& B);
