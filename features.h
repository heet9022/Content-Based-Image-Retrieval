#pragma once
#include "helper.h"
//#include "gaborfeature.h"

class features
{
};

void generateBaselineFeatures(Mat& image, vector<float>& features);

void generateHistogramFeatures(Mat& image, vector<float>& features);

void generate1DHistogramFeatures(Mat& image, vector<float>& features);

void generateMultiHistogramFeatures(Mat& image, vector<float>& features);

void generateColorTextureFeatures(Mat& image, vector<float>& features);

void generateCustomFeature(Mat& image, vector<float>& features);
