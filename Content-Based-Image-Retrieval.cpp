#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#include "helper.h"

int task(string target_file_name, string featureName, int n) {

	fs::path target_path = fs::current_path();
	string dir = "olympus";
	target_path = (target_path/dir)/target_file_name;
	Mat target = imread(target_path.string(), IMREAD_COLOR);

	string database = "_feature_database.csv";

	saveFeatures(dir, featureName);
	vector<char*> fileNames;
	vector<vector<float>> featuresData;
	readFeatures(featureName+database, fileNames, featuresData);
	vector<float> featureTarget = generateFeatures(target, featureName);
	vector<Distance> distances = computeDistance(fileNames, featuresData, featureTarget, featureName);
	
	sort(distances.begin(), distances.end(), 
		[](Distance const& a, Distance const& b) {
			return a.dist < b.dist;
		});

	for (int i = 1; i < n+1; i++) {

		fs::path img_path = ((fs::current_path() / dir) / string(distances[i].filename));
		imshow("results", imread(img_path.string(), IMREAD_COLOR));
		waitKey(0);
	}
	return 0;
}

int main() {

	int task_number = 1;
	switch (task_number) {
		case 1: {
			task("pic.1016.jpg", "baseline", 3);
			break;
		}
		case 2: {
			task("pic.0164.jpg", "hist_matching", 3);
			break;
		}
		case 3: {
			task("pic.0274.jpg", "multi-hist_matching", 3);
			break;
		}
	}
	return 0;
}
