#include "helper.h"

void generateBaselineFeatures(Mat& image, vector<float>& features) {

    int xx = image.rows / 2 - 4;
    int yy = image.cols / 2 - 4;

    for (int x = xx; x <= xx + 8; x++) {
        for (int y = yy; y <= yy + 8; y++) {
            Vec3b pixel = image.at<Vec3b>(x, y);
            for (int c = 0; c < image.channels(); c++) {
                features.push_back(pixel[c]);
            }
        }
    }

}

void generateHistogramFeatures(Mat& image, vector<float>& features) {

    const int bin_size = 8;
    const int Hsize = 256 / bin_size;
    int dim3[3] = { bin_size, bin_size, bin_size };
    Mat hist3d = Mat::zeros(3, dim3, CV_32S);
    float normalizing_sum = image.rows * image.cols;

    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {

            Vec3b pixel = image.at<Vec3b>(x, y);
            int b = pixel[0] / Hsize;
            int g = pixel[1] / Hsize;
            int r = pixel[2] / Hsize;
            hist3d.at<int>(b, g, r) += 1;
        }
    }
    for (int i = 0; i < bin_size; i++) {
        for (int j = 0; j < bin_size; j++) {
            for (int k = 0; k < bin_size; k++) {

                float normalized_value = ((float)hist3d.at<int>(i, j, k)) / normalizing_sum;
                features.push_back(normalized_value);
            }
        }
    }
}

void generateMultiHistogramFeatures(Mat& image, vector<float>& features) {

    //Mat upperImage = image(Rect(0,0, image.cols-1, (image.rows/2) - 1));
    //Mat lowerImage = image(Rect(image.rows / 2, 0, image.cols - 1, (image.rows / 2) -1));
    //generateHistogramFeatures(upperImage, features);
    //generateHistogramFeatures(lowerImage, features);

    const int bin_size = 8;
    const int Hsize = 256 / bin_size;
    int dim3[3] = { bin_size, bin_size, bin_size };
    Mat hist3d_upper = Mat::zeros(3, dim3, CV_32S);
    Mat hist3d_lower = Mat::zeros(3, dim3, CV_32S);
    float normalizing_sum = image.rows * image.cols / 4;

    for (int x = 0; x < image.rows / 2; x++) {
        for (int y = 0; y < image.cols / 2; y++) {

            Vec3b pixel = image.at<Vec3b>(x, y);
            int b = pixel[0] / Hsize;
            int g = pixel[1] / Hsize;
            int r = pixel[2] / Hsize;
            hist3d_upper.at<int>(b, g, r) += 1;
        }
    }
    for (int x = image.rows / 2; x < image.rows; x++) {
        for (int y = image.cols / 2; y < image.cols; y++) {

            Vec3b pixel = image.at<Vec3b>(x, y);
            int b = pixel[0] / Hsize;
            int g = pixel[1] / Hsize;
            int r = pixel[2] / Hsize;
            hist3d_lower.at<int>(b, g, r) += 1;
        }
    }

    for (int i = 0; i < bin_size; i++) {
        for (int j = 0; j < bin_size; j++) {
            for (int k = 0; k < bin_size; k++) {

                float normalized_value = ((float)hist3d_upper.at<int>(i, j, k)) / normalizing_sum;
                features.push_back(normalized_value);
            }
        }
    }
    for (int i = 0; i < bin_size; i++) {
        for (int j = 0; j < bin_size; j++) {
            for (int k = 0; k < bin_size; k++) {

                float normalized_value = ((float)hist3d_lower.at<int>(i, j, k)) / normalizing_sum;
                features.push_back(normalized_value);
            }
        }
    }
}

vector<float> generateFeatures(Mat& image, string featureName) {

    vector<float> features;

    if (featureName == "baseline")
        generateBaselineFeatures(image, features);
    else if (featureName == "hist_matching")
        generateHistogramFeatures(image, features);
    else if (featureName == "multi-hist_matching") 
        generateMultiHistogramFeatures(image, features);
   
    return features;
}

int saveFeatures(string dir, string featureName) {

    fs::path imgDir = fs::current_path();
    imgDir /= dir;

    string fileName = featureName + "_feature_database.csv";
    char* fileName_char = new char[fileName.length() + 1];
    strcpy(fileName_char, fileName.c_str());

    int counter = 0;
    for (const auto& entry : fs::directory_iterator(imgDir)) {

        fs::path path = entry.path();

        if (path.extension() == ".jpg" || path.extension() == ".png" || path.extension() == ".jpeg") {
            Mat image = imread(path.string(), IMREAD_COLOR);
            vector<float> features = generateFeatures(image, featureName);
            char* imgName_char = new char[path.filename().string().length() + 1];
            strcpy(imgName_char, path.filename().string().c_str());
            append_image_data_csv(fileName_char, imgName_char, features, 0);
            counter++; 
        }
    }
    std::cout << counter << " " + featureName << " features saved to csv" << endl;
    return 0;
}

int readFeatures(string filename, vector<char*>& fileNames, vector<vector<float>>& imageData) {

    char* fileName_char = new char[filename.length() + 1];
    strcpy(fileName_char, filename.c_str());
    read_image_data_csv(fileName_char, fileNames, imageData);
    return 0;
}

float sumOfSquaredDifference(vector<float>& A, vector<float>& B) {

    float sum = 0.0;
    for (int j = 0; j < A.size(); j++)
        sum += pow(A[j] - B[j], 2);
    return sum/A.size();

}

float histogramIntersection(vector<float>& A, vector<float>& B) {

    float sum_hist1 = 0.0;
    float sum_hist2 = 0.0;
    float sum = 0.0;

    for (int j = 0; j < A.size(); j++) {

        sum += min(A[j], B[j]);
        sum_hist1 += A[j];
        sum_hist2 += B[j];
    }
    return (1 - (sum / (sum_hist1 * sum_hist2)));

}

vector<Distance> computeDistance(vector<char*> fileNames, vector<vector<float>> featuresData, vector<float> featureTarget, string featureName) {

    int n = featuresData.size();
    vector<Distance> distances;

    if (featureName == "baseline") {

        for (int i = 0; i < n; i++) {

            vector<float> featureReference = featuresData[i];
            Distance distance;
            distance.dist = sumOfSquaredDifference(featureReference, featureTarget);
            distance.filename = fileNames[i];
            distances.push_back(distance);
        }
    }
    else if (featureName == "hist_matching") {

        for (int i = 0; i < n; i++) {

            vector<float> featureReference = featuresData[i];
            Distance distance;
            distance.dist = histogramIntersection(featureReference, featureTarget);
            distance.filename = fileNames[i];
            distances.push_back(distance);
        }
    }
    else if (featureName == "multi-hist_matching") {

        for (int i = 0; i < n; i++) {

            vector<float> featureReference = featuresData[i];
            Distance distance;
            distance.dist = histogramIntersection(featureReference, featureTarget);
            distance.filename = fileNames[i];
            distances.push_back(distance);
        }
    }
    return distances;
}
