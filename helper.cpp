#include "helper.h"
#include "filter.h"
#include "features.h"


vector<float> generateFeatures(Mat& image, string featureName) {

    vector<float> features;

    if (featureName == "baseline")
        generateBaselineFeatures(image, features);
    else if (featureName == "hist_matching")
        generateHistogramFeatures(image, features);
    else if (featureName == "multi-hist_matching")
        generateMultiHistogramFeatures(image, features);
    else if (featureName == "color-texture_matching")
        generateColorTextureFeatures(image, features);
    else if (featureName == "custom_matching")
        generateCustomFeature(image, features);
    
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

        //if (counter == 20)
        //    break;

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
    else if (featureName == "color-texture_matching") {

        int texture_feature_size = 8*9;

        for (int i = 0; i < n; i++) {

            vector<float> featureReference = featuresData[i];
            vector<float> color, color_target;
            vector<float> texture, texture_target;
            for (int j = 0; j < 512; j++) {
                color.push_back(featureReference[j]);
                color_target.push_back(featureTarget[j]);
            }
            for (int j = 512; j < featureReference.size(); j++) {
                texture.push_back(featureReference[j]);
                texture_target.push_back(featureTarget[j]);
            }
            Distance distance;
            float color_dist = histogramIntersection(color, color_target);
            float texture_dist = histogramIntersection(texture, texture_target);
            distance.dist = 0.5 * color_dist + 0.5 * texture_dist;
            distance.filename = fileNames[i];
            distances.push_back(distance);
        }
    }

    else if (featureName == "custom_matching") {

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
