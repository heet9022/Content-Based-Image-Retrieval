#include "helper.h"

Mat sobelX3x3(cv::Mat& src) {

    Mat dst(src.size(), CV_16SC3);

    int sobelX_hor[] = { -1, 0, 1 }; // 1x3
    int sobelX_ver[] = { 1,
                         2,
                         1 }; // 3x1

    int sumB = 0, sumG = 0, sumR = 0;

    Mat temp(src.size(), CV_16SC3);

    for (int x = 1; x < src.rows - 1; x++) {
        for (int y = 1; y < src.cols - 1; y++) {

            sumB = 0;
            sumG = 0;
            sumR = 0;

            for (int i = x - 1; i <= x + 1; i++) {

                Vec3b intensity = src.at<Vec3b>(i, y);
                int filter_point = sobelX_hor[i - (x - 1)];
                sumB += intensity[0] * filter_point;
                sumG += intensity[1] * filter_point;
                sumR += intensity[2] * filter_point;
            }

            temp.at<Vec3s>(x, y)[0] = sumB;
            temp.at<Vec3s>(x, y)[1] = sumG;
            temp.at<Vec3s>(x, y)[2] = sumR;
        }
    }

    for (int x = 1; x < temp.rows - 1; x++) {
        for (int y = 1; y < temp.cols - 1; y++) {

            sumB = 0;
            sumG = 0;
            sumR = 0;

            for (int j = y - 1; j <= y + 1; j++) {

                Vec3s intensity = temp.at<Vec3s>(x, j);
                int filter_point = sobelX_ver[j - (y - 1)];
                sumB += intensity[0] * filter_point;
                sumG += intensity[1] * filter_point;
                sumR += intensity[2] * filter_point;
            }

            dst.at<Vec3s>(x, y)[0] = abs(sumB / 4);
            dst.at<Vec3s>(x, y)[1] = abs(sumG / 4);
            dst.at<Vec3s>(x, y)[2] = abs(sumR / 4);
    
        }
    }

    return dst;
}

Mat sobelY3x3(cv::Mat& src) {

    Mat dst(src.size(), CV_16SC3);

    int sobelY_hor[] = { 1, 2, 1 }; // 1x3
    int sobelY_ver[] = { -1,
                          0,
                          1 }; // 3x1

    int sumB = 0, sumG = 0, sumR = 0;

    Mat temp(src.size(), CV_16SC3);
    for (int x = 1; x < src.rows - 1; x++) {
        for (int y = 1; y < src.cols - 1; y++) {

            sumB = 0;
            sumG = 0;
            sumR = 0;

            for (int i = x - 1; i <= x + 1; i++) {

                Vec3b intensity = src.at<Vec3b>(i, y);
                int filter_point = sobelY_hor[i - (x - 1)];
                sumB += intensity[0] * filter_point;
                sumG += intensity[1] * filter_point;
                sumR += intensity[2] * filter_point;
            }

            temp.at<Vec3s>(x, y)[0] = sumB / 4;
            temp.at<Vec3s>(x, y)[1] = sumG / 4;
            temp.at<Vec3s>(x, y)[2] = sumR / 4;

        }
    }

    for (int x = 1; x < temp.rows - 1; x++) {
        for (int y = 1; y < temp.cols - 1; y++) {

            sumB = 0;
            sumG = 0;
            sumR = 0;

            for (int j = y - 1; j <= y + 1; j++) {

                Vec3s intensity = temp.at<Vec3s>(x, j);
                int filter_point = sobelY_ver[j - (y - 1)];
                sumB += intensity[0] * filter_point;
                sumG += intensity[1] * filter_point;
                sumR += intensity[2] * filter_point;
            }

            dst.at<Vec3s>(x, y)[0] = abs(sumB);
            dst.at<Vec3s>(x, y)[1] = abs(sumG);
            dst.at<Vec3s>(x, y)[2] = abs(sumR);

        }
    }

    return dst;
}

Mat magnitude(cv::Mat& src) {

    //dst.convertTo(dst, CV_16UC3);
    Mat dst(src.size(), CV_16UC3);
    Mat sx = sobelX3x3(src);
    Mat sy = sobelY3x3(src);

    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {

            Vec3s x = sx.at<Vec3s>(i, j);
            Vec3s y = sy.at<Vec3s>(i, j);

            for (int c = 0; c < sx.channels(); c++) {
                dst.at<Vec3s>(i, j)[c] = sqrtf((x[c] * x[c]) + (y[c] * y[c]));
            }
        }
    }
    Mat scaled(src.size(), CV_8UC3);
    Mat grey;
    convertScaleAbs(dst, scaled, 1, 0);
    cv::cvtColor(scaled, grey, cv::COLOR_BGR2GRAY);
    //cout << "Depth: " << grey.depth() << endl;
    return grey;
}

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

void generate1DHistogramFeatures(Mat& image, vector<float>& features) {

    const int bin_size = 8;
    const int Hsize = 256 / bin_size;
    
    float hist1d[bin_size] = {};
    float normalizing_sum = image.rows * image.cols;

    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {

            int pixel = image.at<uchar>(x, y);
            int index = pixel / Hsize;
            hist1d[index] += 1;
        }
    }
    for (int i = 0; i < bin_size; i++) {
        float normalized_value = ((float)hist1d[i]) / normalizing_sum;
        features.push_back(normalized_value);
    }
}

void generateMultiHistogramFeatures(Mat& image, vector<float>& features) {

    const int bin_size = 8;
    const int Hsize = 256 / bin_size;
    int dim3[3] = { bin_size, bin_size, bin_size };
    Mat hist3d_upper = Mat::zeros(3, dim3, CV_32S);
    Mat hist3d_lower = Mat::zeros(3, dim3, CV_32S);
    float normalizing_sum = image.rows * image.cols / 2;

    for (int x = 0; x < image.rows / 2; x++) {
        for (int y = 0; y < image.cols; y++) {

            Vec3b pixel = image.at<Vec3b>(x, y);
            int b = pixel[0] / Hsize;
            int g = pixel[1] / Hsize;
            int r = pixel[2] / Hsize;
            hist3d_upper.at<int>(b, g, r) += 1;
        }
    }
    for (int x = image.rows / 2; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {

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

void generateColorTextureFeatures(Mat& image, vector<float>& features) {

    Mat texture = magnitude(image);
    generate1DHistogramFeatures(texture, features);
    generateHistogramFeatures(image, features);
}

void generateCustomFeature(Mat& image, vector<float>& features) {

    Mat texture = magnitude(image);
    generate1DHistogramFeatures(texture, features);
    generateHistogramFeatures(image, features);
}

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

        for (int i = 0; i < n; i++) {

            vector<float> featureReference = featuresData[i];
            vector<float> color, color_target;
            vector<float> texture, texture_target;
            for (int j = 0; j < 512; j++) {
                color.push_back(featureReference[j]);
                color_target.push_back(featureTarget[j]);
            }
            for (int j = 512; j < 520; j++) {
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
            vector<float> color, color_target;
            vector<float> texture, texture_target;
            for (int j = 0; j < 512; j++) {
                color.push_back(featureReference[j]);
                color_target.push_back(featureTarget[j]);
            }
            for (int j = 512; j < 520; j++) {
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
    return distances;
}
