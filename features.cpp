#include "features.h"
#include "filter.h"
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

void generate2DHistogramFeatures(Mat& mag, Mat& ang, vector<float>& features) {

    const int bin_size_mag = 8;
    const int Hsize_mag = 256 / bin_size_mag;

    const int bin_size_ang = 9;
    const int Hsize_ang = 180 / bin_size_ang;

    Mat hist2d = Mat::zeros(bin_size_mag, bin_size_ang, CV_32S);
    float normalizing_sum = mag.rows * mag.cols;

    for (int x = 0; x < mag.rows; x++) {
        for (int y = 0; y < mag.cols; y++) {

            int pixel_mag = mag.at<uchar>(x, y);
            int mag_idx = pixel_mag / Hsize_mag;

            int pixel_ang = ang.at<float>(x, y);
            int ang_idx = pixel_ang / Hsize_ang;

                hist2d.at<int>(mag_idx, ang_idx) += 1;
        }
    }
    for (int i = 0; i < bin_size_mag; i++) {
        for (int j = 0; j < bin_size_ang; j++) {

            float normalized_value = ((float)hist2d.at<int>(i, j)) / normalizing_sum;
            features.push_back(normalized_value);  
        }
    }
}

void generateHSVHistogramFeatures(Mat& image, vector<float>& features) {

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

//void generateGaborFeatures(cv::Mat& image, vector<float>& features) {
//
//    Mat grey, resized, src_f;
//    cvtColor(image, grey, COLOR_BGR2GRAY);
//    resize(grey, resized, Size(66, 96), 0, 0, INTER_NEAREST);
//    resized.convertTo(src_f, CV_32F);
//
//    GaborFeature gFeature;
//    std::vector<cv::Mat> gaborFeatures = gFeature.getFeature(src_f);
//
//    //std::vector<double> feature;
//
//    for (int x = 0; x < gaborFeatures.size(); x++)
//    {
//        cv::Mat gaborMat = gaborFeatures[x];
//        for (int y = 0; y < gaborMat.rows; y++)
//        {
//            const double* My = gaborMat.ptr<double>(y);
//            for (int z = 0; z < gaborMat.cols; z++)
//            {
//                features.push_back(My[z]);
//            } // end of z-loop
//        } // end of y-loop
//    } // end of x-loop


//}
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

    //Mat gray;
    //cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    //Mat sx = sobelX3x3(gray);
    //Mat sy = sobelY3x3(gray);
    //Mat mag = magnitude(sx, sy, gray);
    //Mat angle = orientation(sx, sy, gray);
    //Mat dst;
    /*for (int i = 1; i < 31; i = i + 2)
    {
        bilateralFilter(image, dst, i, i * 2, i / 2);
 
    }*/
    
    generateHistogramFeatures(image, features);
    Mat g = gaborFilter(image);
    //generateGaborFeatures(dst, features);
    generate1DHistogramFeatures(g, features);
}

void generateCustomFeature(Mat& image, vector<float>& features) {

    //Mat texture = magnitude(image);
    //generate1DHistogramFeatures(texture, features);
    
    Mat fullImageHSV, dst;
    for (int i = 1; i < 31; i = i + 2)
    {
        bilateralFilter(image, dst, i, i * 2, i / 2);

    }
    cvtColor(dst, fullImageHSV, cv::COLOR_BGR2HSV);
    generateHSVHistogramFeatures(fullImageHSV, features);
}