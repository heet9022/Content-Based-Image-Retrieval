#include "readImages.h"
#include "csv_util.h"

int generateFeature(std::filesystem::path path);

int man() {

    std::filesystem::path imgDir = fs::current_path();
    imgDir /= "olympus";

    std::vector<std::filesystem::path> trainSet;
    int counter = 1;
    for (const auto& entry : fs::directory_iterator(imgDir)) {

        std::filesystem::path path = entry.path();

        if (path.extension() == ".jpg" || path.extension() == ".png" || path.extension() == ".jpeg") {
            generateFeature(path);
            trainSet.push_back(path);
            
            counter++;
        }
    }
    return 0;
}

int generateFeature(std::filesystem::path path) {

    cv::Mat image = cv::imread(path.u8string(), cv::IMREAD_COLOR);
    int xCenter = image.rows / 2;
    int yCenter = image.cols / 2;
    int xStart = xCenter - 4;
    int yStart = yCenter - 4;
    cv::Rect ROI(xStart,yStart,9,9);
    cv::Mat croppedImage = image(ROI);
    std::vector<float> features;

    for (int i = 0; i < croppedImage.rows; i++) {
        for (int j = 0; j < croppedImage.cols; j++) {
            cv::Vec3b intensity = croppedImage.at<Vec3b>(i, j);
            for (int c = 0; c < 3; c++)
                features.push_back(intensity[c]);
        }
    }

    char fileName[21] = "feature_database.csv";
    char* imgName = new char[path.filename().string().length() + 1];
    strcpy(imgName, path.string().c_str());
    append_image_data_csv(fileName, imgName, features, 0);
    return 0;

}
