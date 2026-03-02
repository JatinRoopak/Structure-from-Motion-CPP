#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <utils.hpp>
#include <feature_extractor.hpp>

int main() {
    std::string img_path = "../assets/building.jpg";
    cv::Mat cv_img = cv::imread(img_path, cv::IMREAD_COLOR);

    if (cv_img.empty()){
        std::cerr << "Error in loading image " << img_path << std::endl;
        return -1;
    }

    Image greyscale_img = Grayscale(cv_img);

    FeatureExtractor extractor;
    float threshold = 200000000.0f;

    std::vector<Keypoint> corners = extractor.Harriscorner(greyscale_img, threshold);

    cv::Mat out_img = cv_img.clone();

    for (const auto& kp : corners) { //draw circle on corner
        cv::circle(out_img, cv::Point(kp.x, kp.y), 10, cv::Scalar(0, 0, 255), -1);
    }

    cv::imwrite("output_corners.jpg", out_img);
    std::cout << "Saved visualization as output_corners.jpg" << std::endl;

    return 0;
}