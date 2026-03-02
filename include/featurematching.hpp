#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class FeatureMatcher {
    private:
        cv::Ptr<cv::SIFT> detector;
        cv::Ptr<cv::DescriptorMatcher> matcher;

    public:
        FeatureMatcher();

        void extractandmatch(const cv::Mat& img1, const cv::Mat& img2,
        std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
        cv::Mat& desc1, cv::Mat& desc2,
        std::vector<cv::DMatch>& good_matches);
};