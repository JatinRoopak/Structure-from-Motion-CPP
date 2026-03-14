#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PnP_Estimator{
    public:
        static bool estimatePose(
            const std::vector<cv::Point3f>& point3d,
            const std::vector<cv::Point2f>& point2d,
            const cv::Mat& K,
            cv::Mat& R, cv::Mat& t,
            std::vector<int>& inliers);
};