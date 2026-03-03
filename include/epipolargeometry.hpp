#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class EpipolarGeometry{
    public:
        static void extractMatchCoordinates(
            const std::vector<cv::KeyPoint>& kp1,
            const std::vector<cv::KeyPoint>& kp2,
            const std::vector<cv::DMatch>& matches,
            std::vector<cv::Point2f>& pts1,
            std::vector<cv::Point2f>& pts2);
        
        static void applyRank2(cv::Mat& E);

        static void essentialMatrix(
            const std::vector<cv::Point2f>& pts1,
            const std::vector<cv::Point2f>& pts2,
            const cv::Mat& K,
            cv::Mat& E1, cv::Mat& E2,
            std::vector<uchar>& inliersMask);

};