#pragma once 
#include <opencv2/opencv.hpp>
#include <vector>

class PoseRecovery{
    public:
        static void extractCandidatePose( //decompose the essential matrix into 4 (r, t) parts
            const cv::Mat& E,
            std::vector<cv::Mat>& rotation,
            std::vector<cv::Mat>& translations);

        static void triangulation( //triangulation to find the 3D point
            const cv::Mat& p1, const cv::Mat& p2,
            const std::vector<cv::Point2f>& pts1,
            const std::vector<cv::Point2f>& pts2,
            cv::Mat& point3d);

        static int correctPose( //find the best R and t combination
            const std::vector<cv::Mat>& rotation,
            const std::vector<cv::Mat>& translation,
            const std::vector<cv::Point2f>& pts1,
            const std::vector<cv::Point2f>& pts2,
            const cv::Mat& K,
            cv::Mat& best_R, cv::Mat& best_t,
            cv::Mat& best_points3D);

        static double calculateReprojectedError(
            const cv::Mat& P,
            const cv::Mat& point3D,
            const cv::Point2f& point2D);

        static void refine3DPoints(
            const cv::Mat& P1, const cv::Mat P2,
            const std::vector<cv::Point2f>& pts1,
            const std::vector<cv::Point2f>& pts2,
            cv::Mat& point3D);

        static void exportToPLY(const std::string& filename, const cv::Mat& points4D);
};