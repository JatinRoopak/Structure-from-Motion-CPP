#include "PnP.hpp"

bool PnP_Estimator::estimatePose(
    const std::vector<cv::Point3f>& point3d,
    const std::vector<cv::Point2f>& point2d,
    const cv::Mat& K,
    cv::Mat& R, cv::Mat& t,
    std::vector<int>& inliers){

        cv::Mat distCoeff = cv::Mat::zeros(4,1,CV_64F);
        cv::Mat rotation_vector;

        bool success = cv::solvePnPRansac(point3d, point2d, K, distCoeff, rotation_vector, t, false, 200, 5.0, 0.99, inliers);

        if (success){ //convert the angle vector into standard rotation vector
            cv::Rodrigues(rotation_vector, R);
        }

        return success;
    }