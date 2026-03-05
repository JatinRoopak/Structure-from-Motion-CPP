#include "PoseRecovery.hpp"


void PoseRecovery::extractCandidatePose(
    const cv::Mat& E,
    std::vector<cv::Mat>& rotations,
    std::vector<cv::Mat>& translations){

        cv::Mat R1, R2, t;
        cv::decomposeEssentialMat(E, R1, R2, t);

        rotations.push_back(R1); translations.push_back(t);
        rotations.push_back(R1); translations.push_back(-t);
        rotations.push_back(R2); translations.push_back(t);
        rotations.push_back(R2); translations.push_back(-t);
    }

void PoseRecovery::triangulation(
    const cv::Mat& p1, const cv::Mat& p2,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    cv::Mat& points3d){

        cv::triangulatePoints(p1, p2, pts1, pts2, points3d);

    }

int PoseRecovery::correctPose(
    const std::vector<cv::Mat>& rotations,
    const std::vector<cv::Mat>& translations,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    cv::Mat& best_R, cv::Mat& best_t,
    cv::Mat& best_points3D){

        int best_index = -1;
        int max_positive_depths = 0;

        cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // identity matrix no rotation and translation
        P1 = K * P1; // calculating camera pose

        for (int i = 0; i < 4; i++) {
            cv::Mat P2(3, 4, CV_64F);
            rotations[i].copyTo(P2(cv::Rect(0, 0, 3, 3)));
            translations[i].copyTo(P2(cv::Rect(3, 0, 1, 3)));
            P2 = K * P2;

            cv::Mat points4D;
            triangulation(P1, P2, pts1, pts2, points4D);

            int positive_depth_count = 0;

            // Check Cheirality
            for (int j = 0; j < points4D.cols; j++) {
                double w = points4D.at<double>(3, j);
                if (w == 0) continue; 
                
                double x = points4D.at<double>(0, j) / w;
                double y = points4D.at<double>(1, j) / w;
                double z = points4D.at<double>(2, j) / w;

                if (z > 0) {
                    cv::Mat pt3D = (cv::Mat_<double>(3, 1) << x, y, z);
                    cv::Mat pt_cam2 = rotations[i] * pt3D + translations[i];
                    if (pt_cam2.at<double>(2, 0) > 0) {
                        positive_depth_count++;
                    }
                }
            }

            if (positive_depth_count > max_positive_depths) {
                max_positive_depths = positive_depth_count;
                best_index = i;
                best_R = rotations[i].clone();
                best_t = translations[i].clone();
                best_points3D = points4D.clone(); 
            }
        }
        return best_index;
    }