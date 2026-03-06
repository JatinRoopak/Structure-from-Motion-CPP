#include "PoseRecovery.hpp"
#include <fstream>
#include <ceres/ceres.h>

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
                    cv::Mat pts3D = (cv::Mat_<double>(3, 1) << x, y, z);
                    cv::Mat pt_cam2 = rotations[i] * pts3D + translations[i];
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

double PoseRecovery::calculateReprojectedError(
    const cv::Mat& P, 
    const cv::Mat& point3D, 
    const cv::Point2f& swift2DPoint){

        cv::Mat project2d = P*point3D;

        double w = project2d.at<double>(2, 0);
        if (std::abs(w) < 1e-7) return 0.0; // Prevent division by zero

        double actual_x_calculated = project2d.at<double>(0, 0) / w;
        double actual_y_calculated = project2d.at<double>(1, 0) / w;

        // 3. Calculate the distance (error) between the calculated pixel and the actual SIFT pixel
        double dx = actual_x_calculated - swift2DPoint.x;
        double dy = actual_y_calculated - swift2DPoint.y;
        
        // Return the squared distance (e_i)
        return std::sqrt(dx * dx + dy * dy);
    }

void PoseRecovery::refine3DPoints( //perform non linear optimization to minimize the reprojection error (min (||x1 - P1X||sqr + ||x2 - P2X||sqr))
    const cv::Mat& P1, const cv::Mat P2, //camera projection matrix from the two frames 
    const std::vector<cv::Point2f>& pts1, //points given by swift
    const std::vector<cv::Point2f>& pts2,
    cv::Mat& points4D){

        double learning_rate = 0.01;
        double delta = 1e-5;
        int iteration = 10;

        for (int i=0; i<points4D.cols; i++){
            double w = points4D.at<double>(3, i);
            if(std::abs(w) < 1e-7) continue;

            cv::Mat pts3D = (cv::Mat_<double>(4,1) << 
                points4D.at<double>(0, i)/w,
                points4D.at<double>(1, i)/ w,
                points4D.at<double>(2, i)/ w,
                1.0);

            for (int iter = 0; iter <iteration; iter++){ // iteration of gradient descent
                double current_error = calculateReprojectedError(P1, pts3D, pts1[i]) + calculateReprojectedError(P2, pts3D, pts2[i]);

                cv::Mat gradient = cv::Mat::zeros(3,1,CV_64F);

                for (int axis = 0; axis<3; axis++){
                    cv::Mat pt_perturbed = pts3D.clone();
                    pt_perturbed.at<double>(axis, 0)+= delta;

                    double new_error = calculateReprojectedError(P1, pt_perturbed, pts1[i]) + calculateReprojectedError(P2, pt_perturbed, pts2[i]);

                    gradient.at<double>(axis, 0) = (new_error-current_error)/delta;
                }

                //Update the point by moving in opposite direction of the gradient
                pts3D.at<double>(0, 0) -= learning_rate * gradient.at<double>(0, 0);
                pts3D.at<double>(1, 0) -= learning_rate * gradient.at<double>(1, 0);
                pts3D.at<double>(2, 0) -= learning_rate * gradient.at<double>(2, 0);
            }

            points4D.at<double>(0, i) = pts3D.at<double>(0, 0);
            points4D.at<double>(1, i) = pts3D.at<double>(1, 0);
            points4D.at<double>(2, i) = pts3D.at<double>(2, 0);
            points4D.at<double>(3, i) = 1.0;
        }

        std::cout << "Refinement Done " << std::endl;
    }

void PoseRecovery::exportToPLY(const std::string& filename, const cv::Mat& points4D){

    int num_points = points4D.cols;

    std::vector<cv::Vec3d> valid_points; 
    for (int i =0; i<num_points; i++){ // to take care where W = 0
        double w = points4D.at<double>(3,i);
        if(std::abs(w) < 1e-7) continue;
        valid_points.push_back({
            points4D.at<double>(0, i) / w,
            points4D.at<double>(1, i) / w,
            points4D.at<double>(2, i) / w
        });
    }

    std::ofstream plyFile(filename);
    //writing header of ply file
    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << valid_points.size() << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "end_header\n";

    //writing data into ply file
    for (auto& p:valid_points){
        plyFile << p[0] << " " << p[1] << " " << p[2] << "\n";
    }

    plyFile.close();
    std::cout << "Exported " << valid_points.size() << " points to " << filename << "\n";
}


