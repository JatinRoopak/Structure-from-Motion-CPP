#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "featurematching.hpp"
#include "epipolargeometry.hpp"
#include "PoseRecovery.hpp"
#include "PnP.hpp"

int main() {
    std::string videopath = "../assets/Barn/split_a_barn.mp4";

    cv::VideoCapture cap(videopath);
    
    std::vector<cv::Mat> extractFrames;
    cv::Mat frame;

    int frameindex = 0;
    int extractionInterval = 5;

    while (cap.read(frame)){
        if (frameindex % extractionInterval == 0){
            extractFrames.push_back(frame.clone()); //clone every frame after certain interval
        }
        frameindex++;
    }
    cap.release();
    std::cout << "number of frame extracted: " << extractFrames.size() << std::endl;

    cv::Mat img1 = extractFrames[0];
    cv::Mat img2 = extractFrames[1];

    Camera cam(img1.cols, img1.rows);
    cv::Mat K = cam.getIntrinsic();

    FeatureMatcher matcher;
    std::vector<cv::KeyPoint> kp1, kp2;

    cv::Mat desc1, desc2;
    std::vector<cv::DMatch> good_matches;

    matcher.extractandmatch(img1, img2, kp1, kp2, desc1, desc2, good_matches);

    std::vector<cv::Point2f> pts1, pts2;
    EpipolarGeometry::extractMatchCoordinates(kp1, kp2, good_matches, pts1, pts2);

    cv::Mat E1, E2;
    std::vector<uchar> inliersMask;
    EpipolarGeometry::essentialMatrix(pts1, pts2, K, E1, E2, inliersMask);

    std::cout << "\nE1 (All Correspondences):\n" << E1 << std::endl;
    std::cout << "\nE2 (RANSAC):\n" << E2 << std::endl; 

    //extracting the 4 candidate poses from the E2(with RANSAC) matrix
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    PoseRecovery::extractCandidatePose(E2, rotations, translations);

    std::cout << "The 4 Candidate (R, t) Pairs:" << std::endl;
    for (int i = 0; i < 4; i++) { //printing all four combination of R and t. 
        std::cout << "Pair " << i + 1 << ":" << std::endl;
        std::cout << "R:\n" << rotations[i] << std::endl;
        std::cout << "t:\n" << translations[i].t() << std::endl;
    }

    std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            inlier_pts1.push_back(pts1[i]);
            inlier_pts2.push_back(pts2[i]);
        }
    }
    std::cout << "Triangulating using " << inlier_pts1.size() << " RANSAC inliers..." << std::endl;

    //finding best combination using cheirality
    cv::Mat best_R, best_t, points4D;
    int best_index = PoseRecovery::correctPose(rotations, translations, inlier_pts1, inlier_pts2, K, best_R, best_t, points4D);

    std::cout << "\nBest R:\n" << best_R << std::endl;
    std::cout << "Best t:\n" << best_t << std::endl;

    cv::Mat P1 = cv::Mat::eye(3,4,CV_64F); //assume first camera is at origin
    P1 = K*P1;

    cv::Mat P2(3,4,CV_64F);
    best_R.copyTo(P2(cv::Rect(0,0,3,3)));
    best_t.copyTo(P2(cv::Rect(3,0,1,3)));
    P2 = K*P2;

    std::vector<int> inlier_to_map_index(inlier_pts1.size(), -1);
    std::vector<cv::Mat> valid_columns; 
    std::vector<cv::Vec3b> point_colors;

    std::vector<cv::Point2f> filtered_pts1;
    std::vector<cv::Point2f> filtered_pts2;

    int map_idx = 0;

    for (int i = 0; i < points4D.cols; i++) {
        double w = points4D.at<double>(3, i);
        if (std::abs(w) < 1e-7) continue; 
        
        double x = points4D.at<double>(0,i)/w;
        double y = points4D.at<double>(1,i)/w;
        double z = points4D.at<double>(2,i)/w;
        
        cv::Mat pt3d_3x1 = (cv::Mat_<double>(3,1) << x, y, z);
        cv::Mat pt_cam2 = best_R * pt3d_3x1 + best_t;

        cv::Mat pt3d_4x1 = (cv::Mat_<double>(4,1) << x, y, z, 1.0);
        double err1 = PoseRecovery::calculateReprojectedError(P1, pt3d_4x1, inlier_pts1[i]);
        double err2 = PoseRecovery::calculateReprojectedError(P2, pt3d_4x1, inlier_pts2[i]);
        
        if (z > 0 && pt_cam2.at<double>(2,0) > 0 && err1 < 5.0 && err2 < 5.0) {
            valid_columns.push_back(points4D.col(i));
            inlier_to_map_index[i] = map_idx++;

            filtered_pts1.push_back(inlier_pts1[i]);
            filtered_pts2.push_back(inlier_pts2[i]);

            // Grab the pixel color from img1
            int px = std::round(inlier_pts1[i].x);
            int py = std::round(inlier_pts1[i].y);
            if (px >= 0 && px < img1.cols && py >= 0 && py < img1.rows) {
                point_colors.push_back(img1.at<cv::Vec3b>(py, px));
            } else {
                point_colors.push_back(cv::Vec3b(128, 128, 128));
            }
        }
    }

    cv::Mat filtered_points4D;
    if (!valid_columns.empty()) cv::hconcat(valid_columns, filtered_points4D); 
    points4D = filtered_points4D;
    std::cout << "Filtered map: " << points4D.cols << " valid 3D points" << std::endl;

    if (filtered_points4D.empty()) {
        std::cout << "ERROR: No valid 3D points after filtering!" << std::endl;
        return -1;
    }
    points4D = filtered_points4D;

    std::cout << "Filtered map: " << points4D.cols << " valid 3D points" << std::endl;

    //before refinement
    std::cout << "\nExporting Task 2 initial cloud (Before Refinement)..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Barn/before_refinement.ply", points4D, point_colors);

    //after refinement
    std::cout << "Running Non-Linear Refinement..." << std::endl;
    PoseRecovery::refine3DPoints(P1, P2, filtered_pts1, filtered_pts2, points4D);

    std::cout << "Exporting Task 2 refined cloud (After Refinement)..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Barn/after_refinement.ply", points4D, point_colors);


    std::cout << "\n--- Task 3: PnP Pose Estimation (Video Loop) ---" << std::endl;

    // Build the mapping from kp2 keypoint index -> 3D map point index.
    std::vector<int> prev_to_3D_index(kp2.size(), -1);
    int inlier_counter = 0;
    for(size_t i = 0; i < good_matches.size(); i++){
        if(inliersMask[i]){
            int original_kp2_idx = good_matches[i].trainIdx; 
            if (inlier_to_map_index[inlier_counter] != -1) {
                prev_to_3D_index[original_kp2_idx] = inlier_to_map_index[inlier_counter];
            }
            inlier_counter++;
        }
    }

    cv::Mat prev_image = img2; 
    std::vector<cv::KeyPoint> prev_kp = kp2;
    cv::Mat prev_desc = desc2;
    
    cv::Mat current_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat current_t = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat prev_R = best_R.clone();
    cv::Mat prev_t = best_t.clone();
      
    for(size_t f = 2; f < extractFrames.size() ; f++){
        std::cout << "\nProcessing Frame " << f << "/" << extractFrames.size() - 1 << "..." << std::endl;

        cv::Mat current_image = extractFrames[f];
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;

        // 1. Extract SIFT for the new frame
        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();
        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);

        if (current_kp.empty() || current_desc.empty()) continue;

        // 2. Match using FLANN
        cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        flann->knnMatch(prev_desc, current_desc, knn_matches, 2);

        // Lowe ratio test
        std::vector<cv::DMatch> current_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2) { 
                if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance) {
                    current_matches.push_back(knn_matches[i][0]);
                }
            }
        }

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;
        std::vector<int> current_to_3D_index(current_kp.size(), -1);

        // 3. Find 2D-3D correspondences
        for (const auto& match : current_matches) {
            int prev_idx = match.queryIdx; 
            int curr_idx = match.trainIdx; 

            int index_3D = prev_to_3D_index[prev_idx];

            if (index_3D != -1) { 
                double w = points4D.at<double>(3, index_3D);
                if (std::abs(w) > 1e-7) {
                    double z = points4D.at<double>(2, index_3D) / w;
                    
                    if (z > 0) {
                        object_points_3D.push_back(cv::Point3f(
                            points4D.at<double>(0, index_3D) / w,
                            points4D.at<double>(1, index_3D) / w,
                            z
                        ));
                        image_points_2D.push_back(current_kp[curr_idx].pt);
                        current_to_3D_index[curr_idx] = index_3D; 
                    }
                }
            }
        }   

        std::cout << "  Found " << object_points_3D.size() << " valid positive-depth matches." << std::endl;
        
        // 4. Solve PnP
        if (object_points_3D.size() >= 6) { 
            std::vector<int> pnp_inliers;

            bool success = PnP_Estimator::estimatePose(object_points_3D, image_points_2D, K, current_R, current_t, pnp_inliers);
            std::cout << "  PnP returned: success=" << success << " inliers=" << pnp_inliers.size() << std::endl;

            if (success && pnp_inliers.size() >= 8) {
                std::cout << "  Camera Pose Estimated! Inliers: " << pnp_inliers.size() << std::endl;

                cv::Mat P1(3,4,CV_64F);
                prev_R.copyTo(P1(cv::Rect(0,0,3,3)));
                prev_t.copyTo(P1(cv::Rect(3,0,1,3)));
                P1 = K*P1;

                cv::Mat P2(3, 4, CV_64F);
                current_R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
                current_t.copyTo(P2(cv::Rect(3, 0, 1, 3)));
                P2 = K * P2;

                std::vector<cv::Point2f> new_pts1, new_pts2;
                std::vector<int> new_matches_curr_idx; 

                for (const auto& match : current_matches) {
                    int p_idx = match.queryIdx;
                    int c_idx = match.trainIdx;

                    if (prev_to_3D_index[p_idx] == -1) {
                        new_pts1.push_back(prev_kp[p_idx].pt);
                        new_pts2.push_back(current_kp[c_idx].pt);
                        new_matches_curr_idx.push_back(c_idx);
                    }
                }

                if (!new_pts1.empty()) {
                    cv::Mat new_points4D;
                    PoseRecovery::triangulation(P1, P2, new_pts1, new_pts2, new_points4D);
                    new_points4D.convertTo(new_points4D, CV_64F); 

                    std::vector<cv::Mat> valid_new_cols;
                    int old_map_size = points4D.cols;
                    int added_points = 0;

                    for (int i = 0; i < new_points4D.cols; i++) {
                        double w = new_points4D.at<double>(3, i);
                        if (std::abs(w) < 1e-7) continue;

                        double x = new_points4D.at<double>(0, i)/w;
                        double y = new_points4D.at<double>(1, i)/w;
                        double z = new_points4D.at<double>(2, i)/w;

                        cv::Mat pt3d_3x1 = (cv::Mat_<double>(3,1) << x, y, z);
                        cv::Mat pt_cam2 = current_R * pt3d_3x1 + current_t;

                        cv::Mat pt3d_4x1 = (cv::Mat_<double>(4,1) << x, y, z, 1.0);
                        double err1 = PoseRecovery::calculateReprojectedError(P1, pt3d_4x1, new_pts1[i]);
                        double err2 = PoseRecovery::calculateReprojectedError(P2, pt3d_4x1, new_pts2[i]);

                        if (z > 0 && pt_cam2.at<double>(2,0) > 0 && err1 < 5.0 && err2 < 5.0) {
                            valid_new_cols.push_back(new_points4D.col(i));

                            int c_idx = new_matches_curr_idx[i];
                            current_to_3D_index[c_idx] = old_map_size + added_points;
                            added_points++;
                            
                            int px = std::round(new_pts1[i].x);
                            int py = std::round(new_pts1[i].y);
                            if (px >= 0 && px < prev_image.cols && py >= 0 && py < prev_image.rows) {
                                point_colors.push_back(prev_image.at<cv::Vec3b>(py, px));
                            } else {
                                point_colors.push_back(cv::Vec3b(128, 128, 128));
                            }
                        }
                    }

                    if (!valid_new_cols.empty()) {
                        cv::Mat filtered_new;
                        cv::hconcat(valid_new_cols, filtered_new);
                        cv::hconcat(points4D, filtered_new, points4D);
                        std::cout << "  Triangulated " << added_points << " GOOD points. Map Size: " << points4D.cols << std::endl;
                    }
                }

                prev_image = current_image.clone();
                prev_kp = current_kp;
                prev_desc = current_desc.clone();
                prev_to_3D_index = current_to_3D_index;
                prev_R = current_R.clone();
                prev_t = current_t.clone();

            } 
            
            else {
                std::cout << "  PnP Failed for this frame (poor tracking)." << std::endl;
            }
        } 
        
        else {
            std::cout << "  Not enough 3D points to solve PnP. Tracking lost!" << std::endl;
        }
    }

    std::cout << "Processing complete! Exporting final massive cloud..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Barn/barn_cloud_colored.ply", points4D, point_colors);

    std::string videopathB = "../assets/Barn/split_b_barn.mp4";
    cv::VideoCapture capB(videopathB);
    std::vector<cv::Mat> extractFramesB;
    frameindex = 0;
    while (capB.read(frame)){
        if (frameindex % extractionInterval == 0){
            extractFramesB.push_back(frame.clone());
        }
        frameindex++;
    }
    capB.release();
    std::cout << "Split B extracted: " << extractFramesB.size() << std::endl;

    double total_loc_error = 0.0;
    int successful_loc_frames = 0;

    for(size_t f = 0; f < extractFramesB.size(); f++){
        cv::Mat current_image = extractFramesB[f];
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;

        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();
        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);
        if (current_kp.empty() || current_desc.empty()) continue;

        cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        flann->knnMatch(prev_desc, current_desc, knn_matches, 2);

        std::vector<cv::DMatch> current_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance) {
                current_matches.push_back(knn_matches[i][0]);
            }
        }

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;
        std::vector<int> current_to_3D_index(current_kp.size(), -1);

        for (const auto& match : current_matches) {
            int prev_idx = match.queryIdx; 
            int curr_idx = match.trainIdx; 
            int index_3D = prev_to_3D_index[prev_idx];

            if (index_3D != -1) { 
                double w = points4D.at<double>(3, index_3D);
                if (std::abs(w) > 1e-7) {
                    double z = points4D.at<double>(2, index_3D) / w;
                    if (z > 0) {
                        object_points_3D.push_back(cv::Point3f(points4D.at<double>(0, index_3D)/w, points4D.at<double>(1, index_3D)/w, z));
                        image_points_2D.push_back(current_kp[curr_idx].pt);
                        current_to_3D_index[curr_idx] = index_3D; 
                    }
                }
            }
        }   

        if (object_points_3D.size() >= 6) { 
            std::vector<int> pnp_inliers;
            bool success = PnP_Estimator::estimatePose(object_points_3D, image_points_2D, K, current_R, current_t, pnp_inliers);

            if (success && pnp_inliers.size() >= 8) {
                cv::Mat P_loc(3, 4, CV_64F);
                current_R.copyTo(P_loc(cv::Rect(0, 0, 3, 3)));
                current_t.copyTo(P_loc(cv::Rect(3, 0, 1, 3)));
                P_loc = K * P_loc;

                double frame_error_sum = 0.0;
                for (int idx : pnp_inliers) {
                    cv::Mat pt4 = (cv::Mat_<double>(4,1) << object_points_3D[idx].x, object_points_3D[idx].y, object_points_3D[idx].z, 1.0);
                    frame_error_sum += PoseRecovery::calculateReprojectedError(P_loc, pt4, image_points_2D[idx]);
                }
                
                double mean_frame_error = frame_error_sum / pnp_inliers.size();
                total_loc_error += mean_frame_error;
                successful_loc_frames++;

                std::cout << "  Split B Frame " << f << " Localized! Mean Error: " << mean_frame_error << " px" << std::endl;

                prev_image = current_image.clone();
                prev_kp = current_kp;
                prev_desc = current_desc.clone();
                prev_to_3D_index = current_to_3D_index;
            } else {
                std::cout << "  Split B Frame " << f << " PnP Failed." << std::endl;
            }
        } else {
            std::cout << "  Split B Frame " << f << " Tracking Lost (Not enough 3D matches)." << std::endl;
        }
    }

    if (successful_loc_frames > 0) {
        std::cout << "\n>>> FINAL LOCALIZATION ERROR: " << total_loc_error / successful_loc_frames << " px <<<\n" << std::endl;
    }

    return 0;
}