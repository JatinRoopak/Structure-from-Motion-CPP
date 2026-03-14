#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "featurematching.hpp"
#include "epipolargeometry.hpp"
#include "PoseRecovery.hpp"
#include "PnP.hpp"

void saveTrajectoryPLY(const std::string& filename, const std::vector<cv::Point3f>& traj, int r, int g, int b) {
    std::ofstream out(filename);
    out << "ply\nformat ascii 1.0\nelement vertex " << traj.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (const auto& p : traj) {
        out << p.x << " " << p.y << " " << p.z << " " << r << " " << g << " " << b << "\n";
    }
    out.close();
}

struct Observation { int cam_idx; int pt_idx; float x; float y; };

int main() {
    std::string videopath = "../assets/Barn/split_a_barn.mp4";

    cv::VideoCapture cap(videopath);
    
    std::vector<cv::Mat> extractFrames;
    cv::Mat frame;

    int frameindex = 0;
    int extractionInterval = 20;

    while (cap.read(frame)){
        if (frameindex % extractionInterval == 0){
            extractFrames.push_back(frame.clone()); 
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

    cv::Mat F_E2 = K.inv().t() * E2 * K.inv();
    std::vector<cv::Vec3f> lines2;
    std::vector<cv::Point2f> pts1_draw(pts1.begin(), pts1.begin() + 10);
    std::vector<cv::Point2f> pts2_draw(pts2.begin(), pts2.begin() + 10);
    cv::computeCorrespondEpilines(pts1_draw, 1, F_E2, lines2);
    cv::Mat img2_epi = img2.clone();
    for (size_t i = 0; i < 10; i++) {
        cv::circle(img2_epi, pts2_draw[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::line(img2_epi, cv::Point(0, -lines2[i][2]/lines2[i][1]),
                 cv::Point(img2_epi.cols, -(lines2[i][0]*img2_epi.cols + lines2[i][2])/lines2[i][1]), cv::Scalar(0, 0, 255), 1);
    }
    cv::imwrite("../outputs/Barn/epipolar_lines.jpg", img2_epi);

    std::cout << "\nE1 (All Correspondences):\n" << E1 << std::endl;
    std::cout << "\nE2 (RANSAC):\n" << E2 << std::endl; 

    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    PoseRecovery::extractCandidatePose(E2, rotations, translations);

    std::cout << "The 4 Candidate (R, t) Pairs:" << std::endl;
    for (int i = 0; i < 4; i++) { 
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

    cv::Mat best_R, best_t, points4D;
    int best_index = PoseRecovery::correctPose(rotations, translations, inlier_pts1, inlier_pts2, K, best_R, best_t, points4D);

    std::cout << "\nBest R:\n" << best_R << std::endl;
    std::cout << "Best t:\n" << best_t << std::endl;

    cv::Mat P1 = cv::Mat::eye(3,4,CV_64F); 
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

    std::vector<cv::Mat> ba_cameras_R;
    std::vector<cv::Mat> ba_cameras_t;
    std::vector<Observation> ba_obs;
    std::vector<cv::Point3f> trajectory_A;
    std::vector<cv::Point3f> trajectory_B;

    ba_cameras_R.push_back(cv::Mat::eye(3,3,CV_64F));
    ba_cameras_t.push_back(cv::Mat::zeros(3,1,CV_64F));
    ba_cameras_R.push_back(best_R.clone());
    ba_cameras_t.push_back(best_t.clone());

    trajectory_A.push_back(cv::Point3f(0, 0, 0));
    cv::Mat C1 = -best_R.t() * best_t;
    trajectory_A.push_back(cv::Point3f(C1.at<double>(0), C1.at<double>(1), C1.at<double>(2)));

    cv::Mat global_desc;
    std::vector<int> global_to_3D;

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
            inlier_to_map_index[i] = map_idx;
            
            ba_obs.push_back({0, map_idx, inlier_pts1[i].x, inlier_pts1[i].y});
            ba_obs.push_back({1, map_idx, inlier_pts2[i].x, inlier_pts2[i].y});
            map_idx++;

            filtered_pts1.push_back(inlier_pts1[i]);
            filtered_pts2.push_back(inlier_pts2[i]);

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

    std::cout << "\nExporting Task 2 initial cloud (Before Refinement)..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Barn/before_refinement.ply", points4D, point_colors);

    std::cout << "Running Non-Linear Refinement..." << std::endl;
    PoseRecovery::refine3DPoints(P1, P2, filtered_pts1, filtered_pts2, points4D);

    std::cout << "Exporting Task 2 refined cloud (After Refinement)..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Barn/after_refinement.ply", points4D, point_colors);


    std::cout << "\n--- Task 3: PnP Pose Estimation (Video Loop) ---" << std::endl;

    std::vector<int> prev_to_3D_index(kp2.size(), -1);
    int inlier_counter = 0;
    for(size_t i = 0; i < good_matches.size(); i++){
        if(inliersMask[i]){
            int original_kp2_idx = good_matches[i].trainIdx; 
            if (inlier_to_map_index[inlier_counter] != -1) {
                prev_to_3D_index[original_kp2_idx] = inlier_to_map_index[inlier_counter];
                global_desc.push_back(desc2.row(original_kp2_idx));
                global_to_3D.push_back(inlier_to_map_index[inlier_counter]);
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

        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();
        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);

        if (current_kp.empty() || current_desc.empty()) continue;

        cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        flann->knnMatch(prev_desc, current_desc, knn_matches, 2);

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
        
        if (object_points_3D.size() >= 6) { 
            std::vector<int> pnp_inliers;

            bool success = PnP_Estimator::estimatePose(object_points_3D, image_points_2D, K, current_R, current_t, pnp_inliers);
            std::cout << "  PnP returned: success=" << success << " inliers=" << pnp_inliers.size() << std::endl;

            if (success && pnp_inliers.size() >= 8) {
                std::cout << "  Camera pose estimated inliers: " << pnp_inliers.size() << std::endl;

                ba_cameras_R.push_back(current_R.clone());
                ba_cameras_t.push_back(current_t.clone());
                int current_cam_id = ba_cameras_R.size() - 1;

                cv::Mat C = -current_R.t() * current_t;
                trajectory_A.push_back(cv::Point3f(C.at<double>(0), C.at<double>(1), C.at<double>(2)));

                for (int idx : pnp_inliers) {
                    ba_obs.push_back({ current_cam_id, current_to_3D_index[current_matches[idx].trainIdx], image_points_2D[idx].x, image_points_2D[idx].y });
                }

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
                            int new_pt_id = old_map_size + added_points;
                            current_to_3D_index[c_idx] = new_pt_id;

                            ba_obs.push_back({ current_cam_id - 1, new_pt_id, new_pts1[i].x, new_pts1[i].y });
                            ba_obs.push_back({ current_cam_id, new_pt_id, new_pts2[i].x, new_pts2[i].y });

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
                        std::cout << "  Triangulated " << added_points << " Good points. Map Size: " << points4D.cols << std::endl;
                    }
                }

                for (size_t i = 0; i < current_kp.size(); i++) {
                    if (current_to_3D_index[i] != -1) {
                        global_desc.push_back(current_desc.row(i));
                        global_to_3D.push_back(current_to_3D_index[i]);
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

    saveTrajectoryPLY("../outputs/Barn/trajectory_A.ply", trajectory_A, 0, 255, 0);

    std::ofstream cam_file("../outputs/Barn/ba_cameras.csv");
    cam_file << "cam_id,rx,ry,rz,tx,ty,tz\n";
    for(size_t i=0; i<ba_cameras_R.size(); i++){
        cv::Mat rvec; cv::Rodrigues(ba_cameras_R[i], rvec);
        cam_file << i << "," << rvec.at<double>(0) << "," << rvec.at<double>(1) << "," << rvec.at<double>(2) << "," << ba_cameras_t[i].at<double>(0) << "," << ba_cameras_t[i].at<double>(1) << "," << ba_cameras_t[i].at<double>(2) << "\n";
    }
    cam_file.close();

    std::ofstream pts_file("../outputs/Barn/ba_points.csv");
    pts_file << "pt_id,x,y,z\n";
    for(int i=0; i<points4D.cols; i++){
        double w = points4D.at<double>(3,i);
        if(std::abs(w)>1e-7) {
            pts_file << i << "," << points4D.at<double>(0,i)/w << "," << points4D.at<double>(1,i)/w << "," << points4D.at<double>(2,i)/w << "\n";
        }
    }
    pts_file.close();

    std::ofstream obs_file("../outputs/Barn/ba_obs.csv");
    obs_file << "cam_id,pt_id,u,v\n";
    for(const auto& o : ba_obs){
        obs_file << o.cam_idx << "," << o.pt_idx << "," << o.x << "," << o.y << "\n";
    }
    obs_file.close();

    std::ofstream int_file("../outputs/Barn/intrinsics.txt");
    int_file << K.at<double>(0,0) << " " << K.at<double>(1,1) << " " << K.at<double>(0,2) << " " << K.at<double>(1,2);
    int_file.close();

    extractFrames.clear();
    extractFrames.shrink_to_fit();

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

    std::ofstream metrics_file("../outputs/Barn/metrics_B.csv");
    metrics_file << "Frame,MeanError,Inliers,TotalMatches,InlierRatio\n";

    std::cout << "\nBuilding FLANN index for Global Map (" << global_desc.rows << " descriptors)..." << std::endl;
    cv::Ptr<cv::DescriptorMatcher> global_flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    double total_loc_error = 0.0;
    int successful_loc_frames = 0;

    for(size_t f = 0; f < extractFramesB.size(); f++){
        cv::Mat current_image = extractFramesB[f];
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;

        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();
        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);
        if (current_kp.empty() || current_desc.empty()) continue;

        std::vector<std::vector<cv::DMatch>> knn_matches;
        global_flann->knnMatch(current_desc, global_desc, knn_matches, 2);

        std::vector<cv::DMatch> current_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance) {
                current_matches.push_back(knn_matches[i][0]);
            }
        }

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;

        for (const auto& match : current_matches) {
            int curr_idx = match.queryIdx; 
            int global_idx = match.trainIdx; 
            int index_3D = global_to_3D[global_idx];

            if (index_3D != -1) { 
                double w = points4D.at<double>(3, index_3D);
                if (std::abs(w) > 1e-7) {
                    double z = points4D.at<double>(2, index_3D) / w;
                    if (z > 0) {
                        object_points_3D.push_back(cv::Point3f(points4D.at<double>(0, index_3D)/w, points4D.at<double>(1, index_3D)/w, z));
                        image_points_2D.push_back(current_kp[curr_idx].pt);
                    }
                }
            }
        }   

        if (object_points_3D.size() >= 6) { 
            std::vector<int> pnp_inliers;
            cv::Mat loc_R, loc_t;
            bool success = PnP_Estimator::estimatePose(object_points_3D, image_points_2D, K, loc_R, loc_t, pnp_inliers);

            if (success && pnp_inliers.size() >= 8) {
                cv::Mat P_loc(3, 4, CV_64F);
                loc_R.copyTo(P_loc(cv::Rect(0, 0, 3, 3)));
                loc_t.copyTo(P_loc(cv::Rect(3, 0, 1, 3)));
                P_loc = K * P_loc;

                double frame_error_sum = 0.0;
                for (int idx : pnp_inliers) {
                    cv::Mat pt4 = (cv::Mat_<double>(4,1) << object_points_3D[idx].x, object_points_3D[idx].y, object_points_3D[idx].z, 1.0);
                    frame_error_sum += PoseRecovery::calculateReprojectedError(P_loc, pt4, image_points_2D[idx]);
                }
                
                double mean_frame_error = frame_error_sum / pnp_inliers.size();

                if (mean_frame_error > 20.0 || mean_frame_error < 0) {
                    continue; 
                }

                cv::Mat C_loc = -loc_R.t() * loc_t;
                trajectory_B.push_back(cv::Point3f(C_loc.at<double>(0), C_loc.at<double>(1), C_loc.at<double>(2)));

                metrics_file << f << "," << mean_frame_error << "," << pnp_inliers.size() << "," << object_points_3D.size() << "," << (float)pnp_inliers.size() / object_points_3D.size() << "\n";

                total_loc_error += mean_frame_error;
                successful_loc_frames++;

                std::cout << "  Split B Frame " << f << " Localized globally! Mean Error: " << mean_frame_error << " px" << std::endl;
            } else {
                std::cout << "  Split B Frame " << f << " PnP Failed." << std::endl;
            }
        } else {
            std::cout << "  Split B Frame " << f << " Tracking Lost (Not enough global matches)." << std::endl;
        }
    }

    metrics_file.close();
    saveTrajectoryPLY("../outputs/Barn/trajectory_B.ply", trajectory_B, 255, 0, 0);

    if (successful_loc_frames > 0) {
        std::cout << "\n>>> FINAL LOCALIZATION ERROR: " << total_loc_error / successful_loc_frames << " px <<<\n" << std::endl;
    }

    return 0;
}