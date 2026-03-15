#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "featurematching.hpp"
#include "epipolargeometry.hpp"
#include "PoseRecovery.hpp"
#include "PnP.hpp" 

void saveCamerasPLY(const std::string& filename, const std::vector<cv::Mat>& Rs, const std::vector<cv::Mat>& ts, float scale, int cr, int cg, int cb) {
    std::ofstream out(filename);
    int num_cams = Rs.size();
    int num_vertices = num_cams * 2; 
    int num_edges = num_cams;        

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << num_vertices << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "element edge " << num_edges << "\n";
    out << "property int vertex1\nproperty int vertex2\n";
    out << "end_header\n";

    std::vector<cv::Point3f> centers;
    std::vector<cv::Point3f> endpoints;

    for (size_t i = 0; i < num_cams; i++) {
        cv::Mat R = Rs[i];
        cv::Mat t = ts[i];
        cv::Mat R_T = R.t();
        
        cv::Mat C_mat = -R_T * t;
        cv::Point3f C(C_mat.at<double>(0,0), C_mat.at<double>(1,0), C_mat.at<double>(2,0));
        
        cv::Mat Z_cam = (cv::Mat_<double>(3,1) << 0, 0, 1);
        cv::Mat look_vec = R_T * Z_cam;
        cv::Point3f L(look_vec.at<double>(0,0), look_vec.at<double>(1,0), look_vec.at<double>(2,0));
        
        cv::Point3f endpoint = C + L * scale;

        centers.push_back(C);
        endpoints.push_back(endpoint);
    }

    for (const auto& c : centers) {
        out << c.x << " " << c.y << " " << c.z << " " << cr << " " << cg << " " << cb << "\n";
    }
    for (const auto& e : endpoints) {
        out << e.x << " " << e.y << " " << e.z << " 255 0 0\n";
    }
    for (int i = 0; i < num_cams; i++) {
        out << i << " " << (i + num_cams) << "\n";
    }
    out.close();
}

struct Observation { int cam_idx; int pt_idx; float x; float y; };

int main() {
    std::string videopath = "../assets/Truck/split_a_Truck.mp4";

    cv::VideoCapture cap(videopath);
    cv::Mat frame;
    int frameindex = 0;

    int extractionInterval = 30; 
    std::cout << "Total frames to process in Split A: " << (int)(cap.get(cv::CAP_PROP_FRAME_COUNT) / extractionInterval) << std::endl;

    cv::Mat img1, img2;
    while (cap.read(frame)) {
        if (frameindex % extractionInterval == 0) {
            if (img1.empty()) {
                img1 = frame.clone();
            } else if (img2.empty()) {
                img2 = frame.clone();
                frameindex++;
                break;
            }
        }
        frameindex++;
    }
    std::cout << "First 2 frames extracted for initialization." << std::endl;

    Camera cam(img1.cols, img1.rows);
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 0.7 * img1.cols;
    K.at<double>(1, 1) = 0.7 * img1.cols;
    K.at<double>(0, 2) = img1.cols / 2.0;
    K.at<double>(1, 2) = img1.rows / 2.0;
    std::cout << "\nUsing Corrected Intrinsics:\n" << K << std::endl;

    cv::Ptr<cv::SIFT> init_sift = cv::SIFT::create(8000); // 8000 for indoor features
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    init_sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    init_sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    std::vector<std::vector<cv::DMatch>> knn_matches_init;
    matcher->knnMatch(desc1, desc2, knn_matches_init, 2);

    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches_init.size(); i++) {
        if (knn_matches_init[i][0].distance < 0.75f * knn_matches_init[i][1].distance) {
            good_matches.push_back(knn_matches_init[i][0]);
        }
    }

    std::vector<cv::Point2f> pts1, pts2;
    EpipolarGeometry::extractMatchCoordinates(kp1, kp2, good_matches, pts1, pts2);

    cv::Mat E1, E2;
    std::vector<uchar> inliersMask;
    EpipolarGeometry::essentialMatrix(pts1, pts2, K, E1, E2, inliersMask);

    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    PoseRecovery::extractCandidatePose(E2, rotations, translations);

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

    cv::Mat P1 = cv::Mat::eye(3,4,CV_64F); 
    P1 = K*P1;

    cv::Mat P2(3,4,CV_64F);
    best_R.copyTo(P2(cv::Rect(0,0,3,3)));
    best_t.copyTo(P2(cv::Rect(3,0,1,3)));
    P2 = K*P2;

    std::vector<cv::Mat> valid_columns; 
    std::vector<cv::Vec3b> point_colors;
    std::vector<cv::Point2f> filtered_pts1;
    std::vector<cv::Point2f> filtered_pts2;
    std::vector<int> inlier_to_map_index(inlier_pts1.size(), -1);

    std::vector<cv::Mat> ba_cameras_R;
    std::vector<cv::Mat> ba_cameras_t;
    std::vector<Observation> ba_obs;

    ba_cameras_R.push_back(cv::Mat::eye(3,3,CV_64F));
    ba_cameras_t.push_back(cv::Mat::zeros(3,1,CV_64F));
    ba_cameras_R.push_back(best_R.clone());
    ba_cameras_t.push_back(best_t.clone());

    int map_idx = 0;

    for (int i = 0; i < points4D.cols; i++) {
        double w = points4D.at<double>(3, i);
        if (std::abs(w) < 1e-7) continue; 
        
        double x = points4D.at<double>(0,i)/w;
        double y = points4D.at<double>(1,i)/w;
        double z = points4D.at<double>(2,i)/w;
        
        cv::Mat pt3d_3x1 = (cv::Mat_<double>(3,1) << x, y, z);
        cv::Mat pt_cam1 = cv::Mat::eye(3,3,CV_64F) * pt3d_3x1 + cv::Mat::zeros(3,1,CV_64F);
        cv::Mat pt_cam2 = best_R * pt3d_3x1 + best_t;

        cv::Mat pt3d_4x1 = (cv::Mat_<double>(4,1) << x, y, z, 1.0);
        double err1 = PoseRecovery::calculateReprojectedError(P1, pt3d_4x1, inlier_pts1[i]);
        double err2 = PoseRecovery::calculateReprojectedError(P2, pt3d_4x1, inlier_pts2[i]);
        
        if (pt_cam1.at<double>(2,0) > 0 && pt_cam2.at<double>(2,0) > 0 && pt_cam1.at<double>(2,0) < 800.0 && err1 < 8.0 && err2 < 8.0) {
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

    PoseRecovery::exportToPLY("../outputs/Truck/before_refinement.ply", points4D, point_colors);

    std::cout << "\n--- Task 3: PnP Pose Estimation (Pure Frame-to-Frame) ---" << std::endl;

    std::vector<int> prev_to_3D_index(kp2.size(), -1);
    cv::Mat global_desc;
    std::vector<int> global_to_3D;

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

    cv::Mat prev_image = img2.clone(); 
    std::vector<cv::KeyPoint> prev_kp = kp2;
    cv::Mat prev_desc = desc2.clone();
    cv::Mat prev_R = best_R.clone();
    cv::Mat prev_t = best_t.clone();

    cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    int f = 2;

    while (cap.read(frame)) {
        if (frameindex % extractionInterval != 0) {
            frameindex++;
            continue;
        }
        std::cout << "\nProcessing Frame " << f << "..." << std::endl;
        cv::Mat current_image = frame.clone();

        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create(8000);
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;
        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);

        if (current_kp.empty() || current_desc.empty()) { frameindex++; f++; continue; }

        // PURE FRAME TO FRAME MATCHING
        std::vector<std::vector<cv::DMatch>> knn_matches;
        flann->knnMatch(current_desc, prev_desc, knn_matches, 2);

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;
        std::vector<int> current_to_3D_index(current_kp.size(), -1);
        std::vector<int> pnp_to_keypoint_idx; 
        std::vector<int> pnp_to_3d_idx;

        std::vector<cv::DMatch> tracking_matches;

        for (size_t i = 0; i < knn_matches.size(); i++) {
            // RELAXED RATIO TEST FOR INDOOR SCENES (0.8 instead of 0.75)
            if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.8f * knn_matches[i][1].distance) {
                int curr_idx = knn_matches[i][0].queryIdx;
                int prev_idx = knn_matches[i][0].trainIdx;
                
                tracking_matches.push_back(knn_matches[i][0]);

                int index_3D = prev_to_3D_index[prev_idx];
                if (index_3D != -1 && current_to_3D_index[curr_idx] == -1) { 
                    double w = points4D.at<double>(3, index_3D);
                    if (std::abs(w) > 1e-7) {
                        object_points_3D.push_back(cv::Point3f(
                            points4D.at<double>(0, index_3D) / w,
                            points4D.at<double>(1, index_3D) / w,
                            points4D.at<double>(2, index_3D) / w
                        ));
                        image_points_2D.push_back(current_kp[curr_idx].pt);
                        pnp_to_keypoint_idx.push_back(curr_idx);
                        pnp_to_3d_idx.push_back(index_3D);
                        current_to_3D_index[curr_idx] = index_3D;
                    }
                }
            }
        }   

        std::cout << "  Frame-to-Frame Matches for PnP: " << object_points_3D.size() << std::endl;
        
        cv::Mat current_R, current_t;

        if (object_points_3D.size() >= 6) { 
            
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat distCoeff = cv::Mat::zeros(4, 1, CV_64F);
            std::vector<int> pnp_inliers;
            
            bool success = cv::solvePnPRansac(object_points_3D, image_points_2D, K, distCoeff, rvec, tvec, 
                                              false, 200, 8.0, 0.99, pnp_inliers, cv::SOLVEPNP_ITERATIVE);
            
            if (success && pnp_inliers.size() >= 8) {
                std::cout << "  Camera pose estimated. Inliers: " << pnp_inliers.size() << std::endl;
                
                cv::Rodrigues(rvec, current_R);
                current_t = tvec.clone();

                ba_cameras_R.push_back(current_R.clone());
                ba_cameras_t.push_back(current_t.clone());
                int current_cam_id = ba_cameras_R.size() - 1;

                // Validate purely proven tracking points
                for (int idx : pnp_inliers) {
                    int c_idx = pnp_to_keypoint_idx[idx];
                    int pt_3d = pnp_to_3d_idx[idx];
                    ba_obs.push_back({ current_cam_id, pt_3d, image_points_2D[idx].x, image_points_2D[idx].y });
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

                // Triangulate tracking matches that don't have 3D points yet
                for (const auto& match : tracking_matches) {
                    int curr_idx = match.queryIdx;
                    int prev_idx = match.trainIdx;
                    if (current_to_3D_index[curr_idx] == -1 && prev_to_3D_index[prev_idx] == -1) {
                        new_pts1.push_back(prev_kp[prev_idx].pt);
                        new_pts2.push_back(current_kp[curr_idx].pt);
                        new_matches_curr_idx.push_back(curr_idx);
                    }
                }

                std::vector<cv::Point2f> final_pts1, final_pts2;
                std::vector<int> final_curr_idx;

                if (new_pts1.size() >= 8) {
                    std::vector<uchar> inliersF;
                    cv::findFundamentalMat(new_pts1, new_pts2, cv::FM_RANSAC, 3.0, 0.99, inliersF);
                    for (size_t i = 0; i < inliersF.size(); i++) {
                        if (inliersF[i]) {
                            final_pts1.push_back(new_pts1[i]);
                            final_pts2.push_back(new_pts2[i]);
                            final_curr_idx.push_back(new_matches_curr_idx[i]);
                        }
                    }
                } else {
                    final_pts1 = new_pts1;
                    final_pts2 = new_pts2;
                    final_curr_idx = new_matches_curr_idx;
                }

                if (!final_pts1.empty()) {
                    cv::Mat new_points4D;
                    PoseRecovery::triangulation(P1, P2, final_pts1, final_pts2, new_points4D);
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
                        cv::Mat pt_cam1 = prev_R * pt3d_3x1 + prev_t;
                        cv::Mat pt_cam2 = current_R * pt3d_3x1 + current_t;

                        cv::Mat pt3d_4x1 = (cv::Mat_<double>(4,1) << x, y, z, 1.0);
                        double err1 = PoseRecovery::calculateReprojectedError(P1, pt3d_4x1, final_pts1[i]);
                        double err2 = PoseRecovery::calculateReprojectedError(P2, pt3d_4x1, final_pts2[i]);

                        if (pt_cam1.at<double>(2,0) > 0 && pt_cam2.at<double>(2,0) > 0 && pt_cam1.at<double>(2,0) < 800.0 && err1 < 8.0 && err2 < 8.0) {
                            valid_new_cols.push_back(new_points4D.col(i));

                            int c_idx = final_curr_idx[i];
                            int new_pt_id = old_map_size + added_points;
                            current_to_3D_index[c_idx] = new_pt_id;

                            ba_obs.push_back({ current_cam_id - 1, new_pt_id, final_pts1[i].x, final_pts1[i].y });
                            ba_obs.push_back({ current_cam_id, new_pt_id, final_pts2[i].x, final_pts2[i].y });

                            added_points++;
                            
                            int px = std::round(final_pts1[i].x);
                            int py = std::round(final_pts1[i].y);
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
                        std::cout << "  Triangulated " << added_points << " New Points. Map Size: " << points4D.cols << std::endl;
                    }
                }

                // Append new valid 3D points to the Global Map for Split B
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

            } else {
                std::cout << "  PnP Failed for this frame (poor tracking)." << std::endl;
            }
        } else {
            std::cout << "  Not enough 3D points to solve PnP. Tracking lost!" << std::endl;
        }
        
        frameindex++;
        f++;
    }
    cap.release();

    std::cout << "Processing complete! Exporting final massive cloud..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Truck/Truck_cloud_colored.ply", points4D, point_colors);

    // Green dots and Red Look Vectors for Split A
    saveCamerasPLY("../outputs/Truck/trajectory_A.ply", ba_cameras_R, ba_cameras_t, 1.0f, 0, 255, 0);

    std::ofstream cam_file("../outputs/Truck/ba_cameras.csv");
    cam_file << "cam_id,rx,ry,rz,tx,ty,tz\n";
    for(size_t i=0; i<ba_cameras_R.size(); i++){
        cv::Mat rvec; cv::Rodrigues(ba_cameras_R[i], rvec);
        cam_file << i << "," << rvec.at<double>(0) << "," << rvec.at<double>(1) << "," << rvec.at<double>(2) << "," << ba_cameras_t[i].at<double>(0) << "," << ba_cameras_t[i].at<double>(1) << "," << ba_cameras_t[i].at<double>(2) << "\n";
    }
    cam_file.close();

    std::ofstream pts_file("../outputs/Truck/ba_points.csv");
    pts_file << "pt_id,x,y,z\n";
    for(int i=0; i<points4D.cols; i++){
        double w = points4D.at<double>(3,i);
        if(std::abs(w)>1e-7) {
            pts_file << i << "," << points4D.at<double>(0,i)/w << "," << points4D.at<double>(1,i)/w << "," << points4D.at<double>(2,i)/w << "\n";
        }
    }
    pts_file.close();

    std::ofstream obs_file("../outputs/Truck/ba_obs.csv");
    obs_file << "cam_id,pt_id,u,v\n";
    for(const auto& o : ba_obs){
        obs_file << o.cam_idx << "," << o.pt_idx << "," << o.x << "," << o.y << "\n";
    }
    obs_file.close();

    std::ofstream int_file("../outputs/Truck/intrinsics.txt");
    int_file << K.at<double>(0,0) << " " << K.at<double>(1,1) << " " << K.at<double>(0,2) << " " << K.at<double>(1,2);
    int_file.close();

    // =========================================================================
    // SPLIT B: Localization 
    // =========================================================================
    std::string videopathB = "../assets/Truck/split_b_Truck.mp4";
    cv::VideoCapture capB(videopathB);
    std::cout << "\nTotal frames to process in Split B: " << (int)(capB.get(cv::CAP_PROP_FRAME_COUNT) / extractionInterval) << std::endl;
    
    int frameindexB = 0;
    int fB = 0;

    std::ofstream metrics_file("../outputs/Truck/metrics_B.csv");
    metrics_file << "Frame,MeanError,Inliers,TotalMatches,InlierRatio\n";

    if (global_desc.empty()) {
        std::cout << "ERROR: Map empty! Skipping Split B." << std::endl;
        return -1;
    }

    std::cout << "Building FLANN index for Global Map (" << global_desc.rows << " descriptors)..." << std::endl;
    cv::Ptr<cv::DescriptorMatcher> global_flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    double total_loc_error = 0.0;
    int successful_loc_frames = 0;

    std::vector<cv::Mat> splitB_R;
    std::vector<cv::Mat> splitB_t;

    while (capB.read(frame)) {
        if (frameindexB % extractionInterval != 0) {
            frameindexB++;
            continue;
        }
        cv::Mat current_image = frame.clone();
        
        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create(8000);
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;

        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);
        if (current_kp.empty() || current_desc.empty()) { frameindexB++; fB++; continue; }

        std::vector<std::vector<cv::DMatch>> knn_matches;
        global_flann->knnMatch(current_desc, global_desc, knn_matches, 2);

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;

        for (size_t i = 0; i < knn_matches.size(); i++) {
            // Relaxed ratio for global indoor matching
            if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.8f * knn_matches[i][1].distance) {
                int curr_idx = knn_matches[i][0].queryIdx; 
                int global_idx = knn_matches[i][0].trainIdx; 
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
        }   

        if (object_points_3D.size() >= 6) { 
            cv::Mat loc_R, loc_t;
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat distCoeff = cv::Mat::zeros(4, 1, CV_64F);
            std::vector<int> pnp_inliers;

            bool success = cv::solvePnPRansac(object_points_3D, image_points_2D, K, distCoeff, rvec, tvec, 
                                              false, 200, 8.0, 0.99, pnp_inliers, cv::SOLVEPNP_ITERATIVE);

            if (success && pnp_inliers.size() >= 8) {
                cv::Rodrigues(rvec, loc_R);
                loc_t = tvec.clone();

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
                    std::cout << "  Split B Frame " << fB << " REJECTED (High Error: " << mean_frame_error << " px)." << std::endl;
                    frameindexB++; fB++; continue; 
                }

                splitB_R.push_back(loc_R.clone());
                splitB_t.push_back(loc_t.clone());

                metrics_file << fB << "," << mean_frame_error << "," << pnp_inliers.size() << "," << object_points_3D.size() << "," << (float)pnp_inliers.size() / object_points_3D.size() << "\n";

                total_loc_error += mean_frame_error;
                successful_loc_frames++;

                std::cout << "  Split B Frame " << fB << " Localized globally! Mean Error: " << mean_frame_error << " px" << std::endl;
            } else {
                std::cout << "  Split B Frame " << fB << " PnP Failed." << std::endl;
            }
        } else {
            std::cout << "  Split B Frame " << fB << " Tracking Lost (Not enough global matches)." << std::endl;
        }

        frameindexB++;
        fB++;
    }
    capB.release();
    metrics_file.close();

    // Blue dots and Red Look Vectors for Split B
    saveCamerasPLY("../outputs/Truck/trajectory_B.ply", splitB_R, splitB_t, 1.0f, 0, 0, 255);

    if (successful_loc_frames > 0) {
        std::cout << "\n>>> Final localization error: " << total_loc_error / successful_loc_frames << " px <<<\n" << std::endl;
    }

    return 0;
}