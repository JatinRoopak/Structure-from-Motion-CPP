#include <iostream>
#include <fstream>
#include <deque>
#include <set>
#include <algorithm>
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

struct FrameContext {
    cv::Mat desc;
    std::vector<cv::KeyPoint> kp;
    std::vector<int> to_3D_index;
};

int main() {
    std::string videopath = "../assets/Truck/split_a_Truck.mp4";

    cv::VideoCapture cap(videopath);
    cv::Mat frame;
    int frameindex = 0;

    int extractionInterval = 20; 
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

    // Hard Math: 0.7*w Intrinsics
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 0.7 * img1.cols;
    K.at<double>(1, 1) = 0.7 * img1.cols;
    K.at<double>(0, 2) = img1.cols / 2.0;
    K.at<double>(1, 2) = img1.rows / 2.0;
    std::cout << "\nUsing Corrected Intrinsics (0.7 * w):\n" << K << std::endl;

    // SIFT Uncapped: Let it find what it needs on 4K images
    cv::Ptr<cv::SIFT> init_sift = cv::SIFT::create();
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
    std::vector<cv::Point3f> trajectory_A;
    std::vector<cv::Point3f> trajectory_B;

    ba_cameras_R.push_back(cv::Mat::eye(3,3,CV_64F));
    ba_cameras_t.push_back(cv::Mat::zeros(3,1,CV_64F));
    ba_cameras_R.push_back(best_R.clone());
    ba_cameras_t.push_back(best_t.clone());

    trajectory_A.push_back(cv::Point3f(0, 0, 0));
    cv::Mat C1_loc = -best_R.t() * best_t;
    trajectory_A.push_back(cv::Point3f(C1_loc.at<double>(0), C1_loc.at<double>(1), C1_loc.at<double>(2)));

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
        
        // Hard Math: Depth Cap of 200
        if (z > 0 && z < 200.0 && pt_cam2.at<double>(2,0) > 0 && err1 < 8.0 && err2 < 8.0) {
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

    std::cout << "\nExporting Task 2 initial cloud (Before Refinement)..." << std::endl;
    PoseRecovery::exportToPLY("../outputs/Truck/before_refinement.ply", points4D, point_colors);

    std::cout << "\n--- Task 3: PnP Pose Estimation (Video Loop) ---" << std::endl;

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
    cv::Mat prev_R = best_R.clone();
    cv::Mat prev_t = best_t.clone();

    std::deque<FrameContext> sliding_window;
    FrameContext init_ctx;
    init_ctx.desc = desc2.clone();
    init_ctx.kp = kp2;
    init_ctx.to_3D_index = prev_to_3D_index;
    sliding_window.push_back(init_ctx);
      
    int f = 2;
    cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    while (cap.read(frame)) {
        if (frameindex % extractionInterval != 0) {
            frameindex++;
            continue;
        }
        std::cout << "\nProcessing Frame " << f << "..." << std::endl;
        cv::Mat current_image = frame.clone();

        // SIFT Uncapped
        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;
        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);

        if (current_kp.empty() || current_desc.empty()) { frameindex++; f++; continue; }

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;
        std::vector<int> pnp_to_keypoint_idx; 
        std::vector<int> pnp_to_3d_idx;

        std::set<int> seen_3d_points;
        std::vector<int> temp_2d_matched(current_kp.size(), 0);

        for (auto it = sliding_window.rbegin(); it != sliding_window.rend(); ++it) {
            const auto& ref_frame = *it;
            std::vector<std::vector<cv::DMatch>> knn_matches;
            flann->knnMatch(current_desc, ref_frame.desc, knn_matches, 2);

            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance) {
                    int curr_idx = knn_matches[i][0].queryIdx;
                    int ref_idx = knn_matches[i][0].trainIdx;
                    int index_3D = ref_frame.to_3D_index[ref_idx];

                    if (index_3D != -1 && temp_2d_matched[curr_idx] == 0) { 
                        if (seen_3d_points.find(index_3D) == seen_3d_points.end()) {
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
                                
                                seen_3d_points.insert(index_3D);
                                temp_2d_matched[curr_idx] = 1;
                            }
                        }
                    }
                }
            }
        }   

        std::cout << "  Sliding Window Matches for PnP: " << object_points_3D.size() << std::endl;
        
        cv::Mat current_R, current_t;
        std::vector<int> current_to_3D_index(current_kp.size(), -1);

        if (object_points_3D.size() >= 6) { 
            
            std::vector<int> pnp_inliers;
            bool success = PnP_Estimator::estimatePose(object_points_3D, image_points_2D, K, current_R, current_t, pnp_inliers);
            
            if (success && pnp_inliers.size() >= 8) {
                std::cout << "  Camera pose estimated. Inliers: " << pnp_inliers.size() << std::endl;
                
                ba_cameras_R.push_back(current_R.clone());
                ba_cameras_t.push_back(current_t.clone());
                int current_cam_id = ba_cameras_R.size() - 1;

                cv::Mat C = -current_R.t() * current_t;
                trajectory_A.push_back(cv::Point3f(C.at<double>(0), C.at<double>(1), C.at<double>(2)));

                for (int idx : pnp_inliers) {
                    int c_idx = pnp_to_keypoint_idx[idx];
                    int pt_3d = pnp_to_3d_idx[idx];
                    current_to_3D_index[c_idx] = pt_3d;
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

                std::vector<std::vector<cv::DMatch>> local_knn;
                flann->knnMatch(sliding_window.back().desc, current_desc, local_knn, 2);
                
                std::vector<cv::Point2f> new_pts1, new_pts2;
                std::vector<int> new_matches_curr_idx; 

                for (size_t i = 0; i < local_knn.size(); i++) {
                    if (local_knn[i].size() >= 2 && local_knn[i][0].distance < 0.75f * local_knn[i][1].distance) {
                        int p_idx = local_knn[i][0].queryIdx;
                        int c_idx = local_knn[i][0].trainIdx;
                        
                        if (current_to_3D_index[c_idx] == -1 && sliding_window.back().to_3D_index[p_idx] == -1) {
                            new_pts1.push_back(sliding_window.back().kp[p_idx].pt);
                            new_pts2.push_back(current_kp[c_idx].pt);
                            new_matches_curr_idx.push_back(c_idx);
                        }
                    }
                }

                // RESTORED RANSAC FILTER: Do not let garbage geometry into the map
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
                    std::cout << "  RANSAC filtered " << new_pts1.size() << " to " << final_pts1.size() << " valid map expansion points." << std::endl;
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

                        if (pt_cam1.at<double>(2,0) > 0 && pt_cam2.at<double>(2,0) > 0 && z < 200.0 && err1 < 8.0 && err2 < 8.0) {
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

                for (size_t i = 0; i < current_kp.size(); i++) {
                    if (current_to_3D_index[i] != -1) {
                        global_desc.push_back(current_desc.row(i));
                        global_to_3D.push_back(current_to_3D_index[i]);
                    }
                }

                FrameContext curr_ctx;
                curr_ctx.desc = current_desc.clone();
                curr_ctx.kp = current_kp;
                curr_ctx.to_3D_index = current_to_3D_index;
                sliding_window.push_back(curr_ctx);

                if (sliding_window.size() > 5) {
                    sliding_window.pop_front();
                }

                prev_image = current_image.clone();
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

    saveTrajectoryPLY("../outputs/Truck/trajectory_A.ply", trajectory_A, 0, 255, 0);

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

    while (capB.read(frame)) {
        if (frameindexB % extractionInterval != 0) {
            frameindexB++;
            continue;
        }
        cv::Mat current_image = frame.clone();
        
        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();
        std::vector<cv::KeyPoint> current_kp;
        cv::Mat current_desc;

        sift_detector->detectAndCompute(current_image, cv::noArray(), current_kp, current_desc);
        if (current_kp.empty() || current_desc.empty()) { frameindexB++; fB++; continue; }

        std::vector<std::vector<cv::DMatch>> knn_matches;
        global_flann->knnMatch(current_desc, global_desc, knn_matches, 2);

        std::vector<cv::Point3f> object_points_3D;
        std::vector<cv::Point2f> image_points_2D;

        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance) {
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
                    std::cout << "  Split B Frame " << fB << " REJECTED (High Error: " << mean_frame_error << " px)." << std::endl;
                    frameindexB++; fB++; continue; 
                }

                cv::Mat C_loc = -loc_R.t() * loc_t;
                trajectory_B.push_back(cv::Point3f(C_loc.at<double>(0), C_loc.at<double>(1), C_loc.at<double>(2)));

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

    saveTrajectoryPLY("../outputs/Truck/trajectory_B.ply", trajectory_B, 255, 0, 0);

    if (successful_loc_frames > 0) {
        std::cout << "\n>>> Final localization error: " << total_loc_error / successful_loc_frames << " px <<<\n" << std::endl;
    }

    return 0;
}