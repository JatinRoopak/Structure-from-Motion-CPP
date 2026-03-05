#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "featurematching.hpp"
#include "epipolargeometry.hpp"
#include "PoseRecovery.hpp"

int main() {
    std::string videopath = "../assets/split_a_truck.mp4";

    cv::VideoCapture cap(videopath);
    
    std::vector<cv::Mat> extractFrames;
    cv::Mat frame;

    int frameindex = 0;
    int extractionInterval = 30;

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

}