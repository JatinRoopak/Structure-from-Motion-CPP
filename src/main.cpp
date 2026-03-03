#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "featurematching.hpp"
#include "epipolargeometry.hpp"

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
    std::cout << "\nE2 (Robust RANSAC):\n" << E2 << std::endl; 
}