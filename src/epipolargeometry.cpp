#include "epipolargeometry.hpp"

void EpipolarGeometry::extractMatchCoordinates( //filter the coordinates of keypoint for essential matrix function
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) {

    pts1.clear();
    pts2.clear();

    for(const auto& match : matches){
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
}

void EpipolarGeometry::applyRank2(cv::Mat& E){
    cv::Mat w, u, vt;
    cv::SVD::compute(E, w, u, vt, cv::SVD::FULL_UV); //perform svd

    cv::Mat W = cv::Mat::diag(w);
    W.at<double>(2,2) = 0.0; //enforcing the rank 2 on E matrix

    E = u*W*vt;
}

void EpipolarGeometry::essentialMatrix(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    cv::Mat& E1, cv::Mat& E2,
    std::vector<uchar>& inliersMask){
        
    cv::Mat F1 = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT);
    E1 = K.t()*F1*K;
    applyRank2(E1);

    //more robust estimation of E with RANSAC
    double prob = 0.999;
    double threshold = 1.0;
    E2 = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, prob, threshold, inliersMask);
    applyRank2(E2);
}