#include "featurematching.hpp"

FeatureMatcher::FeatureMatcher(){
    detector = cv::SIFT::create();
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); //FLANN based matcher for approximating nearest neighbour 
}

void FeatureMatcher::extractandmatch(const cv::Mat& img1, const cv::Mat& img2,
        std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
        cv::Mat& desc1, cv::Mat& desc2,
        std::vector<cv::DMatch>& good_matches){

    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    //find 2 best matches in image2 for image1
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);

    //lowe ratio test
    const float ratio_threshold = 0.75f;
    for (size_t i = 0; i<knn_matches.size(); i++){
        if(knn_matches[i][0].distance < ratio_threshold*knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}