#include "camera.hpp"

Camera::Camera(int w, int h) : width(w), height(h){
    K = cv::Mat::zeros(3, 3, CV_64F);

    double fx = std::max(width, height);
    double fy = std::max(width, height);
    double cx = width / 2.0;  
    double cy = height / 2.0; 

    // the intrinsic matrix 
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    K.at<double>(2, 2) = 1.0;
}

cv::Mat Camera::getIntrinsic() const{
    return K;
}