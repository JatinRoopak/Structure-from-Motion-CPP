#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.hpp"

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

    return 0;

}