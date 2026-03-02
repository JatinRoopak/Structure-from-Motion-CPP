#pragma once 
#include <opencv2/opencv.hpp>
#include <vector> 

struct Image {
    int rows;
    int cols;
    std::vector<float> data;

    float& at(int r, int c){
        return data[r*cols + c]; //formula to change from 2D to 1D coordinates
    }

    const float& at(int r, int c) const{ //read only mode
        return data[r*cols +c];
    }
};

Image Grayscale(const cv::Mat& cv_color_img);
