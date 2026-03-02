#pragma once
#include <opencv2/opencv.hpp>

class Camera{
    public:
        int width;
        int height;
        cv::Mat K;

        Camera(int w, int h);

        cv::Mat getIntrinsic() const;
};