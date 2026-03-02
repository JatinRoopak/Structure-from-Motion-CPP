#pragma once //let every file access it once
#include "utils.hpp"
#include <vector>

struct Keypoint{
    //cordinates of the corner
    int x;
    int y;
    float response; //how strong the corner is
};

class FeatureExtractor{
    public:
        std::vector<Keypoint> Harriscorner(const Image& img, float threshold);

    private:
        void computeDerivatives(const Image& img, Image& Ix, Image& Iy);
        Image computeHarrisResponse(const Image& Ix, const Image& Iy);
        std::vector<Keypoint> findLocalMaxima(const Image& R, float threshold);
};