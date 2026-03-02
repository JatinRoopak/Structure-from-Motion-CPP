#include "utils.hpp"

Image Grayscale(const cv::Mat& color_img) {
    Image final_img;
    final_img.rows = color_img.rows;
    final_img.cols = color_img.cols;
    
    int total_pixels = final_img.rows * final_img.cols;
    final_img.data.resize(total_pixels);

    unsigned char* raw_pixels = color_img.data;

    for (int i = 0; i < total_pixels; ++i) {
        float b = static_cast<float>(raw_pixels[i * 3 + 0]);
        float g = static_cast<float>(raw_pixels[i * 3 + 1]);
        float r = static_cast<float>(raw_pixels[i * 3 + 2]);

        final_img.data[i] = (0.299f * r) + (0.587f * g) + (0.114f * b);
    }

    return final_img;
}