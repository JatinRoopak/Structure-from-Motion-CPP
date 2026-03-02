#include "feature_extractor.hpp"

std::vector<Keypoint> FeatureExtractor::Harriscorner(const Image& img, float threshold) {
    Image Ix, Iy;

    computeDerivatives(img, Ix, Iy);
    Image R = computeHarrisResponse(Ix, Iy);
    std::vector<Keypoint> corners = findLocalMaxima(R, threshold);
    return corners;
}

void FeatureExtractor::computeDerivatives(const Image& img, Image& Ix, Image& Iy){
    Ix.rows = img.rows;
    Ix.cols = img.cols;
    Ix.data.assign(img.rows*img.cols, 0.0f);

    Iy.rows = img.rows;
    Iy.cols = img.cols;
    Iy.data.assign(img.rows*img.cols, 0.0f);

    for (int r=1; r<img.rows-1; ++r){
        for(int c=1; c<img.cols-1; ++c){
            Ix.at(r,c) = img.at(r,c+1) -img.at(r,c-1); //horizontal gradient
            Iy.at(r,c) = img.at(r+1,c)-img.at(r-1,c); //vertical gradient 
        }
    }
}

Image FeatureExtractor::computeHarrisResponse(const Image& Ix, const Image& Iy){
    int rows = Ix.rows;
    int cols = Ix.cols;

    Image R_img;
    R_img.rows = rows;
    R_img.cols = cols;
    R_img.data.assign(rows*cols, 0.0f);

    float k = 0.04f;
    int offset = 1;

    for (int r = offset; r<rows-offset; ++r){
        for (int c = offset; c< cols-offset; ++c){
            float sum_Ix2 = 0.0f;
            float sum_Iy2 = 0.0f;
            float sum_Ixy = 0.0f;

            //for 9 pixels around (r,c)
            for (int wr = -offset; wr <= offset; ++wr){
                for (int wc = -offset; wc<=offset; ++wc){

                    float ix = Ix.at(r + wr, c + wc);
                    float iy = Iy.at(r + wr, c + wc);

                    sum_Ix2 += (ix * ix);
                    sum_Iy2 += (iy * iy);
                    sum_Ixy += (ix * iy);
                }
            }
            float det = (sum_Ix2 * sum_Iy2) - (sum_Ixy*sum_Ixy);
            float trace = sum_Ix2 + sum_Iy2;
            R_img.at(r,c) = det-k*(trace*trace);
        }
    }

    return R_img;
}

std::vector<Keypoint> FeatureExtractor::findLocalMaxima(const Image& R, float threshold){
    std::vector<Keypoint> corners;
    
    int offset = 1;

    for (int r = offset; r < R.rows - offset; ++r){
        for (int c = offset; c < R.cols - offset; ++c){
            
            float current_score = R.at(r,c);

            if (current_score > threshold){
                
                bool is_local_max = true;

                for (int wr = -offset; wr <= offset; ++wr){
                    for (int wc = -offset; wc <= offset; ++wc) {
                        if (wr == 0 && wc == 0) continue; 

                        if (R.at(r + wr, c + wc) >= current_score){
                            is_local_max = false;
                            break; 
                        }
                    }
                    if (!is_local_max) break; 
                }

                if (is_local_max) {
                    Keypoint kp;
                    kp.x = c; 
                    kp.y = r; 
                    kp.response = current_score;
                    corners.push_back(kp);
                }
            }
        } 
    }
    return corners;
}