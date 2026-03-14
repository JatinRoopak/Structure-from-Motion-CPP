#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <opencv2/opencv.hpp>

bool readPLY(const std::string& filename, std::vector<cv::Point3f>& cloud) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    bool header_ended = false;
    int vertex_count = 0;
    
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string t1, t2;
            iss >> t1 >> t2 >> vertex_count;
        }
        if (line == "end_header") {
            header_ended = true;
            break;
        }
    }
    
    if (!header_ended) return false;
    
    for (int i = 0; i < vertex_count; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z) {
            cloud.push_back(cv::Point3f(x, y, z));
        }
    }
    return true;
}

bool readColmapPoints(const std::string& filename, std::vector<cv::Point3f>& cloud) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int point3D_id;
        float x, y, z;
        if (iss >> point3D_id >> x >> y >> z) {
            cloud.push_back(cv::Point3f(x, y, z));
        }
    }
    return true;
}

void normalizeCloud(std::vector<cv::Point3f>& cloud) {
    if (cloud.empty()) return;
    cv::Point3f centroid(0,0,0);
    for (const auto& p : cloud) {
        centroid.x += p.x; centroid.y += p.y; centroid.z += p.z;
    }
    centroid.x /= cloud.size(); centroid.y /= cloud.size(); centroid.z /= cloud.size();
    
    float max_dist = 0;
    for (auto& p : cloud) {
        p.x -= centroid.x; p.y -= centroid.y; p.z -= centroid.z;
        float d = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        if (d > max_dist) max_dist = d;
    }
    
    if (max_dist > 0) {
        for (auto& p : cloud) {
            p.x /= max_dist; p.y /= max_dist; p.z /= max_dist;
        }
    }
}

float computeOneWayChamfer(const std::vector<cv::Point3f>& cloudA, const std::vector<cv::Point3f>& cloudB) {
    cv::Mat matB(cloudB.size(), 3, CV_32F);
    for(size_t i=0; i<cloudB.size(); i++){
        matB.at<float>(i,0) = cloudB[i].x;
        matB.at<float>(i,1) = cloudB[i].y;
        matB.at<float>(i,2) = cloudB[i].z;
    }
    
    cv::flann::Index flann_index(matB, cv::flann::KDTreeIndexParams(4));
    
    float total_dist = 0;
    std::vector<int> indices(1);
    std::vector<float> dists(1); 
    
    for(const auto& p : cloudA){
        std::vector<float> query = {p.x, p.y, p.z};
        flann_index.knnSearch(query, indices, dists, 1);
        total_dist += std::sqrt(dists[0]); 
    }
    return total_dist / cloudA.size();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./chamfer_eval <cloud1.ply or points3D.txt> <cloud2.ply>\n";
        return -1;
    }
    
    std::string file1 = argv[1];
    std::string file2 = argv[2];
    std::vector<cv::Point3f> cloud1, cloud2;
    
    std::cout << "Loading Cloud 1: " << file1 << "..." << std::endl;
    if (file1.find(".txt") != std::string::npos) readColmapPoints(file1, cloud1);
    else readPLY(file1, cloud1);
    
    std::cout << "Loading Cloud 2: " << file2 << "..." << std::endl;
    readPLY(file2, cloud2);
    
    if(cloud1.empty() || cloud2.empty()){
        std::cout << "ERROR: Failed to read one or both clouds! (Are they binary?)\n";
        return -1;
    }
    
    std::cout << "Normalizing clouds..." << std::endl;
    normalizeCloud(cloud1);
    normalizeCloud(cloud2);
    
    std::cout << "Computing KD-Tree nearest neighbors..." << std::endl;
    float d1 = computeOneWayChamfer(cloud1, cloud2);
    float d2 = computeOneWayChamfer(cloud2, cloud1);
    
    float chamfer = (d1 + d2) / 2.0f;
    std::cout << "\n>>> CHAMFER DISTANCE: " << chamfer << " <<<\n" << std::endl;
    
    return 0;
}