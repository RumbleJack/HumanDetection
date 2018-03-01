#pragma once

#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <vector>
using namespace cv;
using namespace std;

void LBP(Mat img, vector<float> &lbpDescriptor);