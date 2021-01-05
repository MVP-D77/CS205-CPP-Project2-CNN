//
// Created by 任艺伟 on 2020/12/19.
//

#ifndef PROJECT_CNN_CNN_CPP_H
#define PROJECT_CNN_CNN_CPP_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <cmath>
#include "Matrix.hpp"
#include "face_binary_cls.cpp"

using namespace std;
using namespace cv;

//cl
// ass CNN_CPP {
//};
class picture{
private:
    int picture_size;
    int channels;
    float* pixel;
    atomic_int * counter;
public:
    picture();
    explicit picture(Mat image);
    ~picture();
    float * getPixel() const{return pixel;}
    picture(int picture_size,int channels,float *pixel);
    int getPictureSize() const{return picture_size;}
    Matrix pictureToMatrix(int kernel_size,int channel,int stride = 1,int padding = 0);
    void ConBVReLU(const conv_param& currentKernel);
    void MaxPooling(int length,int wide);
    void FullyConnected(const fc_param& fc);
    int getChannels(){return channels;}
};
Matrix convToMatrix(const conv_param& currentKernel);
Matrix BiasToMatrix(const float * bias,int out_size,int out_channels);
#endif //PROJECT_CNN_CNN_CPP_H
