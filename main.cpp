#include <iostream>
#include <opencv2/opencv.hpp>
#include "CNN_CPP.h"
#include "CNN_CPP.cpp"
//#include <omp.h>

using namespace cv;
using namespace std;

int main() {

    Mat newImage = imread("../cx.jpg");
    Mat image;
    resize(newImage,image,Size(128,128),0,0,INTER_LINEAR);
    picture CNNPicture(image);
//

    auto start = std::chrono::steady_clock::now();
    CNNPicture.ConBVReLU(conv_params[0]);
//    Matrix matrix(16,64*64,CNNPicture.getPixel());
//    cout<<matrix;
    CNNPicture.MaxPooling(2,2);
    CNNPicture.ConBVReLU(conv_params[1]);
//    Matrix matrix(32,30*30,CNNPicture.getPixel());
//    cout<<matrix;
    CNNPicture.MaxPooling(2,2);
    CNNPicture.ConBVReLU(conv_params[2]);
//    Matrix matrix(32,8*8,CNNPicture.getPixel());
//    cout<<matrix;
    CNNPicture.FullyConnected(fc_params[0]);
    auto end = std::chrono::steady_clock::now();
    auto duration = std ::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "matrixProduct  , duration = " << duration <<" ms"<< std::endl;
//
    return 0;

}
// Picture 初始化mat 无问题
//float p_weight[12]={1,2,3,4,
//                    5,6,7,8,
//                    9,10,11,12};
//picture newPicture (2,3,p_weight);
//cout<<newPicture.pictureToMatrix(3,3,2,3);

//maxPollingTest 没有问题
//    float pixelMaxPolling[5*5*2] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50};
//    picture maxPollingTest(5,2,pixelMaxPolling);
//    float pixelMaxPolling[4*4*2] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
//    picture maxPollingTest(4,4,pixelMaxPolling);
//    maxPollingTest.MaxPooling(2,2);
//    for(int i=0;i<2*2*2;i++) {
//        cout<<maxPollingTest.getPixel()[i]<<" ";
//        if((i+1)%2==0) cout<<endl;
//    }


//    PictureToMatrix Test 没有问题
//    float firstSituation[4*4*3] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48};
//    picture first(4,3,firstSituation);
//    cout<<first.pictureToMatrix(3,3,2,1);
//    float secondSituation[5*5*3] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75};
//    picture second(5,3,secondSituation);
//    cout<<second.pictureToMatrix(3,3,2,1);
//    cout<<second.pictureToMatrix(3,3,1,0);


//convToMatrix Test 没有问题
//    float pweitht[3*3*3*2] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54};
//    float p_bias[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
//    conv_param cp{1,2,3,3,2,pweitht,p_bias};
//    cout<< convToMatrix(cp);
//    cout<<BiasToMatrix(p_bias,5,16);



