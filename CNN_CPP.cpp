//
// Created by 任艺伟 on 2020/12/19.
//

#include "CNN_CPP.h"

picture::picture(Mat image) {
    Mat BGR[3];
    resize(image, image, Size(128, 128));
    split(image,BGR);
    this->picture_size = image.rows;
    this->channels = image.channels();
    this->pixel = new float[picture_size * picture_size * channels]();
    int k = 0;
    for(int i=0;i<BGR[2].total();i++){
        pixel[k++] = (float)BGR[2].data[i]/255.0f;
    }//R

    for(int i=0;i<BGR[1].total();i++){
        pixel[k++] = (float)BGR[1].data[i]/255.0f;
    }//G

    for(int i=0;i<BGR[0].total();i++){
        pixel[k++] = (float)BGR[0].data[i]/255.0f;
    }//B

    counter = new atomic_int;
    *(counter) = 1;
}

picture::picture() {
    counter = new atomic_int;
    *(counter) = 0;
}

picture::picture(int picture_size, int channels, float *pixel) {
    this->channels = channels;
    this->picture_size = picture_size;
    this->pixel = new float[picture_size * picture_size * channels]();
    for (int i = 0; i < picture_size * picture_size * channels; i++) {
        *(this->pixel + i) = pixel[i];
    }
    this->counter = new atomic_int;
    *(counter) = 1;
}

Matrix picture::pictureToMatrix(int kernel_size, int channel, int stride, int padding) {
//    cout<<this->channels<<" "<<this->pixel[0]<<endl;
    int out_size = floor((this->picture_size - kernel_size + 2 * padding) / stride + 1);
    Matrix matrix(out_size * out_size, kernel_size * kernel_size * channel+1);
    int current_position = 0;
//    bool judge = (this->picture_size - kernel_size + 2 * padding) % stride == 0;
//    int step = judge ? picture_size - kernel_size + padding : picture_size - kernel_size + padding-1;//有问题
//    for (int m = 0; m <= step; m += m > 0 ? stride : m - padding + stride) {
//        for (int n = 0; n <= step; n += n > 0 ? stride : n - padding + stride) {
//            for (int i = 0; i < this->channels; i++) {
//                if (padding > m) current_position += (padding - m) * kernel_size;
//                for (int j = 0; j < kernel_size; j++) {
//                    if (padding <= m &&m<=picture_size - kernel_size|| padding > m
//                    && j < kernel_size - padding||m>picture_size - kernel_size&&m+j<picture_size) {
//                        if (padding > n) current_position +=padding - n;
//                        for (int k = 0; k < kernel_size; k++) {
//                            if (padding <= n&&n<=picture_size - kernel_size|| padding > n
//                            && k < kernel_size - padding||n>picture_size - kernel_size&&n+k<picture_size)
//                                    matrix.getValue()[current_position++] = this->pixel[n + m*picture_size +i * picture_size *picture_size +j * picture_size + k];
//                            else if(n+k>=kernel_size) current_position+= (n+k-picture_size+1)*1;
//                        }
//                    }else if(m+j>=picture_size) current_position+= (m+j-picture_size+1)*kernel_size;
//                }
//            }
//        }
//    }
//
//    for(int m=0;m+kernel_size<=picture_size+2*padding;m+=stride){
//        for(int n=0;n+kernel_size<=picture_size+2*padding;n+=stride){
//            for(int i=0;i<channel;i++){
//                for(int j=0;j<kernel_size;j++){
//                    for(int k=0;k<kernel_size;k++){
//                        if(m+j-padding>=0&&n+k-padding>=0&&m+j<padding+picture_size&&n+k<picture_size+padding){
//                            matrix.getValue()[current_position++]
//                            = this->pixel[i*picture_size*picture_size+picture_size*(m+j-padding)+n+k-padding];
//                        }else current_position++;
//                    }
//                }
//            }
//        }
//    }


    for(int m=0;m+kernel_size<=picture_size+2*padding;m+=stride){
        for(int n=0;n+kernel_size<=picture_size+2*padding;n+=stride){
            for(int i=0;i<channel;i++){
                for(int j=0;j<kernel_size;j++){
                    for(int k=0;k<kernel_size;k++){
                        if(m+j-padding>=0&&n+k-padding>=0&&m+j<padding+picture_size&&n+k<picture_size+padding){
                            matrix.getValue()[current_position++] = this->pixel[i*picture_size*picture_size+picture_size*(m+j-padding)+n+k-padding];
                        }else current_position++;
                    }
                }
            }
            matrix.getValue()[current_position++] = 1;
        }
    }
    return matrix;
}



Matrix convToMatrix(const conv_param& currentKernel) {
    Matrix conv_result(currentKernel.kernel_size * currentKernel.kernel_size * currentKernel.in_channels+1,currentKernel.out_channels);
    int currentPosition = 0;
    for (int k = 0; k <currentKernel.in_channels* currentKernel.kernel_size * currentKernel.kernel_size; k++) {
        for (int i = 0; i < currentKernel.out_channels; i++) {
            conv_result.getValue()[currentPosition++] = currentKernel.p_weight[
                    i * currentKernel.kernel_size * currentKernel.kernel_size * currentKernel.in_channels + k];
        }
    }
    for (int i = 0; i < currentKernel.out_channels; i++) {
        conv_result.getValue()[currentPosition++] = currentKernel.p_bias[i];
    }
    return conv_result;
}

Matrix BiasToMatrix(const float * bias,int out_size,int out_channels){
    Matrix result(out_channels,out_size*out_size);
    int currentPosition = 0;
    for(int j =0;j<out_channels;j++){
        for(int i=0;i<out_size*out_size;i++){
            result.getValue()[currentPosition++] = bias[j];
        }
    }
    return result;
}

void picture::ConBVReLU(const conv_param &currentKernel) {
    Matrix p_matrix = pictureToMatrix(currentKernel.kernel_size,currentKernel.in_channels,currentKernel.stride,currentKernel.pad);
//    p_matrix.transposition();
    int out_size = floor((this->picture_size - currentKernel.kernel_size + 2 * currentKernel.pad) / currentKernel.stride + 1);
    Matrix c_matrix = convToMatrix(currentKernel);
//    c_matrix.transposition();

//    Matrix CBR_matrix(currentKernel.out_channels,out_size*out_size);
//    Matrix b_matrix = BiasToMatrix(currentKernel.p_bias,out_size,currentKernel.out_channels);
//    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, c_matrix.getColumnNumber(),
//                p_matrix.getRowNumber(), c_matrix.getRowNumber(), 1.0, c_matrix.getValue(), c_matrix.getColumnNumber(),
//                p_matrix.getValue(), p_matrix.getColumnNumber(), 0.0, CBR_matrix.getValue(), p_matrix.getRowNumber());
//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c_matrix.getRowNumber(), p_matrix.getColumnNumber(), c_matrix.getColumnNumber(), 1.0, c_matrix.getValue(), c_matrix.getColumnNumber(), p_matrix.getValue(), p_matrix.getColumnNumber(), 0.0, CBR_matrix.getValue(), p_matrix.getColumnNumber());

//    cout<<p_matrix.getRowNumber()<<" "<<p_matrix.getColumnNumber()<<endl;
//    cout<<c_matrix.getRowNumber()<<" "<<c_matrix.getColumnNumber()<<endl;
//    cout<<b_matrix.getRowNumber()<<" "<<b_matrix.getColumnNumber()<<endl;
//    Matrix CBR_matrix = (p_matrix*c_matrix)+b_matrix;
    Matrix CBR_matrix = p_matrix*c_matrix;



    this->picture_size = out_size;
    this->channels = currentKernel.out_channels;
    if(*(counter) == 1)  {
        delete counter;
        delete [] this->pixel ;
    }else if(*(counter)==0){
        delete counter;
    }
    else (*counter)--;

    this->pixel = CBR_matrix.getValue();
    this->counter = CBR_matrix.getCounter();
    (*this->counter)++;
}

void picture::MaxPooling(int stride, int wide) {
    int out_size = this->picture_size%stride==0?picture_size/stride:this->picture_size / stride + 1;
    float * afterPolling= new float[out_size*out_size*channels]();
    int currentPosition = 0;
    for (int k = 0; k < channels; k++) {
        for (int m = 0; m < picture_size; m += stride) {
            for (int n = 0; n < picture_size; n += stride) {
                float resultValue = 0;
                for (int i = 0; i < wide; i++) {
                    if (m + i < picture_size)
                        for (int j = 0; j < wide; j++) {
                            if (n + j < picture_size) resultValue =
                                    max(resultValue, pixel[k*picture_size*picture_size+(m+i)*picture_size+n+j]);
                        }
                }
                afterPolling[currentPosition++] = resultValue;
            }
        }
    }
    if(*(counter) == 1)  {
        delete counter;
        delete [] this->pixel ;
    }else if(*(counter)==0){
        delete counter;
    }
    else (*counter)--;
    this->picture_size = out_size;
    this->pixel = afterPolling;
    this->counter = new atomic_int;
    *(counter) = 1;
}



void picture::FullyConnected(const fc_param &fc) {
    Matrix weight(fc.out_features,fc.in_features,fc.p_weight,1);
    weight.setCounter(2);
    Matrix pictureNow(picture_size*picture_size*channels,1,pixel,1);
    pictureNow.setCounter(counter);
    (*counter)++;
    Matrix res = weight*pictureNow;

    for(int i=0;i<fc.out_features;i++){
        res.getValue()[i] += fc.p_bias[i];
    }
    float sum=0;
    for(int i=0;i<fc.out_features;i++){
        sum += exp(res.getValue()[i]);
//        cout << res.getValue()[i] << endl;
    }
    for(int i = 0;i<fc.out_features;i++){
        res.getValue()[i] = exp(res.getValue()[i])/sum;
        cout<<res.getValue()[i]<<endl;
        if(i==1){
            if(res.getValue()[1]>0.97) cout<<"你很像人！！"<<endl;
            else if(res.getValue()[1]>0.7) cout<<"你可能是个人"<<endl;
            else if(res.getValue()[1]<0.1) cout<<"你必不是人"<<endl;
        }
    }
}

picture::~picture() {
    if(*(counter) == 1)  {
        delete counter;
        delete [] this->pixel ;
    }else if(*(counter)==0){
        delete counter;
    }
    else (*counter)--;
}
