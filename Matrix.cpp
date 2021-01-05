//
// Created by 任艺伟 on 2020/12/3.
//

#include "Matrix.hpp"
#include "immintrin.h"
//#include <omp.h>
//mutex m;

float dotProduct(const float *p1, const float * p2, size_t n, size_t temp)
{
    float sum = 0.0f;
    for (size_t i = 0,j=0; i < n; i++,j += temp)
        sum += (p1[i] * p2[j]);
    return sum;
}

Matrix::Matrix() :rowNumber(0),columnNumber(0),valueItem(nullptr){
//    counter = new int;
    counter = new atomic_int;
    *(counter) = 0;
}

Matrix::Matrix(int rowNumber, int columnNumber,float * value,int signal) {
    this->columnNumber = columnNumber;
    this->rowNumber = rowNumber;
    counter = new atomic_int;
    if(signal==0) {
        this->valueItem = new float[rowNumber * columnNumber]();
        if (value != NULL) {
            for (int i = 0; i < rowNumber * columnNumber; i++) {
                *(this->valueItem + i) = *(value + i);
            }
        }
//    counter = new int;
        *(counter) = 1;
    }else this->valueItem = value;
}

Matrix::Matrix(const Matrix &matrix) {
    this->columnNumber = matrix.columnNumber;
    this->rowNumber = matrix.rowNumber;
    this->valueItem = matrix.valueItem;
    this->counter = matrix.counter;
//    mutex m;
//    m.lock();
    (*counter) +=1;
//    m.unlock();
}


Matrix::~Matrix() {
//    cout<<*counter<<endl;
    if(*(counter) == 1) {
//        cout<<"You really delete the pointer of object Matrix"<<endl;
        delete [] valueItem ;
        delete counter;
    }
    else {
//        mutex m;

//        m.lock();
        (*counter) -=1;
//        m.unlock();
    }
}

Matrix & Matrix::operator=(const Matrix &matrix) {
    if(this == &matrix) return *this;

    if(*(counter) == 1)  {
        delete counter;
        delete [] valueItem ;
    }else if(*(counter)==0){
        delete counter;
    }
    else (*counter)--;

    this->counter = matrix.counter;
    this->rowNumber = matrix.rowNumber;
    this->columnNumber = matrix.columnNumber;
    this->valueItem = matrix.valueItem;
    (*counter)++;
    return *this;
}

Matrix Matrix::operator+(const Matrix &matrix) const {
//    cout<<"type A+B :"<<endl;

    if(matrix.rowNumber == this->rowNumber&&matrix.columnNumber==this->columnNumber){
        Matrix resultMatrix(matrix.rowNumber,matrix.columnNumber);
        resultMatrix.valueItem = new float [resultMatrix.rowNumber*resultMatrix.columnNumber]();
        for(int i=0;i<matrix.columnNumber*matrix.rowNumber;i++){
            float result = *(matrix.valueItem+i)+ *(valueItem+i);
            *(resultMatrix.valueItem+i) = result>0?result:0;
        }
        return resultMatrix;
    } else{
        cout<<"You are do wrong things matrix addition must two matrix with same rowNumber and columnNumber";
        exit(0);
    }
}

Matrix Matrix::operator*(const Matrix &matrix) const {
//    cout << "type A*B :" << endl;

    if (this->columnNumber != matrix.rowNumber) {
        cout << "Wrong!! Please input right mxn and nxp matrixes!" << endl;
        exit(0);
    } else {
        if(matrix.columnNumber!=1) {
            Matrix resultMatrix(matrix.columnNumber, this->rowNumber);
            resultMatrix.valueItem = new float[this->rowNumber * matrix.columnNumber]();
            int k = 0;
            for (int j = 0; j < matrix.columnNumber; j++) {
                for (int i = 0; i < this->rowNumber; i++) {
                    float result = dotProduct(this->valueItem + i * this->columnNumber, matrix.valueItem + j,
                                              this->columnNumber, matrix.columnNumber);
                    *(resultMatrix.valueItem + k) = result>0?result:0;
//                    *(resultMatrix.valueItem+k ) = result;
                    k++;
                }
            }
            return resultMatrix;
        } else{
            Matrix resultMatrix(this->rowNumber,matrix.columnNumber);
            resultMatrix.valueItem = new float[this->rowNumber * matrix.columnNumber]();
            int k = 0;
            for (int i = 0; i < this->rowNumber; i++) {
                for (int j = 0; j < matrix.columnNumber; j++) {
                    *(resultMatrix.valueItem + k) =dotProduct(this->valueItem + i * this->columnNumber, matrix.valueItem + j,this->columnNumber, matrix.columnNumber);
                    k++;
                }
            }
            return resultMatrix;
        }
    }
}

ostream & operator<<(ostream& os,const Matrix & matrix){
//    os.setf(ios_base::fixed,ios_base::floatfield);
//    os.precision(3);

    for(int i=0;i<matrix.rowNumber*matrix.columnNumber;i++){
        os<<*(matrix.valueItem+i)<<" ";
        if((i+1)%matrix.columnNumber==0) os<<endl;
    }

    return os;
}


float dotproduct1(const float *p1, const float * p2, size_t n, size_t temp)
{
    float sum = 0.0f;
    for (size_t i = 0,j=0; i < n; i++,j += temp)
        sum += (p1[i] * p2[j]);
    return sum;
}


float dotproduct2(const float *p1, const float * p2, size_t n,size_t temp)
{
    float sum = 0.0f;
    for (size_t i = 0 ,j=0; i +8<= n; i+=8,j = j+8*temp)
    {
        sum += (p1[i] * p2[j]);
        sum += (p1[i+1] * p2[j+1*temp]);
        sum += (p1[i+2] * p2[j+2*temp]);
        sum += (p1[i+3] * p2[j+3*temp]);
        sum += (p1[i+4] * p2[j+4*temp]);
        sum += (p1[i+5] * p2[j+5*temp]);
        sum += (p1[i+6] * p2[j+6*temp]);
        sum += (p1[i+7] * p2[j+7*temp]);
    }

    size_t current1 = n/8*8;
    size_t current2 = n/8*8*temp;
    for(size_t i=0,j=0;i<n%8;i++,j+=temp){
        sum += (p1[current1+i]*p2[current2+j]);
    }
    return sum;
}


float dotproduct3(const float *p1, const float * p2, size_t n,size_t temp)
{
    float * mid = new float [8]();
    float sum[8] = {0};
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    for (size_t i = 0; i+8 <= n; i+=8)
    {
        for(int j=0;j<8;j++){
            mid[j] = *(p2+(j+i)*temp);
        }
        a = _mm256_load_ps(p1+i);
        b = _mm256_load_ps(mid);
        c =  _mm256_add_ps(c, _mm256_mul_ps(a, b));
    }
    _mm256_store_ps(sum, c);
    float sumRes = 0.0f;
    sumRes+= (sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]);

    size_t current1 = n/8*8;
    size_t current2 = n/8*8*temp;
    for(size_t i=0,j=0;i<n%8;i++,j+=temp){
        sumRes += (p1[current1+i]*p2[current2+j]);
    }

    return sumRes;
}


void Matrix:: transposition(){
    for(int i=0;i<rowNumber;i++){
        for(int j=i+1;j<columnNumber;j++){
            float temp = *(valueItem+i*rowNumber+j);
            *(valueItem+i*rowNumber+j) = *(valueItem+ j*rowNumber+i);
            *(valueItem+j*rowNumber+i) = temp;
        }
    }
    int temp = rowNumber;
    rowNumber = columnNumber;
    columnNumber = temp;
}

float dotproduct4(const float *p1, const float * p2, size_t n, long long int temp)
{
    float * mid = new float [8]();
    float sum[8] = {0};
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

//    omp_set_num_threads(4);
#pragma omp parallel for
    for (size_t i = 0; i+8 <= n; i+=8)
    {
        for(int j=0;j<8;j++){
            mid[j] = *(p2+(j+i)*temp);
        }
        a = _mm256_load_ps(p1+i);
        b = _mm256_load_ps(mid);
        c =  _mm256_add_ps(c, _mm256_mul_ps(a, b));
    }

    _mm256_store_ps(sum, c);
    float sumRes = 0.0f;
    sumRes+= (sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]);
    size_t current1 = n/8*8;
    size_t current2 = n/8*8*temp;
    for(size_t i=0,j=0;i<n%8;i++,j+=temp){
        sumRes += (p1[current1+i]*p2[current2+j]);
    }
    return sumRes;
}

