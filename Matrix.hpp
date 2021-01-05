//
// Created by 任艺伟 on 2020/12/3.
//

#ifndef ASSIGNMENT4_MATRIX_HPP
#define ASSIGNMENT4_MATRIX_HPP
#include <iostream>
#include <atomic>
#include <thread>
#include "cblas.h"
using namespace std;

class Matrix {
private:
    int rowNumber;
    int columnNumber;
    float * valueItem;
//    int * counter;
    atomic_int * counter;
public:
    Matrix();
    Matrix(int rowNumber,int columnNumber,float * value =NULL,int signal =0);
    ~Matrix();
    void transposition();
    Matrix(const Matrix& matrix);
    Matrix& operator =(const Matrix & matrix);
//    void transposition();
//    float getDet() const;
    void setRowColumn(int row){rowNumber = row;}
    void setCounter(int number){*(counter) = number;}
    void setCounter(atomic_int *number){counter = number;}
    void setColumeColumn(int column){columnNumber = column;}
    float * getValue(){return valueItem;}
    int getRowNumber() const{return rowNumber;}
    int getColumnNumber() const{return columnNumber;}
    atomic_int * getCounter() const {return counter;}
    virtual Matrix operator *(const Matrix & matrix) const;
    Matrix operator +(const Matrix & matrix) const;
    friend ostream & operator<<(ostream& os,const Matrix & matrix);
};
float dotproduct1(const float *p1, const float * p2, size_t n, size_t temp);
float dotproduct2(const float *p1, const float * p2, size_t n,size_t temp);
float dotproduct3(const float *p1, const float * p2, size_t n,size_t temp);
float dotproduct4(const float *p1, const float * p2, size_t n, long long int temp);
#endif //ASSIGNMENT4_MATRIX_HPP
