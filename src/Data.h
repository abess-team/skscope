//
// Created by Jin Zhu on 2020/2/18.
//
//  #define R_BUILD
#ifndef SRC_DATA_H
#define SRC_DATA_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#endif
#include <iostream>
#include <vector>

#include "utilities.h"
using namespace std;
using namespace Eigen;

template <class T1, class T2, class T3, class T4>
class Data {
   public:
    T4 x;
    T1 y;
    Eigen::VectorXd weight;
    Eigen::VectorXd x_mean;
    Eigen::VectorXd x_norm;
    T3 y_mean;
    int n;
    int p;
    int M;
    int normalize_type;
    int g_num;
    Eigen::VectorXi g_index;
    Eigen::VectorXi g_size;

    Data() = default;

    Data(T4 &x, T1 &y, int normalize_type, Eigen::VectorXd &weight, Eigen::VectorXi &g_index, bool sparse_matrix,
         int beta_size) {
        this->x = x;
        this->y = y;
        this->normalize_type = normalize_type;
        this->n = x.rows();
        this->p = x.cols();
        this->M = y.cols();

        this->weight = weight;
        this->x_mean = Eigen::VectorXd::Zero(this->p);
        this->x_norm = Eigen::VectorXd::Zero(this->p);

        this->g_index = g_index;
        this->g_num = g_index.size();
        Eigen::VectorXi temp = Eigen::VectorXi::Zero(this->g_num);
        for (int i = 0; i < g_num - 1; i++) temp(i) = g_index(i + 1);
        temp(g_num - 1) = beta_size;
        this->g_size = temp - g_index;
    };

};
#endif  // SRC_DATA_H
