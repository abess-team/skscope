//
// Created by jiangkangkang on 2020/3/9.
//

/**
 * @file utilities.h
 * @brief some utilities for abess package.
 */

#pragma once

#include <Eigen/Eigen>
#include <type_traits>
#include <cfloat>


class UniversalData;

#ifndef	SPDLOG_ACTIVE_LEVEL
    #ifndef NDEBUG
        #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
    #else
        #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
    #endif
    /**
     * 0 = trace < DEBUG < info < WARN < ERROR < critical < off = 6 
     * Usage example:
     *     SPDLOG_ERROR("Some error message with arg: {:03.2f}", 1.23456);
     *     SPDLOG_WARN("Positional args are {1} {0}..", "too", "supported");
     *     SPDLOG_DEBUG("Eigen Matrix\n{}", Eigen::MatrixXd::Ones(3, 4));
     */
#endif
#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/fmt/ostr.h"


constexpr int UNIVERSAL_MODEL = 0;
constexpr int LM_MODEL = 1;
constexpr int LOGISTIC_MODEL = 2;
constexpr int POISSON_MODEL = 3;
constexpr int COX_MODEL = 4;
constexpr int MUL_LM_MODEL = 5;
constexpr int MUL_NOMIAL_MODEL = 6;
constexpr int PCA_MODEL = 7;
constexpr int GAMMA_MODEL = 8;
constexpr int ORDINAL_MODEL = 9;
constexpr int RPCA_MODEL = 10;


void init_spdlog(int console_log_level, int file_log_level, const char* log_file_name);

/**
 * @brief Save the sequential fitting result along the parameter searching.
 * @details All matrix stored here have only one column, and each row correspond to a
 * parameter combination in class Parameters.
 * @tparam Eigen::VectorXd for beta
 * @tparam Eigen::VectorXd for coef0
 */
struct Result {
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>
        beta_matrix; /*!< Each value is the beta corrsponding a parameter combination */
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>
        coef0_matrix; /*!< Each value is the coef0 corrsponding a parameter combination  */
    Eigen::MatrixXd
        ic_matrix; /*!< Each value is the information criterion value corrsponding a parameter combination  */
    Eigen::MatrixXd test_loss_matrix;  /*!< Each value is the test loss corrsponding a parameter combination  */
    Eigen::MatrixXd train_loss_matrix; /*!< Each value is the train loss corrsponding a parameter combination  */
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>
        bd_matrix; /*!< Each value is the sacrfice corrsponding a parameter combination  */
    Eigen::MatrixXd
        effective_number_matrix; /*!< Each value is the effective number corrsponding a parameter combination  */
};

struct FIT_ARG {
    int support_size;
    double lambda;
    Eigen::VectorXd beta_init;
    Eigen::VectorXd coef0_init;
    Eigen::VectorXd bd_init;
    Eigen::VectorXi A_init;

    FIT_ARG(int _support_size, double _lambda, Eigen::VectorXd _beta_init, Eigen::VectorXd _coef0_init, Eigen::VectorXd _bd_init, Eigen::VectorXi _A_init) {
        support_size = _support_size;
        lambda = _lambda;
        beta_init = _beta_init;
        coef0_init = _coef0_init;
        bd_init = _bd_init;
        A_init = _A_init;
    };

    FIT_ARG(){};
};

struct single_parameter {
    int support_size;
    double lambda;

    single_parameter(){};
    single_parameter(int support_size, double lambda) {
        this->support_size = support_size;
        this->lambda = lambda;
    };
};

/**
 * @brief Parameter list
 * @details Store all parameters (e.g. `support_size`, `lambda`), and make a list of their combination.
 * So that the algorithm can extract them one by one.
 */
class Parameters {
   public:
    Eigen::VectorXi support_size_list;
    Eigen::VectorXd lambda_list;
    int s_min = 0;
    int s_max = 0;
    Eigen::Matrix<single_parameter, -1, 1> sequence;

    Parameters() {}
    Parameters(Eigen::VectorXi &support_size_list, Eigen::VectorXd &lambda_list, int s_min, int s_max) {
        this->support_size_list = support_size_list;
        this->lambda_list = lambda_list;
        this->s_min = s_min;
        this->s_max = s_max;
        if (support_size_list.size() > 0) {
            // path = "seq"
            this->build_sequence();
        }
    }

    /**
     * @brief build sequence with all combinations of parameters.
     */
    void build_sequence() {
        // suppose each input vector has size >= 1
        int ind = 0;
        int size1 = (this->support_size_list).size();
        int size2 = (this->lambda_list).size();
        (this->sequence).resize(size1 * size2, 1);

        for (int i1 = 0; i1 < size1; i1++) {  // other order?
            for (int i2 = (1 - pow(-1, i1)) * (size2 - 1) / 2; i2 < size2 && i2 >= 0; i2 = i2 + pow(-1, i1)) {
                int support_size = this->support_size_list(i1);
                double lambda = this->lambda_list(i2);
                single_parameter temp(support_size, lambda);
                this->sequence(ind++) = temp;
            }
        }
    }

    // void print_sequence() {
    //     // for debug
    //     std::cout << "==> Parameter List:" << endl;
    //     for (int i = 0; i < (this->sequence).size(); i++) {
    //         int support_size = (this->sequence(i)).support_size;
    //         double lambda = (this->sequence(i)).lambda;
    //         std::cout << "  support_size = " << support_size << ", lambda = " << lambda << endl;
    //     }
    // }
};

/**
 * @brief return the indexes of all variables in groups in `L`.
 */
Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int beta_size, int N);

/**
 * @brief return part of X, which only contains columns in `ind`.
 */
UniversalData X_seg(UniversalData& X, int n, Eigen::VectorXi& ind, int model_type); 

// template <class UniversalData>
// void X_seg(UniversalData &X, int n, Eigen::VectorXi &ind, UniversalData &X_seg)
// {
//     if (ind.size() == X.cols())
//     {
//         X_seg = X;
//     }
//     else
//     {
//         X_seg.resize(n, ind.size());
//         for (int k = 0; k < ind.size(); k++)
//         {
//             X_seg.col(k) = X.col(ind(k));
//         }
//     }
// };




// void max_k(Eigen::VectorXd &vec, int k, Eigen::VectorXi &result);
void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value);
// Eigen::VectorXi get_value_index(Eigen::VectorXd &nums, double value);
// Eigen::VectorXd vector_slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind);
Eigen::VectorXi vector_slice(Eigen::VectorXi &nums, Eigen::VectorXi &ind);
// Eigen::MatrixXd row_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind);
// Eigen::MatrixXd matrix_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, int axis);

// Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind);
/**
 * @brief complement of A, the whole set is {0..N-1}
 */
Eigen::VectorXi complement(Eigen::VectorXi &A, int N);
// Eigen::VectorXi Ac(Eigen::VectorXi &A, Eigen::VectorXi &U);
/**
 * @brief replace `B` by `C` in `A`
 */
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi &B, Eigen::VectorXi &C);
/**
 * @brief return the indexes of min `k` values in `nums`.
 */
Eigen::VectorXi min_k(Eigen::VectorXd &nums, int k, bool sort_by_value = false);
/**
 * @brief return the indexes of max `k` values in `nums`.
 */
Eigen::VectorXi max_k(Eigen::VectorXd &nums, int k, bool sort_by_value = false);
// Eigen::VectorXi max_k_2(Eigen::VectorXd &vec, int k);

/**
 * @brief Extract `nums` at `ind` position, and store in `A`.
 */
void slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind, Eigen::VectorXd &A);
void slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, Eigen::MatrixXd &A);
void slice(UniversalData& nums, Eigen::VectorXi& ind, UniversalData& A);
/**
 * @brief The inverse action of function slice.
 */
void slice_restore(Eigen::VectorXd &A, Eigen::VectorXi &ind, Eigen::VectorXd &nums, int axis = 0);


void coef_set_zero(int p, int M, Eigen::VectorXd& beta, Eigen::VectorXd& coef0);

// void matrix_sqrt(Eigen::MatrixXd &A, Eigen::MatrixXd &B);
// void matrix_sqrt(Eigen::SparseMatrix<double> &A, Eigen::MatrixXd &B);



// void set_nonzeros(Eigen::MatrixXd &X, Eigen::MatrixXd &x);
// void set_nonzeros(Eigen::SparseMatrix<double> &X, Eigen::SparseMatrix<double> &x);

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::VectorXd &Z,
// Eigen::VectorXd &beta); void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::VectorXd &Z,
// Eigen::VectorXd &beta);

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &Z,
// Eigen::MatrixXd &beta); void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::MatrixXd &Z,
// Eigen::MatrixXd &beta);

// bool check_ill_condition(Eigen::MatrixXd &M);


