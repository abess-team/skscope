/**
 * author: Kangkang Jiang
 * Copyright (C) 2023 abess-team
 * Licensed under the MIT License.
 */

#include <Eigen/Eigen>

#include <string.h>

#include <algorithm>
#include <random>

#include "utilities.h"
#include "UniversalData.h"

using namespace std;
using namespace Eigen;

std::mt19937 GLOBAL_RNG(1);

Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &gindex, Eigen::VectorXi &gsize, int beta_size, int N)
{
    if (L.size() == N)
    {
        return Eigen::VectorXi::LinSpaced(beta_size, 0, beta_size - 1);
    }
    else
    {
        int mark = 0;
        Eigen::VectorXi ind = Eigen::VectorXi::Zero(beta_size);

        for (int i = 0; i < L.size(); i++)
        {
            ind.segment(mark, gsize(L(i))) =
                Eigen::VectorXi::LinSpaced(gsize(L(i)), gindex(L(i)), gindex(L(i)) + gsize(L(i)) - 1);
            mark = mark + gsize(L(i));
        }
        return ind.head(mark).eval();
    }
}

UniversalData X_seg(UniversalData &X, int n, Eigen::VectorXi &ind, int model_type)
{
    return X.slice_by_para(ind);
}

void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value)
{
    if (ind.size() != 0)
    {
        for (int i = 0; i < ind.size(); i++)
        {
            nums(ind(i)) = value;
        }
    }
    return;
}

// replace B by C in A
// to do : binary search
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi &B, Eigen::VectorXi &C)
{
    unsigned int k;
    for (unsigned int i = 0; i < B.size(); i++)
    {
        for (k = 0; k < A.size(); k++)
        {
            if (B(i) == A(k))
            {
                A(k) = C(i);
                break;
            }
        }
    }
    sort(A.data(), A.data() + A.size());
    return A;
}

Eigen::VectorXi min_k(Eigen::VectorXd &vec, int k, bool sort_by_value)
{
    Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1); // [0 1 2 3 ... N-1]
    // shuffle index to avoid repeat results when there are several equal values in vec
    std::shuffle(ind.data(), ind.data() + ind.size(), GLOBAL_RNG);

    auto rule = [vec](int i, int j) -> bool
    { return vec(i) < vec(j); }; // sort rule
    std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
    if (sort_by_value)
    {
        std::sort(ind.data(), ind.data() + k, rule);
    }
    else
    {
        std::sort(ind.data(), ind.data() + k);
    }

    return ind.head(k).eval();
}

Eigen::VectorXi max_k(Eigen::VectorXd &vec, int k, bool sort_by_value)
{
    Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1); // [0 1 2 3 ... N-1]
    // shuffle index to avoid repeat results when there are several equal values in vec
    std::shuffle(ind.data(), ind.data() + ind.size(), GLOBAL_RNG);

    auto rule = [vec](int i, int j) -> bool
    { return vec(i) > vec(j); }; // sort rule
    std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
    if (sort_by_value)
    {
        std::sort(ind.data(), ind.data() + k, rule);
    }
    else
    {
        std::sort(ind.data(), ind.data() + k);
    }
    return ind.head(k).eval();
}

// Eigen::VectorXi max_k_2(Eigen::VectorXd &vec, int k)
// {
//     Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1); //[0 1 2 3 ... N-1]
//     auto rule = [vec](int i, int j) -> bool
//     {
//         return vec(i) > vec(j);
//     }; // sort rule
//     std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
//     std::sort(ind.data(), ind.data() + k);
//     return ind.head(k).eval();
// }

// complement
Eigen::VectorXi complement(Eigen::VectorXi &A, int N)
{
    int A_size = A.size();
    if (A_size == 0)
    {
        return Eigen::VectorXi::LinSpaced(N, 0, N - 1);
    }
    else if (A_size == N)
    {
        Eigen::VectorXi I(0);
        return I;
    }
    else
    {
        Eigen::VectorXi I(N - A_size);
        int cur_index = 0;
        int A_index = 0;
        for (int i = 0; i < N; i++)
        {
            if (A_index >= A_size)
            {
                I(cur_index) = i;
                cur_index += 1;
                continue;
            }
            if (i != A(A_index))
            {
                I(cur_index) = i;
                cur_index += 1;
            }
            else
            {
                A_index += 1;
            }
        }
        return I;
    }
}

// // Ac
// Eigen::VectorXi Ac(Eigen::VectorXi &A, Eigen::VectorXi &U)
// {
//     int A_size = A.size();
//     int N = U.size();
//     if (A_size == 0)
//     {
//         return U;
//     }
//     else if (A_size == N)
//     {
//         Eigen::VectorXi I(0);
//         return I;
//     }
//     else
//     {
//         Eigen::VectorXi I(N - A_size);
//         int cur_index = 0;
//         int A_index = 0;
//         for (int i = 0; i < N; i++)
//         {
//             if (A_index < A.size() && U(i) == A(A_index))
//             {
//                 A_index += 1;
//                 continue;
//             }
//             else
//             {
//                 I(cur_index) = U(i);
//                 cur_index += 1;
//             }
//         }
//         return I;
//     }
// }

void slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind, Eigen::VectorXd &A)
{
    if (ind.size() == 0)
    {
        A = Eigen::VectorXd::Zero(0);
    }
    else
    {
        A = Eigen::VectorXd::Zero(ind.size());
        for (int i = 0; i < ind.size(); i++)
        {
            A(i) = nums(ind(i));
        }
    }
}

void slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, Eigen::MatrixXd &A)
{
    A = Eigen::MatrixXd::Zero(ind.size(), nums.cols());
    if (ind.size() != 0)
    {
        for (int i = 0; i < ind.size(); i++)
        {
            A.row(i) = nums.row(ind(i));
        }
    }
}

void slice(UniversalData &nums, Eigen::VectorXi &ind, UniversalData &A)
{
    A = nums.slice_by_sample(ind);
}

void slice_restore(Eigen::VectorXd &A, Eigen::VectorXi &ind, Eigen::VectorXd &nums, int axis)
{
    if (ind.size() == 0)
    {
        nums = Eigen::VectorXd::Zero(nums.size());
    }
    else
    {
        nums = Eigen::VectorXd::Zero(nums.size());
        for (int i = 0; i < ind.size(); i++)
        {
            nums(ind(i)) = A(i);
        }
    }
    return;
}

void coef_set_zero(int p, int M, Eigen::VectorXd &beta, Eigen::VectorXd &coef0)
{
    beta = Eigen::VectorXd::Zero(p);
    coef0 = Eigen::VectorXd::Zero(M);
    return;
}

// Eigen::SparseMatrix<double> array_product(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &B, int axis)
// {
//     for (int i = 0; i < A.cols(); i++)
//     {
//         A.col(i) = A.col(i) * B;
//     }
//     return A;
// }

// void matrix_sqrt(Eigen::MatrixXd &A, Eigen::MatrixXd &B)
// {
//     A.sqrt().evalTo(B);
// }

// void matrix_sqrt(Eigen::SparseMatrix<double> &A, Eigen::MatrixXd &B)
// {
//     if (A.rows() == 1)
//     {
//         B = Eigen::MatrixXd::Ones(1, 1) * A.cwiseSqrt();
//     }
//     else
//     {
//         Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>>
//             adjoint_eigen_solver(A);
//         // const auto &eigenvalues = adjoint_eigen_solver.eigenvalues();
//         // CHECK_GT(eigenvalues.minCoeff(), -1e-5) //R.minCoeff() 意思是 min(R(:))最小值
//         //     << "MatrixSqrt failed with negative eigenvalues: "
//         //     << eigenvalues.transpose();

//         B = adjoint_eigen_solver.eigenvectors() * (adjoint_eigen_solver.eigenvalues().cwiseSqrt().asDiagonal()) *
//         adjoint_eigen_solver.eigenvectors().transpose();
//         //    .cwiseMax(Eigen::Matrix<FloatType, N, 1>::Zero()) //R.cwiseMax(P)
//         //    .cwiseSqrt()  // R.cwiseSqrt()
//         //    .asDiagonal() * // x.asDiagonal()
//         //    adjoint_eigen_solver.eigenvectors().transpose();
//     }
// }

// void set_nonzeros(Eigen::MatrixXd &X, Eigen::MatrixXd &x)
// {
//     return;
// }

// void set_nonzeros(Eigen::SparseMatrix<double> &X, Eigen::SparseMatrix<double> &x)
// {
//     X.reserve(x.nonZeros() + x.rows());
// }

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::VectorXd &Z,
// Eigen::VectorXd &beta)
// {
//     // Eigen::SparseMatrix<double> XTX = X_new.transpose() * X;

//     // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
//     // solver.compute(X_new.transpose() * X);
//     // beta = solver.solve(X_new.transpose() * Z);
//     Eigen::MatrixXd XTX = X_new.transpose() * X;
//     beta = (XTX).ldlt().solve(X_new.transpose() * Z);
//     return;
// }

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &Z,
// Eigen::MatrixXd &beta)
// {
//     // Eigen::SparseMatrix<double> XTX = X_new.transpose() * X;

//     // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
//     // solver.compute(X_new.transpose() * X);
//     // beta = solver.solve(X_new.transpose() * Z);

//     Eigen::MatrixXd XTX = X_new.transpose() * X;

//     beta = (XTX).ldlt().solve(X_new.transpose() * Z);

//     return;
// }

// void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::VectorXd &Z, Eigen::VectorXd &beta)
// {
//     beta = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     return;
// }

// void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::MatrixXd &Z, Eigen::MatrixXd &beta)
// {
//     beta = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     return;
// }

// bool check_ill_condition(Eigen::MatrixXd &M){
//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
//     double l1 = svd.singularValues()(0);
//     double l2 = svd.singularValues()(svd.singularValues().size()-1);
//     return ((l2 == 0 || l1 / l2 > 1e+10) ? true : false);
// }

void init_spdlog(int console_log_level, int file_log_level, const char *log_file_name)
{
    std::vector<spdlog::sink_ptr> sinks;

    if (console_log_level != SPDLOG_LEVEL_OFF)
    {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::level_enum(console_log_level));
        console_sink->set_pattern("[%T.%e][%s:%#, %!][%^%l%$]: %v");
        sinks.push_back(console_sink);
    }

    if (file_log_level != SPDLOG_LEVEL_OFF)
    {
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_file_name, 1024 * 1024 * 10, 10);
        rotating_sink->set_level(spdlog::level::level_enum(file_log_level));
        rotating_sink->set_pattern("[%Y/%m/%d][%T.%e][elapsed %o][Process %P Thread %t][%s:%#, %!][%^%l%$]: %v");
        sinks.push_back(rotating_sink);
    }

    auto multi_sink_logger = std::make_shared<spdlog::logger>("multi_sink_logger", sinks.begin(), sinks.end());
    multi_sink_logger->set_level(spdlog::level::level_enum(SPDLOG_ACTIVE_LEVEL));
    spdlog::set_default_logger(multi_sink_logger);
}
