#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

struct QuadraticData
{
    const Eigen::MatrixXd Q;
    const Eigen::VectorXd p;

    QuadraticData(const Eigen::MatrixXd &Q, const Eigen::VectorXd &p) : Q(Q), p(p) {}
};

template <class T>
T quadratic_loss(const Matrix<T, -1, 1> &x, pybind11::object const& ex_data)
{
    QuadraticData* data = ex_data.cast<QuadraticData*>();
    return T(0.5 * x.transpose() * data->Q * x) + T(data->p.dot(x));
}
Eigen::VectorXd quadratic_grad(const Eigen::VectorXd &x, pybind11::object const& ex_data)
{
    QuadraticData* data = ex_data.cast<QuadraticData*>();
    return data->Q * x + data->p;
}
Eigen::MatrixXd quadratic_hess(const Eigen::VectorXd &x, pybind11::object const& ex_data)
{
    QuadraticData* data = ex_data.cast<QuadraticData*>();
    return data->Q;
}