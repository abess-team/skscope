/**
 * author: Zezhi Wang
 * Copyright (C) 2023 abess-team
 * Licensed under the MIT License.
 */

#include "UniversalData.h"
#include "utilities.h"
#include <autodiff/forward/dual/eigen.hpp>
#include <iostream>
using namespace std;
using Eigen::Map;
using Eigen::Matrix;

UniversalData::UniversalData(Eigen::Index model_size, Eigen::Index sample_size, pybind11::object &data, UniversalModel *model, ConvexSolver convex_solver)
    : model(model), convex_solver(convex_solver), sample_size(sample_size), model_size(model_size), effective_size(model_size)
{
    this->effective_para_index = VectorXi::LinSpaced(model_size, 0, model_size - 1);
    this->data = shared_ptr<pybind11::object>(new pybind11::object(data));
}

UniversalData UniversalData::slice_by_para(const VectorXi &target_para_index)
{
    UniversalData tem(*this);
    tem.effective_para_index = this->effective_para_index(target_para_index);
    tem.effective_size = target_para_index.size();
    return tem;
}

UniversalData UniversalData::slice_by_sample(const VectorXi &target_sample_index)
{
    UniversalData tem(*this);
    tem.sample_size = target_sample_index.size();
    tem.data = shared_ptr<pybind11::object>(new pybind11::object(model->slice_by_sample(*data, target_sample_index)), model->deleter);
    return tem;
}

Eigen::Index UniversalData::cols() const
{
    return effective_size;
}

Eigen::Index UniversalData::rows() const
{
    return sample_size;
}

const VectorXi &UniversalData::get_effective_para_index() const
{
    return effective_para_index;
}

double UniversalData::loss(const VectorXd &effective_para)
{
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;
    return model->loss(complete_para, *this->data);
}

double UniversalData::loss_and_gradient(const VectorXd &effective_para, VectorXd &gradient)
{
    double value = 0.0;
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;

    if (model->gradient_user_defined)
    {
        // Note: using complete_para to store gradient isn't a good idea, just for saving memory
        tie(value, complete_para) = model->gradient_user_defined(complete_para, *this->data);
        gradient = complete_para(this->effective_para_index);
    }
    else
    { // forward autodiff
        dual value_dual;
        VectorXdual effective_para_dual = effective_para;
        auto func = [this, &complete_para](VectorXdual const &effective_para_dual)
        {
            VectorXdual complete_para_dual = complete_para;
            complete_para_dual(this->effective_para_index) = effective_para_dual;
            return this->model->gradient_autodiff(complete_para_dual, *this->data);
        };
        gradient = autodiff::gradient(func, wrt(effective_para_dual), at(effective_para_dual), value_dual);
        value = val(value_dual);
    }

    return value;
}

void UniversalData::gradient_and_hessian(const VectorXd &effective_para, VectorXd &gradient, MatrixXd &hessian)
{
    gradient.resize(this->effective_size);
    hessian.resize(this->effective_size, this->effective_size);
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    VectorXd complete_grad = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;

    if (model->hessian_user_defined)
    {
        double value = 0.0;
        tie(value, complete_grad) = model->gradient_user_defined(complete_para, *this->data);
        gradient = complete_grad(this->effective_para_index);
        hessian = model->hessian_user_defined(complete_para, *this->data)(this->effective_para_index, this->effective_para_index);
    }
    else
    { // forward autodiff
        dual2nd value_dual;
        VectorXdual2nd gradient_dual;
        VectorXdual2nd effective_para_dual = effective_para;
        auto func = [this, &complete_para](VectorXdual2nd const &effective_para_dual)
        {
            VectorXdual2nd complete_para_dual = complete_para;
            complete_para_dual(this->effective_para_index) = effective_para_dual;
            return this->model->hessian_autodiff(complete_para_dual, *this->data);
        };
        hessian = autodiff::hessian(func, wrt(effective_para_dual), at(effective_para_dual), value_dual, gradient_dual);
        for (Eigen::Index i = 0; i < this->effective_size; i++)
        {
            gradient[i] = val(gradient_dual[i]);
        }
    }
}

double UniversalData::optimize(VectorXd &effective_para)
{
    if (effective_para.size() == 0)
    {
        return model->loss(VectorXd::Zero(this->model_size), *this->data);
    }
    auto value_and_grad = [this](const VectorXd &complete_para, pybind11::object data) -> pair<double, VectorXd>
    {
        if (this->model->gradient_user_defined)
        {
            return this->model->gradient_user_defined(complete_para, data);
        }
        else
        { // forward autodiff
            dual value_dual;
            VectorXdual complete_para_dual = complete_para;
            VectorXd gradient = autodiff::gradient(this->model->gradient_autodiff, wrt(complete_para_dual), at(complete_para_dual, data), value_dual);
            double value = val(value_dual);
            return make_pair(value, gradient);
        }
    };
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;
    double loss;
    tie(loss, complete_para) = this->convex_solver(
        model->loss,
        value_and_grad,
        complete_para,
        this->effective_para_index,
        *this->data);
    effective_para = complete_para(this->effective_para_index);
    return loss;
}

void UniversalModel::set_loss_of_model(function<double(VectorXd const &, pybind11::object const &)> const &f)
{
    loss = f;
}

void UniversalModel::set_gradient_autodiff(function<dual(VectorXdual const &, pybind11::object const &)> const &f)
{
    gradient_autodiff = f;
    gradient_user_defined = nullptr;
}

void UniversalModel::set_hessian_autodiff(function<dual2nd(VectorXdual2nd const &, pybind11::object const &)> const &f)
{
    hessian_autodiff = f;
    hessian_user_defined = nullptr;
}

void UniversalModel::set_gradient_user_defined(function<pair<double, VectorXd>(VectorXd const &, pybind11::object const &)> const &f)
{
    gradient_user_defined = f;
    gradient_autodiff = nullptr;
}

void UniversalModel::set_hessian_user_defined(function<MatrixXd(VectorXd const &, pybind11::object const &)> const &f)
{
    hessian_user_defined = f;
    hessian_autodiff = nullptr;
}

void UniversalModel::set_slice_by_sample(function<pybind11::object(pybind11::object const &, VectorXi const &)> const &f)
{
    slice_by_sample = f;
}

void UniversalModel::set_deleter(function<void(pybind11::object const &)> const &f)
{
    if (f)
    {
        deleter = [f](pybind11::object const *p)
        { f(*p); delete p; };
    }
    else
    {
        deleter = [](pybind11::object const *p)
        { delete p; };
    }
}
