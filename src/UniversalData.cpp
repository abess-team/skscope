#include "UniversalData.h"
#include "utilities.h"
#include <autodiff/forward/dual/eigen.hpp>
#include <iostream>
using namespace std;
using Eigen::Map;
using Eigen::Matrix;

UniversalData::UniversalData(Eigen::Index model_size, Eigen::Index sample_size, ExternData& data, UniversalModel* model, NloptConfig* nlopt_solver)
    : model(model), nlopt_solver(nlopt_solver), sample_size(sample_size), model_size(model_size), effective_size(model_size)
{
    this->effective_para_index = VectorXi::LinSpaced(model_size, 0, model_size - 1);
    this->data = shared_ptr<ExternData>(new ExternData(data));
}

UniversalData UniversalData::slice_by_para(const VectorXi& target_para_index)
{
    UniversalData tem(*this);
    tem.effective_para_index = this->effective_para_index(target_para_index);
    tem.effective_size = target_para_index.size();
    return tem;
}

UniversalData UniversalData::slice_by_sample(const VectorXi& target_sample_index)
{
    UniversalData tem(*this);
    tem.sample_size = target_sample_index.size();
    tem.data = shared_ptr<ExternData>(new ExternData(model->slice_by_sample(*data, target_sample_index)), model->deleter);
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

const VectorXi& UniversalData::get_effective_para_index() const
{
    return effective_para_index;
}

nlopt_function UniversalData::get_nlopt_function(double lambda) 
{
    this->lambda = lambda;
    return [](unsigned n, const double* x, double* grad, void* f_data) {
        UniversalData* data = static_cast<UniversalData*>(f_data);
        Map<VectorXd const> aux_para(x, n - data->effective_size);
        Map<VectorXd const> effective_para(x + n - data->effective_size, data->effective_size);
        if (grad) { // not use operator new
            Map<VectorXd> gradient(grad, n);
            return data->loss_and_gradient(effective_para, aux_para, gradient, data->lambda);
        }
        else {
            return data->loss(effective_para, aux_para, data->lambda);
        }
    };
}

double UniversalData::loss(const VectorXd& effective_para, const VectorXd& aux_para, double lambda)
{
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;
    return model->loss(complete_para, aux_para, *this->data) + lambda * effective_para.squaredNorm();
    
}

double UniversalData::loss_and_gradient(const VectorXd& effective_para, const VectorXd& aux_para, Map<VectorXd>& gradient, double lambda)
{
    double value = 0.0;
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;

    if (model->gradient_user_defined) {
        VectorXd full_grad(this->model_size + aux_para.size());
        tie(value, full_grad) = model->gradient_user_defined(complete_para, aux_para, *this->data);
        gradient.head(aux_para.size()) = full_grad.head(aux_para.size());
        gradient.tail(effective_size) = full_grad.tail(this->model_size)(this->effective_para_index);
        //gradient = model->gradient_user_defined(complete_para, aux_para, *this->data, this->effective_para_index);
        //value = model->loss(complete_para, aux_para, *this->data);
    }
    else { // autodiff
        dual v;
        VectorXdual effective_para_dual = effective_para;
        VectorXdual aux_para_dual = aux_para;
        auto func = [this, &complete_para](VectorXdual const& compute_para, VectorXdual const& aux_para) {
            VectorXdual para = complete_para;
            para(this->effective_para_index) = compute_para;
            return this->model->gradient_autodiff(para, aux_para, *this->data);
        };
        gradient.head(aux_para.size()) = autodiff::gradient(func, wrt(aux_para_dual), at(effective_para_dual, aux_para_dual), v);
        gradient.tail(effective_size) = autodiff::gradient(func, wrt(effective_para_dual), at(effective_para_dual, aux_para_dual), v);
        value = val(v);
    }
    
    gradient.tail(effective_size) += 2 * lambda * effective_para;
    return value + lambda * effective_para.squaredNorm();
}

void UniversalData::gradient_and_hessian(const VectorXd& effective_para, const VectorXd& aux_para, VectorXd& gradient,MatrixXd& hessian, double lambda)
{
    gradient.resize(this->effective_size);
    hessian.resize(this->effective_size, this->effective_size);
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;
    
    if (model->hessian_user_defined) {
        double value = 0.0;
        VectorXd full_grad(this->model_size + aux_para.size());
        tie(value, full_grad) = model->gradient_user_defined(complete_para, aux_para, *this->data);
        gradient = full_grad.tail(this->model_size)(this->effective_para_index);
        hessian = model->hessian_user_defined(complete_para, aux_para, *this->data)(this->effective_para_index, this->effective_para_index);
    }
    else { // autodiff
        dual2nd v;
        VectorXdual2nd g;
        VectorXdual2nd compute_para_dual = effective_para;
        VectorXdual2nd aux_para_dual = aux_para;
        hessian = autodiff::hessian([this, &complete_para](VectorXdual2nd const& compute_para_dual, VectorXdual2nd const& aux_para_dual) {
            VectorXdual2nd para = complete_para;
            para(this->effective_para_index) = compute_para_dual;
            return this->model->hessian_autodiff(para, aux_para_dual, *this->data);
            }, wrt(compute_para_dual), at(compute_para_dual, aux_para_dual), v, g);
        for (Eigen::Index i = 0; i < this->effective_size; i++) {
            gradient[i] = val(g[i]);
        }
    }
    if (lambda != 0.0) {
        gradient += 2 * lambda * effective_para;
        hessian += 2 * lambda * MatrixXd::Identity(this->effective_size, this->effective_size);
    }
}

void UniversalData::init_para(VectorXd & active_para, VectorXd & aux_para){
    if (model->init_para) {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
        complete_para(this->effective_para_index) = active_para;
        tie(complete_para, aux_para) = model->init_para(complete_para, aux_para, *this->data, this->effective_para_index);
        active_para = complete_para(this->effective_para_index);
    }
}


void UniversalModel::set_loss_of_model(function <double(VectorXd const&, VectorXd const&, ExternData const&)> const& f)
{
    loss = f;
}

void UniversalModel::set_gradient_autodiff(function <dual(VectorXdual const&, VectorXdual const&, ExternData const&)> const& f) {
    gradient_autodiff = f;
    gradient_user_defined = nullptr;
}

void UniversalModel::set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, VectorXdual2nd const&, ExternData const&)> const& f) {
    hessian_autodiff = f;
    hessian_user_defined = nullptr;
}

void UniversalModel::set_gradient_user_defined(function <pair<double, VectorXd>(VectorXd const&, VectorXd const&, ExternData const&)> const& f)
{
    gradient_user_defined = f;
    gradient_autodiff = nullptr;
}

void UniversalModel::set_hessian_user_defined(function <MatrixXd(VectorXd const&, VectorXd const&, ExternData const&)> const& f)
{
    hessian_user_defined = f;
    hessian_autodiff = nullptr;
}

void UniversalModel::set_slice_by_sample(function <ExternData(ExternData const&, VectorXi const&)> const& f)
{
    slice_by_sample = f;
}

void UniversalModel::set_deleter(function<void(ExternData const&)> const& f)
{
    if (f) {
        deleter = [f](ExternData const* p) { f(*p); delete p; };
    }
    else {
        deleter = [](ExternData const* p) { delete p; };
    }
}

void UniversalModel::set_init_params_of_sub_optim(function <pair<VectorXd, VectorXd>(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const& f)
{
    init_para = f;
}
