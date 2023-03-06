# pragma once

#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
using ExternData = pybind11::object;
#include <memory>
#include <utility>


#include <functional>
#include <autodiff/forward/dual.hpp>

#include "NloptConfig.h"

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using autodiff::dual;
using autodiff::dual2nd;
using VectorXdual = Eigen::Matrix<dual,-1,1>;
using VectorXdual2nd = Eigen::Matrix<dual2nd,-1,1>;
using std::function;
using std::pair;

using nlopt_function = double (*)(unsigned n, const double* x, double* grad, void* f_data);

class UniversalModel;
// UniversalData includes everything about the statistic model like loss, constraints and the statistic data like samples, operations of data.
// In abess project, UniversalData will be an instantiation of T4 in template class algorithm, other instantiation of T4 often is matrix.
// Thus, UniversalData is just like a matrix in algorithm, and its instance is often denoted as 'x'.
// In order to work like matrix, UniversalData need the help of utility function like X_seg, slice.
class UniversalData {
    // complete_para: the initial para, often `para` for short
    // activate_para: this is a concept of abess algorithm, which is considered to have an impact on the model
    // effective_para: IMPORTANT! this is a concept of class UniversalData, which is used to simulate the selection operation of extern data sets.
    //                 non-effective_para is like being deleted in the extern data sets and can't be used by any function, 
    //                 thus active_para is a subset of effective_para, and effective_para is a subset of complete_para.
    //                 out of Class UniversalData, effective_para is invisible. 
private:
    UniversalModel* model;
    NloptConfig* nlopt_solver;
    Eigen::Index sample_size;
    double lambda = 0.;  // L2 penalty coef for nlopt
    Eigen::Index model_size; // length of complete_para
    VectorXi effective_para_index;// `complete_para[effective_para_index[i]]` is `effective_para[i]`
    Eigen::Index effective_size; // length of effective_para_index
    std::shared_ptr<ExternData> data; // statistic data from user 
public:
    UniversalData() = default;
    UniversalData(Eigen::Index model_size, Eigen::Index sample_size, ExternData& data, UniversalModel* model, NloptConfig* nlopt_solver);
    UniversalData slice_by_para(const VectorXi& target_para_index); // used in util func X_seg() and slice()

    Eigen::Index rows() const; // getter of sample_size
    Eigen::Index cols() const; // getter of effective_para
    const VectorXi& get_effective_para_index() const; // getter of effective_para_index
    UniversalData slice_by_sample(const VectorXi& target_sample_index);
    nlopt_function get_nlopt_function(double lambda); // create a function which can be optimized by nlopt
    double loss(const VectorXd& effective_para, double lambda); // compute the loss with effective_para
    double loss_and_gradient(const VectorXd& effective_para, Eigen::Map<VectorXd>& gradient, double lambda);
    void gradient_and_hessian(const VectorXd& effective_para, VectorXd& gradient,MatrixXd& hessian, double lambda);             
    void init_para(VectorXd & effective_para);  // initialize para for primary_model_fit, default is not change.                                                                                        
    nlopt_opt nlopt_create(unsigned dim) {return this->nlopt_solver->create(dim);}                
};

class UniversalModel{
    friend class UniversalData;
private:
    // size of para will be match for data
    function <double(VectorXd const& para, ExternData const& data)> loss;
    function <dual(VectorXdual const& para, ExternData const& data)> gradient_autodiff;
    function <dual2nd(VectorXdual2nd const& para, ExternData const& data)> hessian_autodiff;
    function <pair<double, VectorXd>(VectorXd const& para, ExternData const& data)> gradient_user_defined;
    function <MatrixXd(VectorXd const& para, ExternData const& data)> hessian_user_defined;
    function <ExternData(ExternData const& old_data, VectorXi const& target_sample_index)> slice_by_sample;
    function <void(ExternData const* p)> deleter = [](ExternData const* p) { delete p; };
    function <VectorXd(VectorXd & para, ExternData const& data, VectorXi const& active_para_index)> init_para = nullptr;

public:
    // register callback function
    void set_loss_of_model(function <double(VectorXd const&, ExternData const&)> const&);
    void set_gradient_autodiff(function <dual(VectorXdual const&, ExternData const&)> const&);
    void set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, ExternData const&)> const&);
    void set_gradient_user_defined(function <pair<double, VectorXd>(VectorXd const&, ExternData const&)> const&);
    void set_hessian_user_defined(function <MatrixXd(VectorXd const&, ExternData const&)> const&);
    void set_slice_by_sample(function <ExternData(ExternData const&, VectorXi const&)> const&);
    void set_deleter(function <void(ExternData const&)> const&);
    void set_init_params_of_sub_optim(function <VectorXd(VectorXd const&, ExternData const&, VectorXi const&)> const&);
};
