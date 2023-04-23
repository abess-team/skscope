# pragma once

#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <memory>
#include <utility>


#include <functional>
#include <autodiff/forward/dual.hpp>

//#include "NloptConfig.h"

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using autodiff::dual;
using autodiff::dual2nd;
using VectorXdual = Eigen::Matrix<dual,-1,1>;
using VectorXdual2nd = Eigen::Matrix<dual2nd,-1,1>;
using std::function;
using std::pair;

using ConvexSolver = function<pair<double, VectorXd>(
    function <double(VectorXd const&, pybind11::object const&)>, // loss_fn
    function <pair<double, VectorXd>(const VectorXd&, pybind11::object)>, // value_and_grad
    const VectorXd&, // complete_para
    const VectorXi&, // effective_para_index
    pybind11::object const& // data
)>;

class UniversalModel;
// UniversalData includes everything about the statistic model like loss, constraints and the statistic data like samples, operations of data.
// In abess project, UniversalData will be an instantiation of UniversalData in template class algorithm, other instantiation of UniversalData often is matrix.
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
    ConvexSolver convex_solver;
    Eigen::Index sample_size;
    Eigen::Index model_size; // length of complete_para
    VectorXi effective_para_index;// `complete_para[effective_para_index[i]]` is `effective_para[i]`
    Eigen::Index effective_size; // length of effective_para_index
    std::shared_ptr<pybind11::object> data; // statistic data from user 
public:
    UniversalData() = default;
    UniversalData(Eigen::Index model_size, Eigen::Index sample_size, pybind11::object& data, UniversalModel* model, ConvexSolver convex_solver);
    UniversalData slice_by_para(const VectorXi& target_para_index); // used in util func X_seg() and slice()

    Eigen::Index rows() const; // getter of sample_size
    Eigen::Index cols() const; // getter of effective_para
    const VectorXi& get_effective_para_index() const; // getter of effective_para_index, only used for log
    UniversalData slice_by_sample(const VectorXi& target_sample_index);
    double loss(const VectorXd& effective_para); // compute the loss with effective_para
    double loss_and_gradient(const VectorXd& effective_para, Eigen::Map<VectorXd>& gradient);
    void gradient_and_hessian(const VectorXd& effective_para, VectorXd& gradient,MatrixXd& hessian);             
    void init_para(VectorXd & effective_para);  // initialize para for primary_model_fit, default is not change.                                                                                        
    double optimize(VectorXd& effective_para);                
};

class UniversalModel{
    friend class UniversalData;
private:
    // size of para will be match for data
    function <double(VectorXd const& para, pybind11::object const& data)> loss;
    function <dual(VectorXdual const& para, pybind11::object const& data)> gradient_autodiff;
    function <dual2nd(VectorXdual2nd const& para, pybind11::object const& data)> hessian_autodiff;
    function <pair<double, VectorXd>(VectorXd const& para, pybind11::object const& data)> gradient_user_defined;
    function <MatrixXd(VectorXd const& para, pybind11::object const& data)> hessian_user_defined;
    function <pybind11::object(pybind11::object const& old_data, VectorXi const& target_sample_index)> slice_by_sample;
    function <void(pybind11::object const* p)> deleter = [](pybind11::object const* p) { delete p; };
    function <VectorXd(VectorXd & para, pybind11::object const& data, VectorXi const& active_para_index)> init_para = nullptr;

public:
    // register callback function
    void set_loss_of_model(function <double(VectorXd const&, pybind11::object const&)> const&);
    void set_gradient_autodiff(function <dual(VectorXdual const&, pybind11::object const&)> const&);
    void set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, pybind11::object const&)> const&);
    void set_gradient_user_defined(function <pair<double, VectorXd>(VectorXd const&, pybind11::object const&)> const&);
    void set_hessian_user_defined(function <MatrixXd(VectorXd const&, pybind11::object const&)> const&);
    void set_slice_by_sample(function <pybind11::object(pybind11::object const&, VectorXi const&)> const&);
    void set_deleter(function <void(pybind11::object const&)> const&);
    void set_init_params_of_sub_optim(function <VectorXd(VectorXd const&, pybind11::object const&, VectorXi const&)> const&);
};
