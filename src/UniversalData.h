#ifndef SRC_UNIVERSALDATA_H
#define SRC_UNIVERSALDATA_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// struct ExternData {}; //TODO
#else
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
using ExternData = pybind11::object;
#include <memory>
#include <utility>
#endif

#include <functional>
#include <autodiff/forward/dual.hpp>

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
    // complete_para: the initial para
    // activate_para: this is a concept of abess algorithm, which is considered to have an impact on the model
    // inactivate_para: complementary of activate_para
    // effective_para: IMPORTANT! this is a concept of class UniversalData according to the concept activate_para of abess algorithm
    //                  and used to simulate the selection operation of extern data sets.
    //                  Its size(effective_size) is match for the length of effective_para_index
private:
    UniversalModel* model;
    Eigen::Index sample_size;
    double lambda = 0.;  // L2 penalty coef for nlopt
    Eigen::Index model_size; // length of complete_para
    VectorXi effective_para_index;// complete_para[effective_para_index[i]] = effective_para[i], ohter location of complete_para is 0
    Eigen::Index effective_size; // length of effective_para_index
    std::shared_ptr<ExternData> data; // statistic data from user
public:
    UniversalData() = default;
    UniversalData(Eigen::Index model_size, Eigen::Index sample_size, ExternData& data, UniversalModel* model);
    UniversalData slice_by_para(const VectorXi& target_para_index); // used in util func X_seg() and slice()

    Eigen::Index rows() const; // getter of sample_size
    Eigen::Index cols() const; // getter of effective_para
    const VectorXi& get_effective_para_index() const; // getter of effective_para_index
    UniversalData slice_by_sample(const VectorXi& target_sample_index);
    nlopt_function get_nlopt_function(double lambda); // create a function which can be optimized by nlopt
    double loss(const VectorXd& effective_para, const VectorXd& aux_para, double lambda); // compute the loss with effective_para
    double gradient(const VectorXd& effective_para, const VectorXd& aux_para, Eigen::Map<VectorXd>& gradient, double lambda); // compute the gradient of effective_para
    void hessian(const VectorXd& effective_para, const VectorXd& aux_para, VectorXd& gradient, MatrixXd& hessian, Eigen::Index index, Eigen::Index size, double lambda); // compute the hessian of sequence from index to (index+size-1) in effective_para                
    void init_para(VectorXd & active_para, VectorXd & aux_para);  // init para and aux_para, default is not change.                                                                                                          
};

class UniversalModel{
    friend class UniversalData;
private:
    // size of para will be match for data
    function <double(VectorXd const& para, VectorXd const& aux_para, ExternData const& data)> loss;
    function <dual(VectorXdual const& para, VectorXdual const& aux_para, ExternData const& data)> gradient_autodiff;
    function <dual2nd(VectorXdual2nd const& para, VectorXdual2nd const& aux_para, ExternData const& data)> hessian_autodiff;
    // only the derivative of aux_para and para[compute_para_index[i]] need be computed, 
    // size of return will equal to the sum of compute_para_index and aux_para. 
    // the derivative of aux_para should be setted before para.
    function <VectorXd(VectorXd const& para, VectorXd const& aux_para, ExternData const& data, VectorXi const& compute_para_index)> gradient_user_defined;
    // only the derivative of para[compute_para_index[i]] need be computed, size of gradient will equal to compute_para_index.
    // compute_para_index: compute_para[i] = para[compute_para_index[i]]
    function <MatrixXd(VectorXd const& para, VectorXd const& aux_para, ExternData const& data, VectorXi const& compute_para_index)> hessian_user_defined;
    function <ExternData(ExternData const& old_data, VectorXi const& target_sample_index)> slice_by_sample;
    function <void(ExternData const* p)> deleter = [](ExternData const* p) { delete p; };
    function <pair<VectorXd, VectorXd>(VectorXd & para, VectorXd & aux_para, ExternData const& data, VectorXi const& active_para_index)> init_para = nullptr;
    //TODO: constraints
public:
    // register callback function
    void set_loss_of_model(function <double(VectorXd const&, VectorXd const&, ExternData const&)> const&);
    void set_gradient_autodiff(function <dual(VectorXdual const&, VectorXdual const&, ExternData const&)> const&);
    void set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, VectorXdual2nd const&, ExternData const&)> const&);
    void set_gradient_user_defined(function <VectorXd(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const&);
    void set_hessian_user_defined(function <MatrixXd(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const&);
    void set_slice_by_sample(function <ExternData(ExternData const&, VectorXi const&)> const&);
    void set_deleter(function <void(ExternData const&)> const&);
    void set_init_para(function <pair<VectorXd, VectorXd>(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const&);
};
#endif //SRC_UNIVERSALDATA_H
