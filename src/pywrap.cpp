#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <tuple>
#include "UniversalData.h"
#include "List.h"
#include "api.h"
#include "utilities.h"
#include "predefined_model.hpp"

std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double, double>
pywrap_Universal(ExternData data, UniversalModel model, NloptConfig nlopt_solver, int model_size, int sample_size,int aux_para_size, int max_iter,
    int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef, int Kfold, Eigen::VectorXi sequence, 
    Eigen::VectorXd lambda_seq, int s_min, int s_max, int screening_size, Eigen::VectorXi g_index, Eigen::VectorXi always_select, 
    int thread, int splicing_type, int sub_search, Eigen::VectorXi cv_fold_id, Eigen::VectorXi A_init, Eigen::VectorXd beta_init, Eigen::VectorXd coef0_init)
{
    List mylist = abessUniversal_API(data, model, nlopt_solver, model_size, sample_size, aux_para_size, max_iter, exchange_num,
        path_type, is_warm_start, ic_type, ic_coef, Kfold, sequence, lambda_seq, s_min, s_max,
        screening_size, g_index, always_select, thread, splicing_type, sub_search, cv_fold_id, A_init, beta_init, coef0_init);
    Eigen::VectorXd beta;
    Eigen::VectorXd aux_para;
    double train_loss = 0;
    double test_loss = 0;
    double ic = 0;
    mylist.get_value_by_name("beta", beta);
    mylist.get_value_by_name("coef0", aux_para);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("test_loss", test_loss);
    mylist.get_value_by_name("ic", ic);
    return std::make_tuple(beta, aux_para, train_loss, test_loss, ic);
}

PYBIND11_MODULE(_scope, m) {
    m.def("pywrap_Universal", &pywrap_Universal);
    pybind11::class_<UniversalModel>(m, "UniversalModel").def(pybind11::init<>())
        .def("set_loss_of_model", &UniversalModel::set_loss_of_model)
        .def("set_gradient_autodiff", &UniversalModel::set_gradient_autodiff)
        .def("set_hessian_autodiff", &UniversalModel::set_hessian_autodiff)
        .def("set_gradient_user_defined", &UniversalModel::set_gradient_user_defined)
        .def("set_hessian_user_defined", &UniversalModel::set_hessian_user_defined)
        .def("set_slice_by_sample", &UniversalModel::set_slice_by_sample)
        .def("set_deleter", &UniversalModel::set_deleter)
        .def("set_init_params_of_sub_optim", &UniversalModel::set_init_params_of_sub_optim);
    m.def("init_spdlog", &init_spdlog);
    pybind11::class_<NloptConfig>(m, "NloptConfig").def(pybind11::init<int, const char *, double, double, double, double, double, unsigned, unsigned>());
    pybind11::class_<QuadraticData>(m, "QuadraticData")
        .def(pybind11::init<Eigen::MatrixXd, Eigen::VectorXd>());
    m.def("quadratic_loss", &quadratic_loss<double>);
    m.def("quadratic_loss", &quadratic_loss<dual>);
    m.def("quadratic_loss", &quadratic_loss<dual2nd>);
    m.def("quadratic_grad", &quadratic_grad);
    m.def("quadratic_hess", &quadratic_hess);
}
