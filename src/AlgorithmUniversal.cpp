#include "AlgorithmUniversal.h"

#ifdef R_BUILD
// [[Rcpp::depends(nloptr)]]
#include <nloptrAPI.h>
#else
#include <nlopt.h> 
#endif

using namespace std;
using namespace Eigen;

double abessUniversal::loss_function(UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& active_para, VectorXd& aux_para, VectorXi& A,
    VectorXi& g_index, VectorXi& g_size, double lambda) 
{
    return active_data.loss(active_para, aux_para, lambda);
}

bool abessUniversal::primary_model_fit(UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& active_para, VectorXd& aux_para, double loss0,
    VectorXi& A, VectorXi& g_index, VectorXi& g_size) 
{
    SPDLOG_DEBUG("optimization begin\nactive set: {}\ninit loss: {}\naux_para:{}\npara:{}", active_data.get_effective_para_index().transpose(), loss0, aux_para.transpose(), active_para.transpose());    
    double value = 0.;
    active_data.init_para(active_para, aux_para);
    unsigned optim_size = active_para.size() + aux_para.size();
    VectorXd optim_para(optim_size);
    optim_para.head(aux_para.size()) = aux_para;
    optim_para.tail(active_para.size()) = active_para;
    nlopt_function f = active_data.get_nlopt_function(this->lambda_level);

    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, optim_size);
    nlopt_set_min_objective(opt, f, &active_data);
    nlopt_result result = nlopt_optimize(opt, optim_para.data(), &value); // positive return values means success
    nlopt_destroy(opt);

    bool success = result > 0;
    if(!success){
        SPDLOG_WARN("nlopt failed to optimize, state: {} ", nlopt_result_to_string(result));
    }
    aux_para = optim_para.head(aux_para.size());
    active_para = optim_para.tail(active_para.size());
    SPDLOG_DEBUG("optimization end\nfinal loss: {}\naux_para:{}\npara:{}", value, aux_para.transpose(), active_para.transpose());
    return success;
}

void abessUniversal::sacrifice(UniversalData& data, UniversalData& XA, MatrixXd& y, VectorXd& para, VectorXd& beta_A, VectorXd& aux_para, VectorXi& A, VectorXi& I, VectorXd& weights, VectorXi& g_index, VectorXi& g_size, int g_num, VectorXi& A_ind, VectorXd& sacrifice, VectorXi& U, VectorXi& U_ind, int num)
{
    SPDLOG_DEBUG("sacrifice begin");
    for (int i = 0; i < A.size(); i++) {
        VectorXd gradient_group(g_size(A[i]));
        MatrixXd hessian_group(g_size(A[i]), g_size(A[i]));
        data.hessian(para, aux_para, gradient_group, hessian_group, g_index(A[i]), g_size(A[i]), this->lambda_level);
        if (g_size(A[i]) == 1) { // optimize for frequent degradation situations
            sacrifice(A[i]) = para(g_index(A[i])) * para(g_index(A[i])) * hessian_group(0, 0);
        }
        else {
            sacrifice(A[i]) = para.segment(g_index(A[i]), g_size(A[i])).transpose() * hessian_group * para.segment(g_index(A[i]), g_size(A[i]));
            sacrifice(A[i]) /= g_size(A[i]);
        }
    }
    for (int i = 0; i < I.size(); i++) {
        VectorXd gradient_group(g_size(I[i]));
        MatrixXd hessian_group(g_size(I[i]), g_size(I[i]));
        data.hessian(para, aux_para, gradient_group, hessian_group, g_index(I[i]), g_size(I[i]), this->lambda_level);
        if (g_size(I[i]) == 1) { // Optimize for degradation situations, it often happens
            if (hessian_group(0, 0) < this->enough_small) {
                SPDLOG_ERROR("hessian is not positive definite:\n{}", hessian_group(0, 0));
                sacrifice(I[i]) = DBL_MAX; // TODO
            }
            else {
                sacrifice(I[i]) = gradient_group(0, 0) * gradient_group(0, 0) / hessian_group(0, 0);
            }
        }
        else {
            LLT<MatrixXd> hessian_group_llt(hessian_group);
            if (hessian_group_llt.info() == NumericalIssue){
                SPDLOG_ERROR("hessian is not positive definite:\n{}", hessian_group);
                sacrifice(I[i]) = DBL_MAX; // TODO
            }
            else{
                MatrixXd inv_hessian_group = hessian_group_llt.solve(MatrixXd::Identity(g_size(i), g_size(i)));
                sacrifice(I[i]) = gradient_group.transpose() * inv_hessian_group * gradient_group;
                sacrifice(I[i]) /= g_size(I[i]);
            }
        }
    }
    SPDLOG_DEBUG("sacrifice end");
    return;
}

double abessUniversal::effective_number_of_parameter(UniversalData& X, UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& beta, VectorXd& active_para, VectorXd& aux_para)
{
    if (this->lambda_level == 0.) return active_data.cols();

    if (active_data.cols() == 0) return 0.;

    MatrixXd hessian(active_data.cols(), active_data.cols());
    VectorXd g;
    active_data.hessian(active_para, aux_para, g, hessian, 0, active_data.cols(), this->lambda_level);
    SelfAdjointEigenSolver<MatrixXd> adjoint_eigen_solver(hessian, EigenvaluesOnly);
    double enp = 0.;
    for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
    }
    return enp;
}
