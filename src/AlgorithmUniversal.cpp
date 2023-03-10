#include "AlgorithmUniversal.h"
#include <nlopt.h> 

using namespace std;
using namespace Eigen;

double abessUniversal::loss_function(UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& active_para, VectorXd& aux_para, VectorXi& A,
    VectorXi& g_index, VectorXi& g_size, double lambda) 
{
    return active_data.loss(active_para);
}

double nlopt_function(unsigned n, const double* x, double* grad, void* f_data) {
    UniversalData* data = static_cast<UniversalData*>(f_data);
    Map<VectorXd const> effective_para(x, n);
    if (grad) { // not use operator new
        Map<VectorXd> gradient(grad, n);
        return data->loss_and_gradient(effective_para, gradient);
    }
    else {
        return data->loss(effective_para);
    }
};

bool abessUniversal::primary_model_fit(UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& active_para, VectorXd& aux_para, double loss0,
    VectorXi& A, VectorXi& g_index, VectorXi& g_size) 
{
    SPDLOG_DEBUG("optimization begin\nactive set: {}\ninit loss: {}\npara:{}", active_data.get_effective_para_index().transpose(), loss0, active_para.transpose());    
    double value = 0.;
    active_data.init_para(active_para);

    nlopt_opt opt = active_data.nlopt_create(active_para.size());
    nlopt_set_min_objective(opt, nlopt_function, &active_data);
    nlopt_result result = nlopt_optimize(opt, active_para.data(), &value); 
    nlopt_destroy(opt);

    bool success = result > 0;
    if(!success){
        SPDLOG_WARN("nlopt failed to optimize, state: {} ", nlopt_result_to_string(result));
    }

    SPDLOG_DEBUG("optimization end\nfinal loss: {}\npara:{}", value, active_para.transpose());
    return success;
}

void abessUniversal::sacrifice(UniversalData& data, UniversalData& XA, MatrixXd& y, VectorXd& para, VectorXd& beta_A, VectorXd& aux_para, VectorXi& A, VectorXi& I, VectorXd& weights, VectorXi& g_index, VectorXi& g_size, int g_num, VectorXi& A_ind, VectorXd& sacrifice, VectorXi& U, VectorXi& U_ind, int num)
{
    SPDLOG_DEBUG("sacrifice begin");
    VectorXd gradient_full;
    MatrixXd hessian_full;
    data.gradient_and_hessian(para, gradient_full, hessian_full);

    int size, index;
    for (auto group_index : A) {
        size = g_size(group_index);
        index = g_index(group_index);
        VectorXd gradient_group = gradient_full.segment(index, size);
        MatrixXd hessian_group = hessian_full.block(index, index, size, size);
        if (size == 1) { // optimize for frequent degradation situations
            sacrifice(group_index) = para(index) * para(index) * hessian_group(0, 0);
        }
        else {
            sacrifice(group_index) = para.segment(index, size).transpose() * hessian_group * para.segment(index, size);
            sacrifice(group_index) /= size;
        }
    }
    for (auto group_index : I) {
        size = g_size(group_index);
        index = g_index(group_index);
        VectorXd gradient_group = gradient_full.segment(index, size);
        MatrixXd hessian_group = hessian_full.block(index, index, size, size);
        if (size == 1) { // optimize for frequent degradation situations
            if (hessian_group(0, 0) < this->enough_small) {
                SPDLOG_ERROR("there exists a submatrix of hessian which is not positive definite!\nactive set is{}\nactive params are {}\ngroup index is {}, hessian is {}", data.get_effective_para_index().transpose(),para.transpose(), index, hessian_group(0,0));
                sacrifice(group_index) = gradient_group(0, 0) * gradient_group(0, 0);
            }
            else {
                sacrifice(group_index) = gradient_group(0, 0) * gradient_group(0, 0) / hessian_group(0, 0);
            }
        }
        else {
            LLT<MatrixXd> hessian_group_llt(hessian_group);
            if (hessian_group_llt.info() == NumericalIssue){
                SPDLOG_ERROR("there exists a submatrix of hessian which is not positive definite!\nactive set is {}\nactive params are {}\ngroup index is {}\nhessian is {}", data.get_effective_para_index().transpose(), para.transpose(), VectorXi::LinSpaced(size, index, size + index - 1), hessian_group);
                sacrifice(group_index) = gradient_group.squaredNorm();
            }
            else{
                MatrixXd inv_hessian_group = hessian_group_llt.solve(MatrixXd::Identity(size, size));
                sacrifice(group_index) = gradient_group.transpose() * inv_hessian_group * gradient_group;
                sacrifice(group_index) /= size;
            }
        }
    }
    SPDLOG_DEBUG("sacrifice end with {}", sacrifice.transpose());
    return;
}

double abessUniversal::effective_number_of_parameter(UniversalData& X, UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& beta, VectorXd& active_para, VectorXd& aux_para)
{
    return active_data.cols();
}
