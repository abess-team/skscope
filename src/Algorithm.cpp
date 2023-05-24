#include "Algorithm.h"
#include <nlopt.h>

using namespace std;
using namespace Eigen;

void Algorithm::fit(UniversalData &train_x, MatrixXd &train_y, VectorXd &train_weight, VectorXi &g_index, VectorXi &g_size,
                    int train_n, int p, int N)
{
    int T0 = this->sparsity_level;
    this->x = &train_x;
    this->y = &train_y;

    // Warm-start:
    //     If warm-start is disabled, they would be just zero.
    this->beta = this->beta_init;
    this->coef0 = this->coef0_init;
    this->bd = this->bd_init;

    // Initialize sub-search:
    //     To speed up, we focus on a subset of all groups, named `U`,
    //     whose size is equal or smaller than N.
    //     (More details can be found in function `get_A` below)
    if (this->sub_search == 0 || this->sparsity_level + this->sub_search > N)
        this->U_size = N;
    else
        this->U_size = this->sparsity_level + this->sub_search;

    // No need to splicing?
    //     If N == T0, we must put all groups into the model.
    if (N == T0)
    {
        this->A_out = VectorXi::LinSpaced(N, 0, N - 1);
        // VectorXd beta_old = this->beta;
        // VectorXd coef0_old = this->coef0;
        bool success = this->primary_model_fit(train_x, train_y, train_weight, this->beta, this->coef0, DBL_MAX,
                                               this->A_out, g_index, g_size);
        // if (!success){
        //   this->beta = beta_old;
        //   this->coef0 = coef0_old;
        // }
        this->train_loss = this->loss_function(train_x, train_y, train_weight, this->beta, this->coef0, this->A_out,
                                               g_index, g_size, this->lambda_level);
        this->effective_number = this->effective_number_of_parameter(train_x, train_x, train_y, train_weight,
                                                                     this->beta, this->beta, this->coef0);
        return;
    }

    // Initial active/inactive set:
    //     Defaultly, choose `T0` groups with largest `bd_init` as initial active set.
    //     If there is no `bd_init` (may be no warm-start), compute it on `beta_init`, `coef0_init`, `A_init`.
    //     However, you can also define your own criterion by rewrite the function.
    VectorXi A = this->inital_screening(train_x, train_y, this->beta, this->coef0, this->A_init,
                                        this->I_init, this->bd, train_weight, g_index, g_size, N);
    VectorXi I = complement(A, N);

    // `A_ind` stores all indexes of active set.
    // For example, if "Group 1" is active and there are three variables inside,
    // `A` will only contain "Group 1" while `A_ind` store all three variables' indexes.
    VectorXi A_ind = find_ind(A, g_index, g_size, (this->beta).rows(), N);
    UniversalData X_A = X_seg(train_x, train_n, A_ind, this->model_type);
    VectorXd beta_A;
    slice(this->beta, A_ind, beta_A);
    // if (this->algorithm_type == 6)
    // {

    // VectorXd coef0_old = this->coef0;
    // Fitting on initial active set
    bool success =
        this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX, A, g_index, g_size);
    // if (!success){
    //   this->coef0 = coef0_old;
    // }else{
    slice_restore(beta_A, A_ind, this->beta);
    this->train_loss = this->loss_function(X_A, train_y, train_weight, beta_A, this->coef0, A, g_index, g_size,
                                           this->lambda_level);
    // }

    this->beta_warmstart = this->beta;
    this->coef0_warmstart = this->coef0;

    // Start splicing:
    //     `C_max` is the maximum exchange number in splicing.
    //     `get_A()` is to find and return the final chosen active set on spasity `T0`.
    int always_select_size = this->always_select.size();
    int C_max = min(min(T0 - always_select_size, this->U_size - T0 - always_select_size), this->exchange_num);

    this->update_tau(train_n, N);
    this->get_A(train_x, train_y, A, I, C_max, this->beta, this->coef0, this->bd, T0, train_weight, g_index, g_size,
                N, this->tau, this->train_loss);

    // Final fitting on `A`:
    //     For higher accuracy, fit again on chosen active set
    //     with stricter settings.
    this->final_fitting(train_x, train_y, train_weight, A, g_index, g_size, train_n, N);

    // Result & Output
    this->A_out = A;
    this->effective_number =
        this->effective_number_of_parameter(train_x, X_A, train_y, train_weight, this->beta, beta_A, this->coef0);
    this->group_df = A_ind.size();

    return;
};

void Algorithm::get_A(UniversalData &X, MatrixXd &y, VectorXi &A, VectorXi &I, int &C_max, VectorXd &beta, VectorXd &coef0,
                      VectorXd &bd, int T0, VectorXd &weights, VectorXi &g_index, VectorXi &g_size,
                      int N, double tau, double &train_loss)
{
    // Universal set:
    //     We only consider splicing on a set `U`,
    //     which may not contain all groups, but we hope all "useful" groups are included.
    //     We need to extract these groups out, e.g. `X`->`X_U`, `A`->`A_U`,
    //     and they have a new index from 0 to `U_size`-1.
    VectorXi U(this->U_size);
    VectorXi U_ind;
    VectorXi g_index_U(this->U_size);
    VectorXi g_size_U(this->U_size);
    UniversalData *X_U = new UniversalData;
    VectorXd beta_U;
    VectorXi A_U(T0);
    VectorXi I_U(this->U_size - T0);
    VectorXi always_select_U(this->always_select.size());

    if (this->U_size == N)
    {
        // If `U_size` == `N`, focus on all groups.
        U = VectorXi::LinSpaced(N, 0, N - 1);
    }
    else
    {
        // If `U_size` < `N`, focus on `U_size` groups with larger sacrifices.
        U = max_k(bd, this->U_size, true);
    }

    // int p = X.cols();
    int n = X.rows();
    int C = C_max;

    // The outer iteration:
    //     1. extract data from U
    //     2. splicing & fitting on U (inner iteration), update active set
    //     3. update U
    //     4. if U changed, exit
    int iter = 0;
    while (iter++ < this->max_iter)
    {
        //("important search iteration");
        // mapping
        if (this->U_size == N)
        {
            // If consider all groups, it is no need to map or give a new index.
            delete X_U;
            X_U = &X;
            U_ind = VectorXi::LinSpaced((this->beta).rows(), 0, (this->beta).rows() - 1);
            beta_U = beta;
            g_size_U = g_size;
            g_index_U = g_index;
            A_U = A;
            I_U = I;
            always_select_U = this->always_select;
        }
        else
        {
            // Extract `X`, `beta`, `g_index`, `g_size`, `always_select` on U,
            // give them new index (from 0 to U_size-1),
            // and name as `X_U`, `beta_U`, `g_index_U`, `g_size_U`, `always_select_U` respectively.
            U_ind = find_ind(U, g_index, g_size, (this->beta).rows(), N);
            *X_U = X_seg(X, n, U_ind, this->model_type);
            slice(beta, U_ind, beta_U);

            int pos = 0;
            for (int i = 0; i < U.size(); i++)
            {
                g_size_U(i) = g_size(U(i));
                g_index_U(i) = pos;
                pos += g_size_U(i);
            }

            // Since we have ranked U from large to small with sacrifice,
            // the first `T0` group should be initial active sets.
            A_U = VectorXi::LinSpaced(T0, 0, T0 - 1);
            I_U = VectorXi::LinSpaced(this->U_size - T0, T0, this->U_size - 1);

            int *temp = new int[N], s = this->always_select.size();
            memset(temp, 0, N);
            for (int i = 0; i < s; i++)
                temp[this->always_select(i)] = 1;
            for (int i = 0; i < this->U_size; i++)
            {
                if (s <= 0)
                    break;
                if (temp[U(i)] == 1)
                {
                    always_select_U(this->always_select.size() - s) = i;
                    s--;
                }
            }
            delete[] temp;
        }

        // The inner iteration:
        //     1. splicing on U
        //     2. update A_U
        int num = -1;
        while (true)
        {
            num++;
            // SPDLOG_DEBUG("splicing iteration");
            VectorXi A_ind = find_ind(A_U, g_index_U, g_size_U, U_ind.size(), this->U_size);
            UniversalData X_A = X_seg(*X_U, n, A_ind, this->model_type);
            VectorXd beta_A;
            slice(beta_U, A_ind, beta_A);

            VectorXd bd_U = VectorXd::Zero(this->U_size);
            this->sacrifice(*X_U, X_A, y, beta_U, beta_A, coef0, A_U, I_U, weights, g_index_U, g_size_U,
                            this->U_size, A_ind, bd_U, U, U_ind, num);

            for (int i = 0; i < always_select_U.size(); i++)
            {
                bd_U(always_select_U(i)) = DBL_MAX;
            }

            // Splicing:
            //     Try to exchange items in active and inactive set,
            //     If new loss is smaller, accept it and return TRUE.
            double l0 = train_loss;
            bool exchange = this->splicing(*X_U, y, A_U, I_U, C_max, beta_U, coef0, bd_U, weights, g_index_U,
                                           g_size_U, this->U_size, tau, l0);

            if (exchange)
                train_loss = l0;
            else
                // A_U is unchanged, so break.
                break;
        }

        // If A_U not change, U will not change and we can stop.
        if (A_U.size() == 0 || A_U.maxCoeff() == T0 - 1)
            break;

        // Update & Restore beta, A from U
        slice_restore(beta_U, U_ind, beta);

        VectorXi ind = VectorXi::Zero(N);
        for (int i = 0; i < T0; i++)
            ind(U(A_U(i))) = 1;

        int tempA = 0, tempI = 0;
        for (int i = 0; i < N; i++)
            if (ind(i) == 0)
                I(tempI++) = i;
            else
                A(tempA++) = i;

        // Compute sacrifices in full set
        VectorXi A_ind0 = find_ind(A, g_index, g_size, (this->beta).rows(), N);
        UniversalData X_A0 = X_seg(X, n, A_ind0, this->model_type);
        VectorXd beta_A0;
        slice(beta, A_ind0, beta_A0);
        VectorXi U0 = VectorXi::LinSpaced(N, 0, N - 1); // U0 contains all groups
        VectorXi U_ind0 = VectorXi::LinSpaced((this->beta).rows(), 0, (this->beta).rows() - 1);
        this->sacrifice(X, X_A0, y, beta, beta_A0, coef0, A, I, weights, g_index, g_size, N, A_ind0, bd, U0, U_ind0,
                        0);

        if (this->U_size == N)
        {
            // If U is the full set, there is no need to update, so stop.
            for (int i = 0; i < this->always_select.size(); i++)
                bd(this->always_select(i)) = DBL_MAX;

            break;
        }
        else
        {
            // If U is changed in the new situation, update it and iter again.
            for (int i = 0; i < T0; i++)
                bd(A(i)) = DBL_MAX;
            VectorXi U_new = max_k(bd, this->U_size, true);

            U = U_new;
            C_max = C;
        }
    }

    if (this->U_size != N)
        delete X_U;

    return;
};

bool Algorithm::splicing(UniversalData &X, MatrixXd &y, VectorXi &A, VectorXi &I, int &C_max, VectorXd &beta, VectorXd &coef0,
                         VectorXd &bd, VectorXd &weights, VectorXi &g_index, VectorXi &g_size,
                         int N, double tau, double &train_loss)
{
    if (C_max <= 0)
        return false;

    VectorXd beta_A_group = bd(A);
    VectorXd d_I_group = bd(I);

    VectorXi A_min_k = min_k(beta_A_group, C_max, true);
    VectorXi I_max_k = max_k(d_I_group, C_max, true);
    VectorXi s1 = A(A_min_k);
    VectorXi s2 = I(I_max_k);

    VectorXi A_exchange, best_A_exchange;
    VectorXi A_ind_exchage, best_A_ind_exchage;
    UniversalData X_A_exchage;
    VectorXd beta_A_exchange, best_beta_A_exchange;
    VectorXd coef0_A_exchange, best_coef0_A_exchange;

    double L, best_loss = train_loss;
    int best_exchange_num = 0;

    for (int k = C_max; k >= 1;)
    {
        SPDLOG_INFO("exchange num is {}", k);
        A_exchange = diff_union(A, s1, s2);
        A_ind_exchage = find_ind(A_exchange, g_index, g_size, (this->beta).rows(), N);
        X_A_exchage = X.slice_by_para(A_ind_exchage); 
        beta_A_exchange = beta(A_ind_exchage);
        coef0_A_exchange = coef0;
        bool success = this->primary_model_fit(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange,
                                               train_loss, A_exchange, g_index, g_size);
        L = this->loss_function(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, A_exchange, g_index,
                                g_size, this->lambda_level);
        if (L < best_loss)
        {
            best_A_exchange = A_exchange;
            best_A_ind_exchage = A_ind_exchage;
            best_beta_A_exchange = beta_A_exchange;
            best_coef0_A_exchange = coef0_A_exchange;
            best_loss = L;
            best_exchange_num = k;
            if (this->is_greedy)
                break;
        }
        k = this->splicing_type == 1 ? k - 1 : k / 2;
        s1 = s1.head(k).eval();
        s2 = s2.head(k).eval();
    }

    if (train_loss - best_loss <= tau)
        return false;

    train_loss = best_loss;
    A = best_A_exchange;
    I = complement(best_A_exchange, N);
    slice_restore(best_beta_A_exchange, best_A_ind_exchage, beta);
    coef0 = best_coef0_A_exchange;
    C_max = best_exchange_num;
    SPDLOG_INFO("best exchange num is {}", best_exchange_num);
    return true;
};

VectorXi Algorithm::inital_screening(UniversalData &X, MatrixXd &y, VectorXd &beta, VectorXd &coef0, VectorXi &A, VectorXi &I,
                                     VectorXd &bd, VectorXd &weights, VectorXi &g_index,
                                     VectorXi &g_size, int &N)
{
    if (bd.size() == 0)
    {
        // variable initialization
        int beta_size = X.cols();
        bd = VectorXd::Zero(N);

        // calculate beta & d & h
        VectorXi A_ind = find_ind(A, g_index, g_size, beta_size, N);
        UniversalData X_A = X_seg(X, X.rows(), A_ind, this->model_type);
        VectorXd beta_A;
        slice(beta, A_ind, beta_A);

        VectorXi U = VectorXi::LinSpaced(N, 0, N - 1);
        VectorXi U_ind = VectorXi::LinSpaced(beta_size, 0, beta_size - 1);
        this->sacrifice(X, X_A, y, beta, beta_A, coef0, A, I, weights, g_index, g_size, N, A_ind, bd, U, U_ind, 0);
        // A_init
        for (int i = 0; i < A.size(); i++)
        {
            bd(A(i)) = DBL_MAX / 2;
        }
        // alway_select
        for (int i = 0; i < this->always_select.size(); i++)
        {
            bd(this->always_select(i)) = DBL_MAX;
        }
    }

    // get Active-set A according to max_k bd
    VectorXi A_new = max_k(bd, this->sparsity_level);

    return A_new;
}

void Algorithm::final_fitting(UniversalData &train_x, MatrixXd &train_y, VectorXd &train_weight, VectorXi &A,
                              VectorXi &g_index, VectorXi &g_size, int train_n, int N)
{
    VectorXi A_ind = find_ind(A, g_index, g_size, (this->beta).rows(), N);
    UniversalData X_A = X_seg(train_x, train_n, A_ind, this->model_type);
    VectorXd beta_A;
    slice(this->beta, A_ind, beta_A);

    // coef0_old = this->coef0;
    bool success =
        this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX, A, g_index, g_size);
    // if (!success){
    //   this->coef0 = coef0_old;
    // }else{
    slice_restore(beta_A, A_ind, this->beta);
    this->train_loss = this->loss_function(X_A, train_y, train_weight, beta_A, this->coef0, A, g_index, g_size,
                                           this->lambda_level);
    // }
}
/*
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
*/
bool Algorithm::primary_model_fit(UniversalData &active_data, MatrixXd &y, VectorXd &weights, VectorXd &active_para, VectorXd &aux_para, double loss0,
                                  VectorXi &A, VectorXi &g_index, VectorXi &g_size)
{
    SPDLOG_DEBUG("optimization begin\nactive set: {}\ninit loss: {}\npara:{}", active_data.get_effective_para_index().transpose(), loss0, active_para.transpose());
    double value = active_data.optimize(active_para);
    SPDLOG_DEBUG("optimization end\nfinal loss: {}\npara:{}", value, active_para.transpose());
    return true;
    /*
    active_data.init_para(active_para);

    nlopt_opt opt = active_data.nlopt_create(active_para.size());
    nlopt_set_min_objective(opt, nlopt_function, &active_data);
    nlopt_result result = nlopt_optimize(opt, active_para.data(), &value);
    nlopt_destroy(opt);

    bool success = result > 0;
    if(!success){
        SPDLOG_WARN("nlopt failed to optimize, state: {} ", nlopt_result_to_string(result));
    }
    */
}

void Algorithm::sacrifice(UniversalData &data, UniversalData &XA, MatrixXd &y, VectorXd &para, VectorXd &beta_A, VectorXd &aux_para, VectorXi &A, VectorXi &I, VectorXd &weights, VectorXi &g_index, VectorXi &g_size, int g_num, VectorXi &A_ind, VectorXd &sacrifice, VectorXi &U, VectorXi &U_ind, int num)
{
    SPDLOG_DEBUG("sacrifice begin");
    VectorXd gradient_full;
    MatrixXd hessian_full;
    data.gradient_and_hessian(para, gradient_full, hessian_full);

    int size, index;
    for (auto group_index : A)
    {
        size = g_size(group_index);
        index = g_index(group_index);
        VectorXd gradient_group = gradient_full.segment(index, size);
        MatrixXd hessian_group = hessian_full.block(index, index, size, size);
        if (size == 1)
        { // optimize for frequent degradation situations
            sacrifice(group_index) = para(index) * para(index) * hessian_group(0, 0);
        }
        else
        {
            sacrifice(group_index) = para.segment(index, size).transpose() * hessian_group * para.segment(index, size);
            sacrifice(group_index) /= size;
        }
    }
    for (auto group_index : I)
    {
        size = g_size(group_index);
        index = g_index(group_index);
        VectorXd gradient_group = gradient_full.segment(index, size);
        MatrixXd hessian_group = hessian_full.block(index, index, size, size);
        if (size == 1)
        { // optimize for frequent degradation situations
            if (hessian_group(0, 0) < this->enough_small)
            {
                SPDLOG_ERROR("there exists a submatrix of hessian which is not positive definite!\nactive set is{}\nactive params are {}\ngroup index is {}, hessian is {}", data.get_effective_para_index().transpose(), para.transpose(), index, hessian_group(0, 0));
                sacrifice(group_index) = gradient_group(0, 0) * gradient_group(0, 0);
            }
            else
            {
                sacrifice(group_index) = gradient_group(0, 0) * gradient_group(0, 0) / hessian_group(0, 0);
            }
        }
        else
        {
            LLT<MatrixXd> hessian_group_llt(hessian_group);
            if (hessian_group_llt.info() == NumericalIssue)
            {
                SPDLOG_ERROR("there exists a submatrix of hessian which is not positive definite!\nactive set is {}\nactive params are {}\ngroup index is {}\nhessian is {}", data.get_effective_para_index().transpose(), para.transpose(), VectorXi::LinSpaced(size, index, size + index - 1), hessian_group);
                sacrifice(group_index) = gradient_group.squaredNorm();
            }
            else
            {
                MatrixXd inv_hessian_group = hessian_group_llt.solve(MatrixXd::Identity(size, size));
                sacrifice(group_index) = gradient_group.transpose() * inv_hessian_group * gradient_group;
                sacrifice(group_index) /= size;
            }
        }
    }
    SPDLOG_DEBUG("sacrifice end with {}", sacrifice.transpose());
    return;
}
