#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <Eigen/Eigen>
#include <tuple>
#include <vector>

#include "Algorithm.h"
#include "UniversalData.h"
#include "Data.h"
#include "Metric.h"
#include "OpenMP.h"
#include "path.h"
#include "screening.h"
#include "utilities.h"
#include "predefined_model.hpp"

using namespace Eigen;
using namespace std;

/**
 * @brief The main function of scope
 * @param data                          a pybind11 warpper of UniversalData
 * @param universal_model                         a pybind11 warpper of UniversalModel
 * @param model_size                    The number of the parameters.
 * @param sample_size                   The number of the samples.
 * @param weight                        Individual weights for each sample.
 * @param sigma                         Sample covariance matrix. For PCA problem under Kfold=1, it should be given as
 * input, instead of X.
 * @param normalize_type                Type of normalization on X before fitting the algorithm. If normalize_type=0,
 * normalization will not be used.
 * @param algorithm_type                Algorithm type.
 * @param model_type                    Model type.
 * @param max_iter                      Maximum number of iterations taken for the splicing algorithm to converge.
 *                                      Due to the limitation of loss reduction, the splicing algorithm must be able to
 * converge. The number of iterations is only to simplify the implementation.
 * @param exchange_num                  Max exchange variable num for the splicing algorithm.
 * @param path_type                     The method to be used to select the optimal support size.
 *                                      For path_type = 1, we solve the best subset selection problem for each size in
 * support_size. For path_type = 2, we solve the best subset selection problem with support size ranged in (s_min,
 * s_max), where the specific support size to be considered is determined by golden section.
 * @param is_warm_start                 When tuning the optimal parameter combination, whether to use the last solution
 * as a warm start to accelerate the iterative convergence of the splicing algorithm.
 * @param ic_type                       The type of criterion for choosing the support size. Available options are
 * "sic", "ebic", "bic", "aic".
 * @param Kfold                         The folds number to use the Cross-validation method. If Kfold=1,
 * Cross-validation will not be used.
 * @param sequence                      An integer vector representing the alternative support sizes. Only used for
 * path_type = 1.
 * @param s_min                         The lower bound of golden-section-search for sparsity searching. Only used for
 * path_type = 2.
 * @param s_max                         The higher bound of golden-section-search for sparsity searching. Only used for
 * path_type = 2.
 * @param thread                        Max number of multithreads. If thread = 0, the program will use the maximum
 * number supported by the device.
 * @param screening_size                Screen the variables first and use the chosen variables in abess process.
 *                                      The number of variables remaining after screening. It should be an integer
 * smaller than p. If screen_size = -1, screening will not be used.
 * @param g_index                       The group index for each variable.
 * @param always_select                 An array contains the indexes of variables we want to consider in the model.
 * @param primary_model_fit_max_iter    The maximal number of iteration in `primary_model_fit()` (in Algorithm.h).
 * @param primary_model_fit_epsilon     The epsilon (threshold) of iteration in `primary_model_fit()` (in Algorithm.h).
 * @param splicing_type                 The type of splicing in `fit()` (in Algorithm.h).
 *                                      "0" for decreasing by half, "1" for decresing by one.
 * @param sub_search                    The number of inactive sets that are split when splicing. It should be a
 * positive integer.
 * @return result list.
 */

/**
 * @brief The main workflow for abess.
 * @tparam MatrixXd for y, XTy, XTone
 * @tparam VectorXd for beta
 * @tparam VectorXd for coef0
 * @tparam UniversalData for X
 * @param x sample matrix
 * @param y response matrix
 * @param n sample size
 * @param model_size number of variables
 * @param normalize_type type of normalize
 * @param weight weight of each sample
 * @param algorithm_type type of algorithm
 * @param path_type type of path: 1 for sequencial search and 2 for golden section search
 * @param is_warm_start whether enable warm-start
 * @param ic_type type of information criterion, used for not CV
 * @param Kfold number of folds, used for CV
 * @param parameters parameters to be selected, including `support_size`, `lambda`
 * @param screening_size size of screening
 * @param g_index the first position of each group
 * @param early_stop whether enable early-stop
 * @param thread number of threads used for parallel computing
 * @param sparse_matrix whether sample matrix `x` is sparse matrix
 * @param cv_fold_id user-specified cross validation division
 * @param A_init initial active set
 * @param algorithm_list the algorithm pointer
 * @return the result of abess, including the best model parameters
 */

tuple<VectorXd, double, double, double>
pywrap_Universal(pybind11::object universal_data, UniversalModel universal_model, ConvexSolver convex_solver, int model_size, int sample_size, int aux_para_size, int max_iter,
                 int exchange_num, int path_type, bool is_greedy, bool use_hessian, bool is_dynamic_exchange_num, bool is_warm_start, int ic_type, double ic_coef, int Kfold, VectorXi sequence,
                 VectorXd lambda_seq, int s_min, int s_max, int screening_size, VectorXi g_index, VectorXi always_select,
                 int thread, int splicing_type, int sub_search, VectorXi cv_fold_id, VectorXi A_init, VectorXd beta_init, VectorXd coef0_init)
{
#ifdef _OPENMP
    // initParallel();
    int max_thread = omp_get_max_threads();
    if (thread == 0 || thread > max_thread)
    {
        thread = max_thread;
    }

    setNbThreads(thread);
    omp_set_num_threads(thread);
#endif

    SPDLOG_DEBUG("SCOPE begin!");
    UniversalData x(model_size, sample_size, universal_data, &universal_model, convex_solver); // UniversalData is just like a matrix.
    MatrixXd y = MatrixXd::Zero(sample_size, aux_para_size);                                   // Invalid variable, create it just for interface compatibility
    int normalize_type = 0;                                                                    // offer normalized data if need
    VectorXd weight = VectorXd::Ones(sample_size);                                             // only can be implemented inside the model
    Parameters parameters(sequence, lambda_seq, s_min, s_max);

    int algorithm_list_size = max(thread, Kfold);
    vector<Algorithm *> algorithm_list(algorithm_list_size);
    for (int i = 0; i < algorithm_list_size; i++)
    {
        algorithm_list[i] = new Algorithm(max_iter, is_warm_start, exchange_num, always_select, splicing_type, is_greedy, sub_search, use_hessian, is_dynamic_exchange_num);
    }

    bool early_stop = true, sparse_matrix = true;
    int beta_size = model_size;

    // Data packing & normalize:
    //     pack & initial all information of data,
    //     including normalize.
    Data data(x, y, normalize_type, weight, g_index, sparse_matrix, beta_size);

    // Screening:
    //     if there are too many noise variables,
    //     screening can choose the `screening_size` most important variables
    //     and then focus on them later.
    VectorXi screening_A;
    if (screening_size >= 0)
    {
        screening_A = screening(data, algorithm_list, screening_size, beta_size,
                                                                             parameters.lambda_list(0), A_init);
    }

    // Prepare for CV:
    //     if CV is enable,
    //     specify train and test data,
    //     and initialize the fitting argument inside each fold.
    Metric *metric = new Metric(ic_type, ic_coef, Kfold);
    if (Kfold > 1)
    {
        metric->set_cv_train_test_mask(data, data.n, cv_fold_id);
        metric->set_cv_init_fit_arg(beta_size, data.M);
    }

    // Fitting and loss:
    //     follow the search path,
    //     fit on each parameter combination,
    //     and calculate ic/loss.
    vector<Result> result_list(Kfold);
    if (path_type == 1)
    {
        // sequentical search
#pragma omp parallel for
        for (int i = 0; i < Kfold; i++)
        {
            sequential_path_cv(data, algorithm_list[i], metric, parameters, early_stop, i, A_init, beta_init, coef0_init, result_list[i]);
        }
    }
    else
    {
        // golden section search
        gs_path(data, algorithm_list, metric, parameters, A_init, beta_init, coef0_init, result_list);
    }

    // Get bestmodel && fit bestmodel:
    //     choose the best model with lowest ic/loss
    //     and if CV, refit on full data.
    int min_loss_index = 0;
    int sequence_size = (parameters.sequence).size();
    Matrix<VectorXd, Dynamic, 1> beta_matrix(sequence_size, 1);
    Matrix<VectorXd, Dynamic, 1> coef0_matrix(sequence_size, 1);
    Matrix<VectorXd, Dynamic, 1> bd_matrix(sequence_size, 1);
    MatrixXd ic_matrix(sequence_size, 1);
    MatrixXd test_loss_sum = MatrixXd::Zero(sequence_size, 1);
    MatrixXd train_loss_matrix(sequence_size, 1);
    MatrixXd effective_number_matrix(sequence_size, 1);

    if (Kfold == 1)
    {
        // not CV: choose lowest ic
        beta_matrix = result_list[0].beta_matrix;
        coef0_matrix = result_list[0].coef0_matrix;
        ic_matrix = result_list[0].ic_matrix;
        train_loss_matrix = result_list[0].train_loss_matrix;
        effective_number_matrix = result_list[0].effective_number_matrix;
        ic_matrix.col(0).minCoeff(&min_loss_index);
    }
    else
    {
        // CV: choose lowest test loss
        for (int i = 0; i < Kfold; i++)
        {
            test_loss_sum += result_list[i].test_loss_matrix;
        }
        test_loss_sum /= ((double)Kfold);
        test_loss_sum.col(0).minCoeff(&min_loss_index);

        VectorXi used_algorithm_index = VectorXi::Zero(algorithm_list_size);

        // refit on full data
#pragma omp parallel for
        for (int ind = 0; ind < sequence_size; ind++)
        {
            int support_size = parameters.sequence(ind).support_size;
            double lambda = parameters.sequence(ind).lambda;

            int algorithm_index = omp_get_thread_num();
            used_algorithm_index(algorithm_index) = 1;

            VectorXd beta_init;
            VectorXd coef0_init;
            VectorXi A_init; // start from a clear A_init (not from the given one)
            coef_set_zero(beta_size, data.M, beta_init, coef0_init);
            VectorXd bd_init = VectorXd::Zero(data.g_num);

            // warmstart from CV's result
            for (int j = 0; j < Kfold; j++)
            {
                beta_init = beta_init + result_list[j].beta_matrix(ind) / Kfold;
                coef0_init = coef0_init + result_list[j].coef0_matrix(ind) / Kfold;
                bd_init = bd_init + result_list[j].bd_matrix(ind) / Kfold;
            }

            // fitting
            algorithm_list[algorithm_index]->update_sparsity_level(support_size);
            algorithm_list[algorithm_index]->update_lambda_level(lambda);
            algorithm_list[algorithm_index]->update_beta_init(beta_init);
            algorithm_list[algorithm_index]->update_coef0_init(coef0_init);
            algorithm_list[algorithm_index]->update_bd_init(bd_init);
            algorithm_list[algorithm_index]->update_A_init(A_init, data.g_num);
            algorithm_list[algorithm_index]->fit(data.x, data.y, data.weight, data.g_index, data.g_size, data.n, data.p,
                                                 data.g_num);

            // update results
            beta_matrix(ind) = algorithm_list[algorithm_index]->get_beta();
            coef0_matrix(ind) = algorithm_list[algorithm_index]->get_coef0();
            train_loss_matrix(ind) = algorithm_list[algorithm_index]->get_train_loss();
            ic_matrix(ind) = metric->ic(data.n, data.M, data.g_num, algorithm_list[algorithm_index]);
            effective_number_matrix(ind) = algorithm_list[algorithm_index]->get_effective_number();
        }
    }

    VectorXd beta;
    if (screening_size < 0)
    {
        beta = beta_matrix(min_loss_index);
    }
    else
    {
        // restore the changes if screening is used.
        beta = VectorXd::Zero(model_size);
        beta(screening_A) = beta_matrix(min_loss_index);
    }

    delete metric;
    for (int i = 0; i < algorithm_list_size; i++)
    {
        delete algorithm_list[i];
    }
    SPDLOG_DEBUG("SCOPE end!");

    return make_tuple(beta,
                      train_loss_matrix(min_loss_index),
                      test_loss_sum(min_loss_index),
                      ic_matrix(min_loss_index));
}

PYBIND11_MODULE(_scope, m)
{
    m.def("pywrap_Universal", &pywrap_Universal);
    pybind11::class_<UniversalModel>(m, "UniversalModel").def(pybind11::init<>()).def("set_loss_of_model", &UniversalModel::set_loss_of_model).def("set_gradient_autodiff", &UniversalModel::set_gradient_autodiff).def("set_hessian_autodiff", &UniversalModel::set_hessian_autodiff).def("set_gradient_user_defined", &UniversalModel::set_gradient_user_defined).def("set_hessian_user_defined", &UniversalModel::set_hessian_user_defined).def("set_slice_by_sample", &UniversalModel::set_slice_by_sample).def("set_deleter", &UniversalModel::set_deleter);
    m.def("init_spdlog", &init_spdlog);
    //pybind11::class_<NloptConfig>(m, "NloptConfig").def(pybind11::init<int, const char *, double, double, double, double, double, unsigned, unsigned>());
    pybind11::class_<QuadraticData>(m, "QuadraticData")
        .def(pybind11::init<MatrixXd, VectorXd>());
    m.def("quadratic_loss", &quadratic_loss<double>);
    m.def("quadratic_loss", &quadratic_loss<dual>);
    m.def("quadratic_loss", &quadratic_loss<dual2nd>);
    m.def("quadratic_grad", &quadratic_grad);
    m.def("quadratic_hess", &quadratic_hess);
}
