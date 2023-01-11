

#include <Eigen/Eigen>

#include "List.h"



#include <iostream>
#include <vector>

#include "Algorithm.h"
#include "AlgorithmUniversal.h"
#include "utilities.h"
#include "workflow.h"
#include "api.h"

typedef Eigen::Triplet<double> triplet;

using namespace Eigen;
using namespace std;



List abessUniversal_API(ExternData data, UniversalModel model, int model_size, int sample_size, int aux_para_size, int max_iter, int exchange_num, int path_type,
    bool is_warm_start, int ic_type, double ic_coef, int Kfold, Eigen::VectorXi sequence, Eigen::VectorXd lambda_seq, int s_min, int s_max,
    int screening_size, Eigen::VectorXi g_index, Eigen::VectorXi always_select, int thread, int splicing_type, int sub_search,
    Eigen::VectorXi cv_fold_id, Eigen::VectorXi A_init, Eigen::VectorXd beta_init, Eigen::VectorXd coef0_init)

{
#ifdef _OPENMP
    // Eigen::initParallel();
    int max_thread = omp_get_max_threads();
    if (thread == 0 || thread > max_thread) {
        thread = max_thread;
    }

    Eigen::setNbThreads(thread);
    omp_set_num_threads(thread);
#endif

    SPDLOG_DEBUG("SCOPE begin!");
    UniversalData x(model_size, sample_size, data, &model); // UniversalData is just like a matrix.
    MatrixXd y = MatrixXd::Zero(sample_size, aux_para_size); // Invalid variable, create it just for interface compatibility
    int normalize_type = 0; // offer normalized data if need
    VectorXd weight = VectorXd::Ones(sample_size);  // only can be implemented inside the model
    Parameters parameters(sequence, lambda_seq, s_min, s_max);
    List out_result;

    int algorithm_list_size = max(thread, Kfold); 
    vector<Algorithm<MatrixXd, VectorXd, VectorXd, UniversalData>*> algorithm_list(algorithm_list_size);
    for (int i = 0; i < algorithm_list_size; i++) {
        algorithm_list[i] = new abessUniversal(max_iter, is_warm_start, exchange_num, always_select, splicing_type, sub_search);
    }

    bool early_stop = true, sparse_matrix = true;
    out_result = abessWorkflow<MatrixXd, VectorXd, VectorXd, UniversalData>(x, y, sample_size, model_size, normalize_type, weight, 6, path_type, is_warm_start, ic_type, ic_coef, Kfold,
        parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init, beta_init, coef0_init, algorithm_list);
    
    for (int i = 0; i < algorithm_list_size; i++) {
        delete algorithm_list[i];
    }
    SPDLOG_DEBUG("SCOPE end!");
    return out_result;
}
