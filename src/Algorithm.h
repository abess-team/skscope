/**
 * @file Algorithm.h
 * @brief the algorithm for fitting on given parameter.
 * @author   Jin Zhu (zhuj37@mail2.sysu.edu.cn),
 * Kangkang Jiang (jiangkk3@mail2.sysu.edu.cn),
 * Junhao Huang (huangjh256@mail2.sysu.edu.cn)
 * @version  0.0.1
 * @date     2021-07-31
 * @copyright  GNU General Public License (GPL)
 */

/*****************************************************************************
 *  OpenST Basic tool library                                                 *
 *  Copyright (C) 2021 Kangkang Jiang  jiangkk3@mail2.sysu.edu.cn                         *
 *                                                                            *
 *  This file is part of OST.                                                 *
 *                                                                            *
 *  This program is free software; you can redistribute it and/or modify      *
 *  it under the terms of the GNU General Public License version 3 as         *
 *  published by the Free Software Foundation.                                *
 *                                                                            *
 *  You should have received a copy of the GNU General Public License         *
 *  along with OST. If not, see <http://www.gnu.org/licenses/>.               *
 *                                                                            *
 *  Unless required by applicable law or agreed to in writing, software       *
 *  distributed under the License is distributed on an "AS IS" BASIS,         *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 *  See the License for the specific language governing permissions and       *
 *  limitations under the License.                                            *
 *                                                                            *
 *----------------------------------------------------------------------------*
 *  Remark         : Description                                              *
 *----------------------------------------------------------------------------*
 *  Change History :                                                          *
 *  <Date>     | <Version> | <Author>       | <Description>                   *
 *----------------------------------------------------------------------------*
 *  2021/07/31 | 0.0.1     | Kangkang Jiang | First version                   *
 *----------------------------------------------------------------------------*
 *                                                                            *
 *****************************************************************************/

#pragma once

#include <Eigen/Eigen>
#include <cfloat>

#include "utilities.h"
#include "UniversalData.h"

/**
 * @brief Variable select based on splicing algorithm.
 */

class Algorithm
{
public:
    int model_fit_max;       // Maximum number of iterations taken for the primary model fitting.
    int model_type;          // primary model type.
    int algorithm_type;      // algorithm type.
    int group_df = 0;        // freedom
    int sparsity_level = 0;  // Number of non-zero coefficients.
    double lambda_level = 0; // l2 normalization coefficients.
    // Eigen::VectorXi train_mask;
    int max_iter;     // Maximum number of iterations taken for the splicing algorithm to converge.
    int exchange_num; // Max exchange variable num.
    bool warm_start;  // When tuning the optimal parameter combination, whether to use the last solution as a warm start
                      // to accelerate the iterative convergence of the splicing algorithm.
    UniversalData *x = NULL;
    Eigen::MatrixXd *y = NULL;
    Eigen::VectorXd beta;       // coefficients.
    Eigen::VectorXd bd;         // sacrifices.
    Eigen::VectorXd coef0;      // intercept.
    double train_loss = 0.;     // train loss.
    Eigen::VectorXd beta_init;  // initialization coefficients.
    Eigen::VectorXd coef0_init; // initialization intercept.
    Eigen::VectorXi A_init;     // initialization active set.
    Eigen::VectorXi I_init;     // initialization inactive set.
    Eigen::VectorXd bd_init;    // initialization bd vector.

    Eigen::VectorXi A_out; // final active set.
    Eigen::VectorXi I_out; // final active set.

    bool lambda_change; // lambda_change or not.

    Eigen::VectorXi always_select;    // always select variable.
    double tau;                       // algorithm stop threshold
    int primary_model_fit_max_iter;   // The maximal number of iteration for primaty model fit
    double primary_model_fit_epsilon; // The epsilon (threshold) of iteration for primaty model fit

    Eigen::VectorXd beta_warmstart;  // warmstart beta.
    Eigen::VectorXd coef0_warmstart; // warmstart intercept.

    double effective_number; // effective number of parameter.
    int splicing_type;       // exchange number update mathod.
    int sub_search;          // size of sub_searching in splicing
    int U_size;
    double enough_small = 1e-9;

    Algorithm(int max_iter = 30,
              bool warm_start = true,
              int exchange_num = 5,
              Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0),
              int splicing_type = 0,
              int sub_search = 0)
        : algorithm_type(6),
          model_type(0),
          max_iter(max_iter),
          primary_model_fit_max_iter(10),
          primary_model_fit_epsilon(1e-8),
          warm_start(warm_start),
          exchange_num(exchange_num),
          always_select(always_select),
          splicing_type(splicing_type),
          sub_search(sub_search) {}

    void set_warm_start(bool warm_start) { this->warm_start = warm_start; }

    void update_beta_init(Eigen::VectorXd &beta_init) { this->beta_init = beta_init; }

    void update_A_init(Eigen::VectorXi &A_init, int g_num)
    {
        this->A_init = A_init;
        this->I_init = complement(A_init, g_num);
    }

    void update_bd_init(Eigen::VectorXd &bd_init) { this->bd_init = bd_init; }

    void update_coef0_init(Eigen::VectorXd coef0) { this->coef0_init = coef0; }

    void update_group_df(int group_df) { this->group_df = group_df; }

    void update_sparsity_level(int sparsity_level) { this->sparsity_level = sparsity_level; }

    void update_lambda_level(double lambda_level)
    {
        this->lambda_change = this->lambda_level != lambda_level;
        this->lambda_level = lambda_level;
    }

    void update_exchange_num(int exchange_num) { this->exchange_num = exchange_num; }

    void update_tau(int n, int p)
    {
        if (n == 1)
        {
            this->tau = 0.0;
        }
        else
        {
            this->tau =
                0.01 * (double)this->sparsity_level * log((double)p) * log(log((double)n)) / (double)n;
        }
    }

    bool get_warm_start() { return this->warm_start; }

    double get_train_loss() { return this->train_loss; }

    int get_group_df() { return this->group_df; }

    double get_effective_number() { return this->effective_number; }

    int get_sparsity_level() { return this->sparsity_level; }

    Eigen::VectorXd get_beta() { return this->beta; }

    Eigen::VectorXd get_coef0() { return this->coef0; }

    Eigen::VectorXi get_A_out() { return this->A_out; };

    Eigen::VectorXi get_I_out() { return this->I_out; };

    Eigen::VectorXd get_bd() { return this->bd; }

    /**
     * @param train_x sample matrix for training
     * @param train_y response matrix for training
     * @param train_weight weight of each sample
     * @param g_index the first position of each group
     * @param g_size size of each group
     * @param train_n sample size for training, i.e. the number of rows in `train_x`
     * @param p number of variables, i.e. the number of columns in `train_x`
     * @param N number of different groups
     */
    void fit(UniversalData &train_x, Eigen::MatrixXd &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
             int train_n, int p, int N);

    void get_A(UniversalData &X, Eigen::MatrixXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, Eigen::VectorXd &coef0,
               Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
               int N, double tau, double &train_loss);

    bool splicing(UniversalData &X, Eigen::MatrixXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, Eigen::VectorXd &coef0,
                  Eigen::VectorXd &bd, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
                  int N, double tau, double &train_loss);

    Eigen::VectorXi inital_screening(UniversalData &X, Eigen::MatrixXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I,
                                     Eigen::VectorXd &bd, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                                     Eigen::VectorXi &g_size, int &N);

    void final_fitting(UniversalData &train_x, Eigen::MatrixXd &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &A,
                       Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int train_n, int N);

    /**
     * Compute the loss of active_data with L2 penalty, where the value of parameter is active_para.
     * Only these three paras will be used.
     * @param active_data                                          UniversalData which has been limited to active set A
     * @param active_para                                          the value of active parameters adapted to active_data
     *
     * @return a double value indicating the loss
     */
    double loss_function(UniversalData &active_data, MatrixXd &y, VectorXd &weights, VectorXd &active_para, VectorXd &aux_para, VectorXi &A,
                         VectorXi &g_index, VectorXi &g_size, double lambda)
    {
        return active_data.loss(active_para);
    }
    /**
     * compute the sacrifice of data
     * Only these seven paras will be used.
     * @param data                                                 UniversalData which include both active set A and inactive set I
     * @param para                                                 the value of effective parameters adapted to data
     * @param A                                                    the index in g_index of group in active set
     * @param I                                                    the index in g_index of group in inactive set
     * @param g_index                                              the index in para of all groups
     * @param g_size                                               the length of all groups
     * @param sacrifice                                            a column vector which will be replaced by results
     */
    void sacrifice(UniversalData &data, UniversalData &XA, MatrixXd &y, VectorXd &para, VectorXd &beta_A, VectorXd &aux_para, VectorXi &A, VectorXi &I, VectorXd &weights, VectorXi &g_index, VectorXi &g_size, int g_num, VectorXi &A_ind, VectorXd &sacrifice, VectorXi &U, VectorXi &U_ind, int num);

    /**
     * optimize the loss of active_data with L2 penalty
     * Only these three paras will be used.
     * @param active_data                                          UniversalData which will be optimized, it has been limited to active set A
     * @param active_para                                          a column vector of initial values for active parameters
     *
     * @return a boolean value indicating successful completion of the optimization algorithm, and results are stored in active_para and aux_para.
     */
    bool primary_model_fit(UniversalData &active_data, MatrixXd &y, VectorXd &weights, VectorXd &active_para, VectorXd &aux_para, double loss0,
                           VectorXi &A, VectorXi &g_index, VectorXi &g_size);
    /**
     * compute the effective number of parameters which will be used to compute information criterion
     * Only these two paras will be used.
     * @param active_data                                          UniversalData which has been limited to active set A
     *
     * @return a double value indicating the effective number of parameters
     */
    double effective_number_of_parameter(UniversalData& X, UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& beta, VectorXd& active_para, VectorXd& aux_para) { return active_data.cols(); }
};
