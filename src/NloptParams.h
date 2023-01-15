# pragma once

#include <nlopt.h>
#include <cstring>
#include "utilities.h"

class NloptParams
{
private:
    nlopt_algorithm algorithm; 
    double stopval;
    double ftol_rel;
    double ftol_abs;
    double xtol_rel;
    double maxtime;
    unsigned population;
    unsigned vector_storage;
public:
    NloptParams(int algo_num, const char * algo_name, double stopval, double ftol_rel, double ftol_abs, double xtol_rel, double maxtime, unsigned population, unsigned vector_storage) : algorithm(static_cast<nlopt_algorithm>(algo_num)), stopval(stopval), ftol_rel(ftol_rel), xtol_rel(xtol_rel), maxtime(maxtime), population(population), vector_storage(vector_storage){
        if(std::strcmp(nlopt_algorithm_name(algorithm), algo_name) != 0){
            SPDLOG_ERROR("nlopt algorithm's setting failed! Maybe the version of nlopt doesn't match.\nThis is the specified algorithm: {0}\nBut this algorithm will be used: {1}", algo_name, nlopt_algorithm_name(algorithm));
        }
    }
    
    nlopt_opt create(unsigned dim){
        nlopt_opt opt = nlopt_create(algorithm, dim);
        nlopt_set_stopval(opt, stopval);
        nlopt_set_ftol_rel(opt, ftol_rel);
        nlopt_set_ftol_abs(opt, ftol_abs);
        nlopt_set_xtol_rel(opt, xtol_rel);
        nlopt_set_maxtime(opt, maxtime);
        nlopt_set_population(opt, population);
        nlopt_set_vector_storage(opt, vector_storage);
        return opt;
    }
};
