// #pragma once
#include <iostream>
#include <fstream>
// #include "../include/ad_engine.hpp"
#include "../../include/distributions_interval.hpp"


#define C_int Constant<interval>
#define V_int Variable<interval>


int main(){
    C_int a_prior(-18, 18);
    C_int X_data(-5,5);
    C_int Y_data(-0.61, 150.68);

    auto Y_q = a_prior * X_data * X_data;

    C_int sigma(0.2 , 0.2);
    auto Y = normal_dist_INTERVAL(Y_q, sigma, Y_data);

    V_int a_mu_var(-18, 18), a_sigma(0.1, 5);
    auto a_variational = normal_dist_INTERVAL(a_mu_var, a_sigma, a_prior);

    C_int num_data(50,50);

    C_int a_prior_log(-1.55,-1.55);

    auto everything_but_derivative =  ( num_data * Y.log_pdf() + a_prior_log) - a_variational.log_pdf();
    std::cout <<"Everything but derivative: " << std::endl;
    everything_but_derivative.value().see();

    C_int derivative_wrt_a_mu = C_int(a_variational.diff_log(a_mu_var) );
    std::cout <<"Derivative wrt a mu: " << std::endl;
    derivative_wrt_a_mu.value().see();

    C_int derivative_wrt_a_sigma = C_int(a_variational.diff_log(a_sigma));
    std::cout <<"Derivative wrt a sigma: " << std::endl;
    derivative_wrt_a_sigma.value().see();

    
}