// #pragma once
#include <iostream>
#include <fstream>
// #include "../include/ad_engine.hpp"
#include "../../include/distributions_interval.hpp"


#define C_int Constant<interval>
#define V_int Variable<interval>


int main()
{
    C_int a0_prior(0,75);
    C_int a1_prior(0,36);

    C_int X_data(-5,5);C_int Y_data(-1.2,675.68);

    auto Y_q = a0_prior * X_data * X_data + a1_prior * X_data;

    C_int sigma(7,7);
    auto Y = normal_dist_INTERVAL(Y_q, sigma, Y_data);

    V_int a0_mu_var(0,75), a0_sigma(0.1, 5);
    auto a0_variational = normal_dist_INTERVAL(a0_mu_var, a0_sigma, a0_prior);

    V_int a1_mu_var(0,36), a1_sigma(0.1, 5);
    auto a1_variational = normal_dist_INTERVAL(a1_mu_var, a1_sigma, a1_prior);

    C_int num_data(50,50);

    C_int a0_prior_log(1.87,1.87), a1_prior_log(1.55,1.55);

    auto everything_but_derivative =  ( num_data * Y.log_pdf() + a0_prior_log + a1_prior_log) - a0_variational.log_pdf() - a1_variational.log_pdf();
    std::cout <<"Everything but derivative: " << std::endl;
    everything_but_derivative.value().see();

    C_int derivative_wrt_a0_mu = C_int(a0_variational.diff_log(a0_mu_var) );
    std::cout <<"Derivative wrt a0 mu: " << std::endl;
    derivative_wrt_a0_mu.value().see();

    C_int derivative_wrt_a0_sigma = C_int(a0_variational.diff_log(a0_sigma));
    std::cout <<"Derivative wrt a0 sigma: " << std::endl;
    derivative_wrt_a0_sigma.value().see();

    C_int derivative_wrt_a1_mu = C_int(a1_variational.diff_log(a1_mu_var) );
    std::cout <<"Derivative wrt a1 mu: " << std::endl;
    derivative_wrt_a1_mu.value().see();

    C_int derivative_wrt_a1_sigma = C_int(a1_variational.diff_log(a1_sigma));
    std::cout <<"Derivative wrt a1 sigma: " << std::endl;
    derivative_wrt_a1_sigma.value().see();

}