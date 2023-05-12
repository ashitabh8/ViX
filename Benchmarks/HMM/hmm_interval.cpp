// #pragma once
#include <iostream>
#include <fstream>
// #include "../include/ad_engine.hpp"
#include "../../include/distributions_interval.hpp"

int main()
{

    Variable<interval> mu_var(50,200); //use this 
    // Constant<interval> data_var(250,310);
	Variable<interval> sigma_var(0.1,5); 
    Constant<interval> range(50,200);
 	auto normal_variational = normal_dist_INTERVAL(mu_var, sigma_var,range);

     Variable<interval> prior_mu_var(50,200);
    Variable<interval> prior_sigma_var(1,1);
 	auto prior_normal = normal_dist_INTERVAL(prior_mu_var, prior_sigma_var,mu_var);


    // Variable<interval> mu_obs(-750,750); //use this 
	Variable<interval> sigma_obs(0.5,0.5); 
    Constant<interval> range_obs(50,200);
 	auto N_obs = normal_dist_INTERVAL(mu_var, sigma_obs,range_obs);


   

    // Variable<interval> observed_mu_var(prior_normal.sample()); 
    // std::cout << "Observed mu: " << std::endl;
    // observed_mu_var.value().see();
    // // std::cout << "Observed mu: " << observed_mu_var.value() << std::endl;
    // Variable<interval> observed_sigma_var(2.,2.);
	// Constant<interval> sample_var(100.,100.);
 	// auto observed_normal = normal_dist_INTERVAL(observed_mu_var, observed_sigma_var,sample_var);

    auto everything_but_derivative = ( (prior_normal.log_pdf() +  N_obs.log_pdf()) - normal_variational.log_pdf());
    everything_but_derivative.value().see();


	Constant<interval> derivative_wrt_mu = Constant<interval>(normal_variational.diff_log(mu_var) );
    derivative_wrt_mu.value().see();

    Constant<interval> derivative_wrt_sigma = Constant<interval>(normal_variational.diff_log(sigma_var));
    derivative_wrt_sigma.value().see();


    // auto ELBO = everything_but_derivative * derivative_wrt_mu;
    // ELBO.value().see();

    // auto ELBO2 = everything_but_derivative * derivative_wrt_sigma;


	// interval range_ = observed_normal.sample();
	// range_.see();
}
