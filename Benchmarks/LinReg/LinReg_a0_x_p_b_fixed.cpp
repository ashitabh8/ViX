#include "../../include/distributions_latest.hpp"
#include "../../include/ios.hpp"
#include <iostream>
#include <fstream>
#include <random>


// using fixed_10_6 = fpm::fixed<std::int16_t, std::int32_t, 6>;

#define T fpm::fixed<std::int32_t, std::int64_t, 8>
#define SIGMA_LB 0.5
#define SIGMA_UB 2
#define BITSHIFT 9
#define EARLY_STOPPING 5

// template<typename K = float, int N>
// std::array<float, N> linspace(K start, K end)
// {
//   std::array<float, N> linspaced;

//   float delta = (end - start) / (N - 1);

//   for(int i=0; i < N-1; ++i)
//     {
//       linspaced[i] = (start + delta * i);
//     }
//   linspaced[N-1] = (end); // I want to ensure that start and end
//                             // are exactly the same as the input
//   return linspaced;
// }


template<int NUM_DATA_SAMPLES>
auto gen_data()
{
     float SIGMA = 0.5;
     float M = 6;
     float B = 2;
    std::default_random_engine gen;
    // const unsigned NUM_DATA_SAMPLES = 100;

    std::array<float, NUM_DATA_SAMPLES> y_samples_temp;
    std::array<float, NUM_DATA_SAMPLES> x_samples_temp = linspace<float, NUM_DATA_SAMPLES>(-5,5);
    for(int i = 0; i < NUM_DATA_SAMPLES; i++)
    {
        std::normal_distribution<float> dist(M * x_samples_temp[i]  + B, SIGMA);
        y_samples_temp[i] =dist(gen);
    }
    std::array<T, NUM_DATA_SAMPLES> y_samples;
    std::array<T, NUM_DATA_SAMPLES> x_samples;
    for(int i = 0; i < NUM_DATA_SAMPLES; i++)
    {
        y_samples[i] = (T) y_samples_temp[i];
        x_samples[i] = (T) x_samples_temp[i];
    }

    return std::make_pair(x_samples, y_samples);
}

int main()
{
    // Ground truth - Calculated using OLS LinReg sklearn
    // 2* x + 6
    // y  = Normal(a0 * x + a1, sigma)

    // std::ofstream output_file("Linre6_x_2_data_32bit.csv");
    std::ofstream result_file("Linre6_x_2_result_fixed.csv");
    const int NUM_DATA_SAMPLES = 1000;
    auto data_pair = gen_data<NUM_DATA_SAMPLES>();

    auto x_test = data_pair.first;
    auto y_test = data_pair.second;

    // for(int i = 0; i < NUM_DATA_SAMPLES; i++)
    // {
    //     output_file << x_test[i] << "," << y_test[i] << std::endl;
    // }
    // abort();
    //Variational assignment of bias: a1


    //  Variational assignment of coefficent: a0
    Variable<T> a0_mean(1), a0_sigma(1);
    Constant<T> a0_sample(0);
    auto a0 = normal_dist<T>(a0_mean, a0_sigma,a0_sample);

    // Variational assignment of coefficent: a1
    Variable<T> a1_mean(1), a1_sigma(1);
    Constant<T> a1_sample(0);
    auto a1 = normal_dist<T>(a1_mean, a1_sigma,a1_sample);

    // input data: x
    Constant<T> X(0);

    // Linear regression BN prior
    Constant<T> lb_a0(-24), ub_a0(24);
    auto a0_prior = uniform_dist<T>(lb_a0, ub_a0, a0_sample);

    Constant<T> lb_a1(-8), ub_a1(8);
    auto a1_prior = uniform_dist<T>(lb_a1, ub_a1, a1_sample);

    // Constant<T> a1_true(6);
    Constant<T> sigma(2);
    Constant<T> Y_container(0);
    auto LinReg_eq = a0 * X  + a1;
    auto Y = normal_dist<T>(LinReg_eq, sigma,Y_container);

    const int NUM_SAMPLES = 4;
    constexpr int BITSHIFT_SAMPLES = std::log2(NUM_SAMPLES);
    constexpr T SCALING = (T) std::pow(2, BITSHIFT);
    int NUM_ITERS = 500;
    T learning_rate = (T) 0.01;
    T LB = (T) -128;
    T UB = (T) 128;
    T current_del_elbo_mu_a0 = (T) 0;
    T current_del_elbo_sigma_a0 = (T) 0;
    T current_del_elbo_mu_a1 = (T) 0;
    T current_del_elbo_sigma_a1 = (T) 0;
    T zero = (T) 0;
    T one = (T) 1;
    T current_score = (T) 0;
    T total_loss  = (T) 0;
    T Beta = (T) 0.9;
    T V_t_minus_1 = (T) 0;
    T V_t = (T) 0;
    T S_t_minus_1 = (T) 0;
    T S_t = (T) 0;
    T second_term = (T) 0;
    T curr_sample_a0 = (T) 0;
    T curr_sample_a1 = (T) 0;
    T step_a0_mu = (T) 0;
    T step_a0_sigma = (T) 0;
    T step_a1_mu = (T) 0;
    T step_a1_sigma = (T) 0;
    T new_value_a0_mu = (T) 0;
    T new_value_a0_sigma = (T) 0;
    T new_value_a1_mu = (T) 0;
    T new_value_a1_sigma = (T) 0;
    result_file << "iter,elbo,a0_mu,a0_sigma,a1_mu,a1_sigma" << std::endl;
    std::array<T, EARLY_STOPPING> loss_arr = {0,0,0,0,0};
    for(int c_iter = 0; c_iter < NUM_ITERS; c_iter++)
    {
        // std::cout << "Iteration: " << c_iter << std::endl;
        current_del_elbo_mu_a0 = zero;
        current_del_elbo_sigma_a0 = zero;
        current_del_elbo_mu_a1 = zero;
        current_del_elbo_sigma_a1 = zero;

        
        total_loss = zero;
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            a0_sample(a0.sample());
            a1_sample(a1.sample()); // Seeding better will fix this problem
            a1_sample(a1.sample());

            while(curr_sample_a1 < a1_prior.lb_value() || curr_sample_a1 > a1_prior.ub_value());
            current_score = (T) 0;

            for(int data_k = 0; data_k < NUM_DATA_SAMPLES; data_k++)
            {
                X(x_test[data_k]);
                Y_container(y_test[data_k]);

                    current_score += T::from_raw_value(Y.log_pdf().value().raw_value() >> BITSHIFT);
            }

            second_term = current_score + (a0_prior.log_pdf().value() + a1_prior.log_pdf().value() - a0.log_pdf().value() - a1.log_pdf().value())/SCALING;
            total_loss += second_term;
            current_del_elbo_mu_a0 += a0.diff_log(a0_mean)*(second_term);
            current_del_elbo_sigma_a0 += a0.diff_log(a0_sigma)*(second_term);
            current_del_elbo_mu_a1 += a1.diff_log(a1_mean)*(second_term);
            current_del_elbo_sigma_a1 += a1.diff_log(a1_sigma)*(second_term);
        }

        step_a0_mu = grad_clipping<T>(T::from_raw_value(current_del_elbo_mu_a0.raw_value()>> BITSHIFT_SAMPLES), LB, UB); // Remove these extra variables when doing performance testing
        step_a0_sigma =grad_clipping<T>(T::from_raw_value(current_del_elbo_sigma_a0.raw_value()>> BITSHIFT_SAMPLES), LB, UB);
        
        step_a1_mu = grad_clipping<T>(T::from_raw_value(current_del_elbo_mu_a1.raw_value()>> BITSHIFT_SAMPLES), LB, UB); // Remove these extra variables when doing performance testing
        step_a1_sigma =grad_clipping<T>(T::from_raw_value(current_del_elbo_sigma_a1.raw_value()>> BITSHIFT_SAMPLES), LB, UB);

        result_file << c_iter << "," << total_loss/NUM_SAMPLES << "," << a0_mean.value() << "," << a0_sigma.value() << "," << a1_mean.value() << "," << a1_sigma.value() << std::endl;

        new_value_a0_mu = a0_mean.value() + learning_rate * (step_a0_mu);
        new_value_a0_sigma = a0_sigma.value() + learning_rate * (step_a0_sigma);
        new_value_a1_mu = a1_mean.value() + learning_rate * (step_a1_mu);
        new_value_a1_sigma = a1_sigma.value() + learning_rate * (step_a1_sigma);


        a0_mean(grad_clipping<T>(new_value_a0_mu, a0_prior.lb_value(), a0_prior.ub_value()));
        a0_sigma(grad_clipping<T>(new_value_a0_sigma, SIGMA_LB, SIGMA_UB));
        a1_mean(grad_clipping<T>(new_value_a1_mu, a1_prior.lb_value(), a1_prior.ub_value()));
        a1_sigma(grad_clipping<T>(new_value_a1_sigma, SIGMA_LB, SIGMA_UB));
    }
}