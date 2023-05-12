#include "../../include/distributions_latest.hpp"
#include "../../include/ios.hpp"
#include <iostream>
#include <fstream>
#include <random>

// 14 fractional bits + 2^5 scaling factor gives good results.

// #define T double
#define T fpm::fixed<std::int32_t, std::int64_t, 9>
#define SIGMA_LB 0.1
#define SIGMA_UB 5

// std::mt19937 gen(3);
std::default_random_engine gen;

// class Results{
//     public:
//     T mean;
//     T sigma;
//     T elbo;

//     Results(T mean, T sigma, T elbo){
//         this->mean = mean;
//         this->sigma = sigma;
//         this->elbo = elbo;
//     }
// };


template<typename Obs_ty, typename prior_T, typename guide_T, typename mu_learn_ty, typename std_learn_ty>
auto 
Infer(Obs_ty &N, prior_T& T_prior, guide_T& T_var,mu_learn_ty& mu_var, std_learn_ty& sigma_var, T learning_rate, int  NUM_SAMPLES, int NUM_ITERS = 1000)
{
    T diff_elbo_mean = (T) 0;
    T diff_elbo_sigma = (T) 0;
    T second_term_diff_elbo = (T) 0;
    // T learning_rate = (T) 0.01;
    T LB = (T) -128;
    T UB = (T) 128;
    T final_grad_mean = (T) 0;
    T final_grad_sigma = (T) 0;
    T total_elbo = (T) 0;
    T SCALING = 8;
    //[ARD:REMOVE_BEGIN]
    // std::ofstream result_file(argv[1]);
    // result_file << "iter,elbo,mean,sigma,mu_ref,grad" << std::endl;
    // std::cout << "Num iters: " << NUM_ITERS << std::endl;
    // std::cout << "Learning rate: " << learning_rate << std::endl;
    // all_type_info<T>();
    //[ARD:REMOVE_END]
    T max_elbo = std::numeric_limits<T>::lowest();
    T max_mean = (T) 0;
    T max_sigma = (T) 0;
    Results max_results(max_mean, max_sigma, max_elbo);

    for(int c_iter = 0;c_iter < NUM_ITERS; c_iter++)
    {
        diff_elbo_mean =0;
        diff_elbo_sigma =0;
        total_elbo =0;
        for(int c_sample = 0; c_sample < NUM_SAMPLES; c_sample++)
        {
            T_prior(T_var.sample()); // Add sampling contraints if it doesnt work

            second_term_diff_elbo = N.log_pdf().value()/SCALING + T_prior.log_pdf().value()/SCALING - T_var.log_pdf().value()/SCALING;
            diff_elbo_mean += T_var.diff_log(T_var.mu)*(second_term_diff_elbo);
            diff_elbo_sigma += T_var.diff_log(T_var.stddev)*(second_term_diff_elbo);
            total_elbo += second_term_diff_elbo;
        }
        final_grad_mean = grad_clipping<T>(diff_elbo_mean/NUM_SAMPLES, LB, UB);
        final_grad_sigma = grad_clipping<T>(diff_elbo_sigma/NUM_SAMPLES, LB, UB);
        
        if(total_elbo > max_elbo)
        {
            max_results.mean = mu_var.value();
            max_results.sigma = sigma_var.value();
            max_results.elbo = total_elbo;
        }

        //[ARD:REMOVE_BEGIN]
        // result_file << c_iter << "," << total_elbo/NUM_SAMPLES << "," << mu_var.value() << "," << sigma_var.value() << ",95"<<", "<< final_grad_mean<< std::endl;
        //[ARD:REMOVE_END]
        mu_var(mu_var.value() + learning_rate*final_grad_mean);
        sigma_var(grad_clipping<T>(sigma_var.value() + learning_rate*final_grad_sigma, SIGMA_LB, SIGMA_UB));
        // std::cout << "Iteration: " << c_iter<< "--- mu: " << mu_var.value() << ", sigma: " << sigma_var.value() << std::endl;
    }

    return max_results;
}

template < typename Tc = T, typename std::enable_if<std::is_base_of<fpm::fixedpoint_base, Tc>::value>::type* = nullptr> inline
Tc sample(Tc& mu_value, Tc& stddev_value)
{
    auto sign_and_int_bits = Tc::integer_bits;

    uint32_t random32_1 = gen();
    random32_1 = random32_1 << sign_and_int_bits;
    random32_1 = random32_1 >> sign_and_int_bits;

    uint32_t random32_2 = gen();
    random32_2 = random32_2 << sign_and_int_bits;
    random32_2 = random32_2 >> sign_and_int_bits;

    uint32_t random32_3 = gen();
    random32_3 = random32_3 << sign_and_int_bits;
    random32_3 = random32_3 >> sign_and_int_bits;

    auto sum = random32_1 + random32_2 + random32_3 ;
    sum = sum << 1;

    auto shifted_mean = Tc::from_raw_value(sum) - Tc(3);
    return (shifted_mean *stddev_value) + mu_value;
}


template <typename Tc = T, typename std::enable_if<std::is_arithmetic<Tc>::value>::type* = nullptr> inline
Tc sample(Tc& mu_value, Tc& stddev_value)
{
    std::normal_distribution<Tc> dist(mu_value, stddev_value);
    return dist(gen);
}


template<int N>
void gen_data(std::array<T,N> &xtrue, std::array<T, N> &yobs)
{
    // T init_x0 = 0;
    // T init_sigma = 2500;
    // T x0 = sample(init_x0, init_sigma); // this decides the "starting point" of the data
    T x0 = 0;
    T noise = 1;
    for(int i = 0; i < N; i++)
    {
        xtrue[i] = sample(x0, noise);
        yobs[i] = sample(xtrue[i], noise);
    }
}

template <int N>
void print_data(std::array<T, N> &xtrue, std::array<T, N> &yobs)
{
    for(int i = 0; i < N; i++)
    {
        std::cout << xtrue[i] << "," << yobs[i] << std::endl;
    }
}

int main(int argc, char ** argv)
{
    // Gaussian Prior
    // std::array<T, 50> true_positions;
    // for(int i = 0; i < 50; i++)
    // {
    //     true_positions[i] = 20 + i*3;
    // }
    // std::array<T, 50> gps_positions = {19.663621799623133, 22.701998379665135, 26.625344294898518, 31.676056240738973, 32.87644433923754, 32.757787338169564, 37.610137407443005, 40.891744002716344, 43.157647366491545, 46.078425584863666, 48.48542704151595, 52.9842109273574, 54.22367965835081, 58.465805119931, 61.057301818355526, 65.83448234831087, 68.5441793347521, 72.18062943896645, 73.42796142564858, 76.66570361676332, 79.63686112998352, 82.6500520101582, 87.88495011647473, 88.54735786174282, 93.36021465851054, 95.99197581644883, 99.06388655940603, 101.36347209608479, 103.69534745414822, 106.94266256028004, 109.32168867898059, 113.70094768740155, 116.43250602632058, 119.18094537249802, 120.11474334332311, 124.88536424721259, 127.91705024086177, 131.21093707928904, 134.26206157629002, 136.91683310056376, 140.41439757954075, 141.7408333154458, 145.55585512463207, 148.94844543842734, 151.27399220478426, 155.65847008275867, 159.85827570055625, 160.013841681458, 164.22236878918153, 168.5884802454716};
    const int NUM_DATA_POINTS = 100;
    // std::array<T, NUM_DATA_POINTS> true_positions{459.979, 460.169, 461.193, 461.077, 461.017, 459.9, 458.491, 461.907, 461.624, 459.654, 458.266, 459.731, 461.607, 461.263, 461.167, 459.833, 459.177, 460.603, 461.527, 460.948, 459.613, 459.02, 460.329, 459.387, 459.376, 459.225, 462.167, 461.053, 461.276, 460.308, 460.679, 461.324, 461.494, 460.002, 459.813, 461.453, 461.525, 458.033, 460.055, 459.403,
    //  460.301, 459.92, 461.673, 459.957, 459.15, 460.995, 461.721, 461.169, 460.812, 460.684, 460.672, 460.761, 459.897, 460.584, 459.6, 459.8, 460.585, 460.723, 460.563, 459.321, 460.879, 461.135, 458.986, 460.611, 460.222, 457.068, 460.148, 462.751, 462.022, 459.898, 460.394, 460.27, 458.567, 460.103, 458.101, 460.243, 461.057, 460.337, 459.654, 462.086, 460.374, 462.231, 459.982, 460.549, 460.657, 460.81, 460.141, 459.813, 459.964, 459.584, 460.725, 461.873, 460.658, 460.882, 459.959, 461.426, 460.294, 459.188, 460.474, 459.586};
    
    std::array<T, NUM_DATA_POINTS> yobs_positions;

    const int num_experiments = 100;

    std::array<float, num_experiments> rmse_all_experiments;
    std::array<float, num_experiments> geomean_all_experiments;


    // Generating observations according to probzelus bc
    for(int i = 0; i < NUM_DATA_POINTS; i++)
    {
        if (i == 0)
        {
            yobs_positions[0] = (T) 50;
        }
        else
        {
            yobs_positions[i] = yobs_positions[i-1] + (T) 0.1;
        }
    }

     std::cout << "rmse,geomean,num_iters,num_data_samples,num_obs_samples" <<std::endl;

    // // Initialising the prior
    T init_mu = 0;
    T init_sigma = 2500;

    for(int curr_exp=0; curr_exp <num_experiments; curr_exp++)
    {

    Constant<T> mu_T(0), sigma_T(1), data_T(0);
    auto T_prior = normal_dist<T>(mu_T, sigma_T,data_T);

    Constant<T> sigma_N(0.5), data_N(20);
    auto N_obs = normal_dist<T>(T_prior, sigma_N,data_N);

    // Varaitional assignment of T (the posterior)
    Variable<T> mu_var(0), sigma_var(1);
    auto T_var = normal_dist<T>(mu_var, sigma_var, data_T);

    // // Learning
    int NUM_SAMPLES = 4;
    int NUM_ITERS = 200;
    // // auto results = Infer(N, T_prior, T_var, mu_var, sigma_var, 0.01, NUM_SAMPLES, NUM_ITERS);

    std::array<T, NUM_DATA_POINTS> predicted_positions;
    T half = 0.5;

    for(int i = 0; i < yobs_positions.size(); i++)
    {
        data_N(yobs_positions[i]);
        // mu_T(mu_T.value() + time_step*velocity_control_with_noise[i]);
        auto results = Infer(N_obs, T_prior, T_var, mu_var, sigma_var, 0.01, NUM_SAMPLES, NUM_ITERS);

        auto noisy_obs = sample(results.mean, half);
        mu_T(noisy_obs);
        mu_var(noisy_obs);

        sigma_T(1);
        predicted_positions[i] = results.mean;
    }



     std::array<float, NUM_DATA_POINTS> predicted_positions_float;
    for(int i = 0; i < predicted_positions.size(); i++)
    {
        predicted_positions_float[i] = (float)predicted_positions[i];
    }

    std::array<float, NUM_DATA_POINTS> yobs_positions_float;
    for(int i = 0; i < yobs_positions.size(); i++)
    {
        yobs_positions_float[i] = (float)yobs_positions[i];
    }
    float rmse_i = rmse(yobs_positions_float, predicted_positions_float,10);
    float geomean_i = geomean(yobs_positions_float, predicted_positions_float,10);
    rmse_all_experiments[curr_exp] = rmse_i;
    geomean_all_experiments[curr_exp] = geomean_i;

    std::cout << rmse_i << "," << geomean_i << "," << NUM_ITERS << "," << NUM_DATA_POINTS << "," << NUM_SAMPLES << std::endl;
    // std::cout << "timestep,obs,pred\n";

    // for(int i = 0; i < predicted_positions_float.size(); i++)
    // {
    //     std::cout << i << "," << yobs_positions[i] << "," << predicted_positions_float[i] << std::endl;
    // }
    // // print rmse and geomean

    // std::cout << "rmse: " << rmse_i << "   geomean: " << geomean_i << std::endl;
    // break;
    }
    
}