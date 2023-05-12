//[ARD:REMOVE_BEGIN]
#include "../../include/distributions_latest.hpp"
#include "../../include/ios.hpp"
#include <iostream>
#include <fstream>
#include <random>
//[ARD:REMOVE_END]

//[ARD:UNDEFS]

//[ARD:ADD_HEADER]



#define T [PY:TYPE]
#define SIGMA_LB 0.1
#define SIGMA_UB 5


int main(int argc, char ** argv)//[ARD:SETUP_DEF]
{   


    //[ARD:SERIAL_MONITOR]
    //[ARD:UNCOMMENT_BEGIN]
    // randomSeed(analogRead(0));
    //[ARD:UNCOMMENT_END]


    //[ARD:UNCOMMENT_BEGIN]
    // const int num_experiments = [PY:NUM_EXPERIMENTS];
    // std::array<float, num_experiments> means;
    // std::array<float, num_experiments> sigmas;
    // std::array<float, num_experiments> rmse_all_exp;
    // std::array<float, 1> y_true{90.3};
    // std::array<float, 1> y_pred;
    // Serial.println("N_true,N,rmse,time_ms,num_iters,num_data_samples,num_obs_data");
    // unsigned long total_time = 0;
    

    // for(int i = 0; i < num_experiments; i++)
    // {
        // randomSeed(analogRead(0));
        // T max_elbo = std::numeric_limits<T>::lowest();
        // T max_mean = 0;
        // T max_sigma = 0;
    //[ARD:UNCOMMENT_END]

    // Gaussian Prior
    Constant<T> mu_T(70), sigma_T(5), data_T(0);
    auto T_prior = normal_dist<T>(mu_T, sigma_T,data_T);

    Constant<T> sigma_N(2), data_N(100);
    auto N = normal_dist<T>(T_prior, sigma_N,data_N);

    // Varaitional assignment of T (the posterior)
    Variable<T> mu_var(70), sigma_var(1);
    auto T_var = normal_dist<T>(mu_var, sigma_var, data_T);

    // Learning
    int NUM_SAMPLES = [PY:NUM_SAMPLES];
    int NUM_ITERS = [PY:NUM_ITERS];
    T diff_elbo_mean = (T) 0;
    T diff_elbo_sigma = (T) 0;
    T second_term_diff_elbo = (T) 0;
    T learning_rate = (T) [PY:LEARNING_RATE];
    T LB = (T) -128;
    T UB = (T) 128;
    T final_grad_mean = (T) 0;
    T final_grad_sigma = (T) 0;
    T total_elbo = (T) 0;

    //[ARD:REMOVE_BEGIN]
    std::ofstream result_file(argv[1]);
    result_file << "iter,elbo,mean,sigma,mu_ref,grad" << std::endl;
    std::cout << "Num iters: " << NUM_ITERS << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    all_type_info<T>();
    //[ARD:REMOVE_END]


    //[ARD:START_TIME]
    for(int c_iter = 0;c_iter < NUM_ITERS; c_iter ++)
    {
        diff_elbo_mean =0;
        diff_elbo_sigma =0;
        total_elbo =0;
        for(int c_sample = 0; c_sample < NUM_SAMPLES; c_sample ++)
        {
            data_T(T_var.sample());
            second_term_diff_elbo = N.log_pdf().value() + T_prior.log_pdf().value() - T_var.log_pdf().value();
            diff_elbo_mean += T_var.diff_log(mu_var)*(second_term_diff_elbo);
            diff_elbo_sigma += T_var.diff_log(sigma_var)*(second_term_diff_elbo);
            total_elbo += second_term_diff_elbo;
        }
        final_grad_mean = grad_clipping<T>(diff_elbo_mean/NUM_SAMPLES, LB, UB);
        final_grad_sigma = grad_clipping<T>(diff_elbo_sigma/NUM_SAMPLES, LB, UB);
        //[ARD:REMOVE_BEGIN]
        result_file << c_iter << "," << total_elbo/NUM_SAMPLES << "," << mu_var.value() << "," << sigma_var.value() << ",80"<<", "<< final_grad_mean<< std::endl;
        //[ARD:REMOVE_END]
        mu_var(mu_var.value() + learning_rate*final_grad_mean);
        sigma_var(grad_clipping<T>(sigma_var.value() + learning_rate*final_grad_sigma, SIGMA_LB, SIGMA_UB));
       
       //[ARD:UNCOMMENT_BEGIN]
            // if(total_elbo/NUM_SAMPLES > max_elbo)
            //{
                // max_elbo = total_elbo/NUM_SAMPLES;
                // max_mean = mu_var.value();
                // max_sigma = sigma_var.value();
            //}
            //[ARD:UNCOMMENT_END]
    }

    //[ARD:END_TIME]


    //[ARD:UNCOMMENT_BEGIN]
    // means[i] = (float) max_mean;
    // sigmas[i] = (float) max_sigma;
    // total_time+=curr_total_time;
    // y_pred[0] = (float) max_mean;
    // float rmse_current = rmse(y_pred, y_true);
    // rmse_all_exp[i] = rmse_current;
    // Serial.println("90.3," + String((float)max_mean,5) + "," + String((float)rmse_current,5) +"," + String((float)curr_total_time) +"," + String((float)NUM_ITERS) +"," + String((float)NUM_SAMPLES) +"," + String((float)1));
    //[ARD:UNCOMMENT_END]


    //[ARD:UNCOMMENT_BEGIN]
    //}
    // Serial.println("Average Mean: " + String((float)mean_vec(means)));
    // Serial.println("Average Sigma: " + String((float)mean_vec(sigmas)));
    // Serial.println("Average Total time: " + String((float)total_time/num_experiments));
    // Serial.println("Average RMSE: " + String((float)mean_vec(rmse_all_exp)));
    //[ARD:UNCOMMENT_END]
}


//[ARD:ADD_LOOP]