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



int main(int argc, char **argv) //[ARD:SETUP_DEF]
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
    // std::array<float, 1> y_true{26.5};
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


    // std::array<T,3> data_arr = {8.914, 7.0088, 9.339};
    std::array<T,3> data_arr = {[PY:DATA]};
    Constant<T> lb(5), ub(50), plankton_sample(0);
    auto plankton_param = uniform_dist<T>(lb, ub, plankton_sample);

    Constant<T> N(100), data(0);
    auto mu_plankton = T{0.5} * plankton_param;
    auto sigma_plankton = sqrt(T{0.25} * plankton_param);

    auto plankton = normal_dist<T>(mu_plankton, sigma_plankton, data);

    // Variational assignment of plankton
    Variable<T> mu_plankton_var(27.5), sigma_plankton_var(1);
    auto plankton_var = normal_dist<T>(mu_plankton_var, sigma_plankton_var, plankton_sample);

    const int NUM_SAMPLES = [PY:NUM_SAMPLES];
    T NUM_SAMPLES_T = (T) NUM_SAMPLES;
    int NUM_ITERS = [PY:NUM_ITERS];
    T learning_rate = (T) [PY:LEARNING_RATE];
    T sigma_lb = (T) 0.1;
    T sigma_ub = (T) SIGMA_UB;
    T diff_elbo_mean = (T) 0;
    T diff_elbo_sigma = (T) 0;
    T LB = (T) -128;
    T UB = (T) 128;
    T final_grad_mean = (T) 0;
    T final_grad_sigma = (T) 0;
    T total_elbo = (T) 0;
    T second_term_diff_elbo = (T) 0;
    T curr_sample = (T) 0;
    T zero_p_epsilon = (T) 0.01;
    T zero = (T) 0;
    T one_m_epsilon = (T) 0.99;
    T plankton_score = (T) 0;
    //[ARD:REMOVE_BEGIN]
    std::ofstream result_file(argv[1]);
    std::cout << "Num iters: " << NUM_ITERS << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    all_type_info<T>();
    result_file << "iter,elbo,mean,sigma,grad" << std::endl;
    //[ARD:REMOVE_END]


    //[ARD:START_TIME]
    for(int c_iter = 0; c_iter < NUM_ITERS; c_iter++)
    {
        total_elbo = (T) 0;
        diff_elbo_mean = (T) 0;
        diff_elbo_sigma = (T) 0;
        for(int c_sample = 0; c_sample < NUM_SAMPLES; c_sample++)
        {
            // plankton_sample(plankton_param.sample());
            do{
                curr_sample = plankton_var.sample();
            } while ( curr_sample < lb.value() || curr_sample > ub.value());

            plankton_sample(curr_sample);
            plankton_score = total_score_w_data(plankton, data_arr);

            second_term_diff_elbo = plankton_score + plankton_param.score() - plankton_var.score();
            diff_elbo_mean += second_term_diff_elbo * plankton_var.diff_log(mu_plankton_var);
            diff_elbo_sigma += second_term_diff_elbo * plankton_var.diff_log(sigma_plankton_var);
            total_elbo += second_term_diff_elbo;
            // total_elbo += plankton_var.log_pdf().value();
        }
        final_grad_mean = grad_clipping<T>(diff_elbo_mean / NUM_SAMPLES_T, LB, UB);
        final_grad_sigma = grad_clipping<T>(diff_elbo_sigma / NUM_SAMPLES_T, LB, UB);
        //[ARD:REMOVE_BEGIN]
        result_file << c_iter<<", " << total_elbo/NUM_SAMPLES << ", " << mu_plankton_var.value() << ", " << sigma_plankton_var.value() << ", "<< final_grad_mean<< std::endl;
        //[ARD:REMOVE_END]

        //[ARD:UNCOMMENT_BEGIN]
            // if(total_elbo/NUM_SAMPLES > max_elbo)
            //{
                // max_elbo = total_elbo/NUM_SAMPLES;
                // max_mean = mu_plankton_var.value();
                // max_sigma = sigma_plankton_var.value();
            //}
            //[ARD:UNCOMMENT_END]
        mu_plankton_var(grad_clipping<T>(mu_plankton_var.value() + learning_rate * final_grad_mean, lb.value(), ub.value()));
        sigma_plankton_var(grad_clipping<T>(sigma_plankton_var.value() + learning_rate * final_grad_sigma,sigma_lb, sigma_ub));

    }

    //[ARD:END_TIME]


    //[ARD:UNCOMMENT_BEGIN]
    // means[i] = max_mean;
    // sigmas[i] = max_sigma;
    // total_time+=curr_total_time;
    // y_pred[0] = max_mean;
    // float rmse_current = rmse(y_pred, y_true);
    // rmse_all_exp[i] = rmse_current;
    // Serial.println("26.5," + String((float)max_mean) + "," + String((float)rmse_current) +"," + String((float)curr_total_time) +"," + String((float)NUM_ITERS) +"," + String((float)NUM_SAMPLES) +"," + String((float)3));
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