

//[ARD:REMOVE_END]

#undef max
#undef min
#undef abs
#undef round

#include "distributions_latest.hpp"
#define T fpm::fixed<std::int32_t, std::int64_t, 21>
#define SIGMA_LB 0.1
#define SIGMA_UB 5
#define EARLY_STOPPING 5



void setup()
{

Serial.begin(9600); while(!Serial){;}
     randomSeed(analogRead(0));


    


     const int num_experiments = 50;
     std::array<float, num_experiments> means;
     std::array<float, num_experiments> sigmas;
     std::array<float, num_experiments> rmse_all_exp;
     std::array<float, 1> y_true{0.45};
     std::array<float, 1> y_pred;
     Serial.println("N_true,N,rmse,time_ms,num_iters,num_data_samples,num_obs_data");
     unsigned long total_time = 0;
    

     for(int i = 0; i < num_experiments; i++)
     {
         randomSeed(analogRead(0));
         T max_elbo = std::numeric_limits<T>::lowest();
         T max_mean = 0;
         T max_sigma = 0;
    
        Constant<T> lb(1), ub(100), p_sample(0);
    auto p_prior = uniform_dist<T>(lb, ub, p_sample);
    Constant<T> N(1), data(0.45);
    auto mu_gauss = T{N.value()} * p_prior/T{100};
    std::cout << "mu_gauss: " << mu_gauss.value() << std::endl;
    auto sigma_gauss = sqrt(T{N.value()} * (p_prior/T{100}) * (T{1} - (p_prior/T{100})));
    std::cout << "sigma_gauss: " << sigma_gauss.value() << std::endl;
    auto succ = normal_dist<T>(mu_gauss, sigma_gauss, data);
    // auto succ = binomial_dist<T>(N, p_prior, data);

    Variable<T> mu_p(50), sigma_p(1);
    auto p = normal_dist<T>(mu_p, sigma_p, p_sample);

        const int NUM_SAMPLES = 4;
        constexpr int BITSHIFT = std::log2(NUM_SAMPLES);
        const int NUM_ITERS = 1500;
        T NUM_SAMPLES_T = (T) NUM_SAMPLES;
        T learning_rate = (T) 0.01;
        T SCALING = 2;
        T sigma_lb = (T) SIGMA_LB;
        T diff_elbo_mean = (T) 0;
        T diff_elbo_sigma = (T) 0;
        T second_term_diff_elbo = (T) 0;
        T LB = (T) -128;
        T UB = (T) 128;
        T final_grad_mean = (T) 0;
        T final_grad_sigma = (T) 0;
        T total_elbo = (T) 0;
        T zero_p_epsilon = (T) 0.01;
        T zero = (T) 0;
        T sigma_ub = (T) 0.3;
        T one_m_epsilon = (T) 0.99;
        T convergence = (T) 0.01;
        std::array<T, EARLY_STOPPING> loss_arr = {0,0,0,0,0};
        T curr_sample;

        //[ARD:REMOVE_END]

unsigned long start_time = millis();
        for(int c_iter = 0;c_iter < NUM_ITERS; c_iter ++)
        {

            total_elbo = zero;
            diff_elbo_mean = zero;
            diff_elbo_sigma = zero;

            for(int c_sample = 0; c_sample < NUM_SAMPLES; c_sample ++)
            {
                do{
                    curr_sample = p.sample();
                }while ( curr_sample < lb.value() || curr_sample > ub.value() );

                p_sample(curr_sample);
                second_term_diff_elbo = succ.score()/SCALING + p_prior.score()/SCALING - p.log_pdf().value()/SCALING;
                total_elbo += second_term_diff_elbo;
                diff_elbo_mean += p.diff_log(mu_p)*(second_term_diff_elbo);

            }
            final_grad_mean = grad_clipping<T>(diff_elbo_mean/NUM_SAMPLES, LB, UB);
            final_grad_sigma = grad_clipping<T>(diff_elbo_sigma/NUM_SAMPLES, LB, UB);

            //[ARD:REMOVE_END]

             if(total_elbo/NUM_SAMPLES_T > max_elbo)
            {
                 max_elbo = total_elbo/NUM_SAMPLES_T;
                 max_mean = mu_p.value()/T{100};
                 max_sigma = sigma_p.value();
            }

            mu_p(grad_clipping<T>(mu_p.value() + learning_rate*final_grad_mean, lb.value(), ub.value()));
            sigma_p(grad_clipping<T>(sigma_p.value() + learning_rate*final_grad_sigma, sigma_lb, SIGMA_UB));

        }
unsigned long end_time = millis();                                     unsigned long curr_total_time = end_time - start_time;


     means[i] = (float) max_mean;
     sigmas[i] = (float) max_sigma;
     total_time+=curr_total_time;
     y_pred[0] = (float) max_mean;
     float rmse_current = rmse(y_pred, y_true);
     rmse_all_exp[i] = rmse_current;
     Serial.println("0.45," + String((float)max_mean,5) + "," + String((float)rmse_current,5) +"," + String((float)curr_total_time) +"," + String((float)NUM_ITERS) +"," + String((float)NUM_SAMPLES) +"," + String((float)1));


    }
     Serial.println("Average Mean: " + String((float)mean_vec(means)));
     Serial.println("Average Sigma: " + String((float)mean_vec(sigmas)));
     Serial.println("Average Total time: " + String((float)total_time/num_experiments));
     Serial.println("Average RMSE: " + String((float)mean_vec(rmse_all_exp)));
}


void loop(){}
