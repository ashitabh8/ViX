//[ARD:REMOVE_END]

#undef max
#undef min
#undef abs
#undef round

#include "distributions_latest.hpp"

// using fixed_10_6 = fpm::fixed<std::int16_t, std::int32_t, 6>;

#define T double
#define SIGMA_LB 0.1
#define SIGMA_UB 5
#define BITSHIFT 8
#define EARLY_STOPPING 5


template<int NUM_DATA_SAMPLES>
auto gen_data(float range = 5)
{
     float SIGMA = 0.5;
     float M = 10;
     float B = 27;
     std::default_random_engine gen(random(1000));
    // const unsigned NUM_DATA_SAMPLES = 100;

    //[ARD:REMOVE_END]
    // const unsigned NUM_DATA_SAMPLES = 100;

    std::array<float, NUM_DATA_SAMPLES> y_samples_temp;
    std::array<float, NUM_DATA_SAMPLES> x_samples_temp = linspace<float, NUM_DATA_SAMPLES>(-range,range);
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

float get_true_y(float x)
{
    float a0 = 10;
    float a1 = 27;
    return a0 * x + a1;
}

float get_true_y(float x, float a0, float a1)
{
    return a0 * x * x +a1 * x;
}


void setup()
{
Serial.begin(9600); while(!Serial){;}
     randomSeed(analogRead(0));


     const int num_experiments = 2;
     std::array<float, num_experiments> a0_mus;
     std::array<float, num_experiments> a0_sigmas;
     std::array<float, num_experiments> a1_mus;
     std::array<float, num_experiments> a1_sigmas;
     std::array<float, num_experiments> rmse_all_exp;
     Serial.println("a0_true,a1_true,a0,a1,rmse,time_ms,num_iters,num_data_samples,num_obs_data");
     unsigned long total_time = 0;



    const int NUM_DATA_SAMPLES = 50;

    auto data_pair = gen_data<NUM_DATA_SAMPLES>();
    auto temp = gen_data<50>(2);
    std::array<float, 50> x_test_set;
    for(int i = 0; i < 50; i++)
    {
        x_test_set[i] = (float) temp.first[i];
    }
    std::array<float, 50> y_test_set;
    for(int k = 0; k < x_test_set.size(); k++)
    {
        y_test_set[k] = get_true_y(x_test_set[k]);
    }

    auto x_test = data_pair.first;
    auto y_test = data_pair.second;


     for(int i = 0; i < num_experiments; i++)
     {
     
         randomSeed(analogRead(0));
         T max_elbo = std::numeric_limits<T>::lowest();
         T best_a0_mu = 0;
         T best_a0_sigma = 0;
         T best_a1_mu = 0;
         T best_a1_sigma = 0;

    //  Variational assignment of coefficent: a0
    Variable<T> a0_mean(15), a0_sigma(1);
    Constant<T> a0_sample(0);
    auto a0 = normal_dist<T>(a0_mean, a0_sigma,a0_sample);

    // Variational assignment of coefficent: a1
    Variable<T> a1_mean(25), a1_sigma(1);
    Constant<T> a1_sample(0);
    auto a1 = normal_dist<T>(a1_mean, a1_sigma,a1_sample);

    // input data: x
    Constant<T> X(0);

    // Linear regression BN prior
    Constant<T> lb_a0(0), ub_a0(30);
    auto a0_prior = uniform_dist<T>(lb_a0, ub_a0, a0_sample);

    Constant<T> lb_a1(0), ub_a1(50);
    auto a1_prior = uniform_dist<T>(lb_a1, ub_a1, a1_sample);

    // Constant<T> a1_true(6);
    Constant<T> sigma(0.5);
    Constant<T> Y_container(0);
    auto LinReg_eq = a0 * X  + a1;
    auto Y = normal_dist<T>(LinReg_eq, sigma,Y_container);

    const int NUM_SAMPLES = 4;
    constexpr int BITSHIFT_SAMPLES = std::log2(NUM_SAMPLES);
    constexpr T SCALING = (T) std::pow(2, BITSHIFT);
    int NUM_ITERS = 1500;
    T learning_rate = (T) 0.01;

    //[ARD:REMOVE_END]

    T LB = (T) -128;
    T UB = (T) 128;
    T current_del_elbo_mu_a0 = (T) 0;
    T current_del_elbo_sigma_a0 = (T) 0;
    T current_del_elbo_mu_a1 = (T) 0;
    T current_del_elbo_sigma_a1 = (T) 0;
    T zero = (T) 0;
    T one = (T) 1;
    T current_score = (T) 0;
    T total_elbo  = (T) 0;
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
    
    std::array<T, EARLY_STOPPING> loss_arr = {0,0,0,0,0};
unsigned long start_time = millis();
    for(int c_iter = 0; c_iter < NUM_ITERS; c_iter++)
    {
        // std::cout << "Iteration: " << c_iter << std::endl;
        current_del_elbo_mu_a0 = zero;
        current_del_elbo_sigma_a0 = zero;
        current_del_elbo_mu_a1 = zero;
        current_del_elbo_sigma_a1 = zero;

        
        total_elbo = zero;
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            while(true)
            {
                curr_sample_a0 = a0.sample();
                if(curr_sample_a0 >= a0_prior.lb_value() && curr_sample_a0 <= a0_prior.ub_value())
                {
                    a0_sample(curr_sample_a0);
                    break;
                }
            }

            while(true)
            {
                curr_sample_a1 = a1.sample();
                // curr_sample_a1 = a1.sample();
                if(curr_sample_a1 >= a1_prior.lb_value() && curr_sample_a1 <= a1_prior.ub_value())
                {
                    a1_sample(curr_sample_a1);
                    break;
                }
            }
            
            current_score = (T) 0;

            for(int data_k = 0; data_k < NUM_DATA_SAMPLES; data_k++)
            {
                X(x_test[data_k]);
                Y_container(y_test[data_k]);

                    current_score += Y.log_pdf().value();
            }

            second_term = current_score + (a0_prior.log_pdf().value() + a1_prior.log_pdf().value() - a0.log_pdf().value() - a1.log_pdf().value());
            total_elbo += second_term;
            current_del_elbo_mu_a0 += a0.diff_log(a0_mean)*(second_term);
            current_del_elbo_sigma_a0 += a0.diff_log(a0_sigma)*(second_term);
            current_del_elbo_mu_a1 += a1.diff_log(a1_mean)*(second_term);
            current_del_elbo_sigma_a1 += a1.diff_log(a1_sigma)*(second_term);
        }

        step_a0_mu = grad_clipping<T>(current_del_elbo_mu_a0/NUM_SAMPLES, LB, UB); // Remove these extra variables when doing performance testing
        step_a0_sigma =grad_clipping<T>(current_del_elbo_sigma_a0/NUM_SAMPLES, LB, UB);
        
        step_a1_mu = grad_clipping<T>(current_del_elbo_mu_a1/NUM_SAMPLES, LB, UB); // Remove these extra variables when doing performance testing
        step_a1_sigma =grad_clipping<T>(current_del_elbo_sigma_a1/NUM_SAMPLES, LB, UB);
        //[ARD:REMOVE_END]
        new_value_a0_mu = a0_mean.value() + learning_rate * (step_a0_mu);
        new_value_a0_sigma = a0_sigma.value() + learning_rate * (step_a0_sigma);
        new_value_a1_mu = a1_mean.value() + learning_rate * (step_a1_mu);
        new_value_a1_sigma = a1_sigma.value() + learning_rate * (step_a1_sigma);

             if(total_elbo/NUM_SAMPLES > max_elbo)
            {
                 max_elbo = total_elbo/NUM_SAMPLES;
                 best_a0_mu = a0_mean.value();
                 best_a0_sigma = a0_sigma.value();
                 best_a1_mu = a1_mean.value();
                 best_a1_sigma = a1_sigma.value();
            }


        a0_mean(grad_clipping<T>(new_value_a0_mu, a0_prior.lb_value(), a0_prior.ub_value()));
        a0_sigma(grad_clipping<T>(new_value_a0_sigma, SIGMA_LB, SIGMA_UB));
        a1_mean(grad_clipping<T>(new_value_a1_mu, a1_prior.lb_value(), a1_prior.ub_value()));
        a1_sigma(grad_clipping<T>(new_value_a1_sigma, SIGMA_LB, SIGMA_UB));
    }

unsigned long end_time = millis();                                     unsigned long curr_total_time = end_time - start_time;


     a0_mus[i] = best_a0_mu;
     a0_sigmas[i] = best_a0_sigma;
     a1_mus[i] = best_a1_mu;
     a1_sigmas[i] = best_a1_sigma;
     total_time+=curr_total_time;

     std::array<float, x_test_set.size()> y_preds;
     for(int j = 0; j < x_test_set.size(); j++)
     {
         y_preds[j] = (float) (a0_mus[i] * x_test_set[j]  + a1_mus[i]);
     }
     float rmse_i = rmse(y_preds, y_test_set);
     rmse_all_exp[i] = rmse_i;
     Serial.println("10,27,"+String((float)a0_mus[i],5)+","+String((float)a1_mus[i],5) + "," + String((float)rmse_i,5) + "," + String((float)curr_total_time) + "," +String(NUM_ITERS) + "," + String(NUM_SAMPLES) + "," + String(NUM_DATA_SAMPLES));


    }
     Serial.println("Average a0 Mean: " + String((float)mean_vec(a0_mus)));
     Serial.println("Average a0 Sigma: " + String((float)mean_vec(a0_sigmas)));
     Serial.println("Average a1 Mean: " + String((float)mean_vec(a1_mus)));
     Serial.println("Average a1 Sigma: " + String((float)mean_vec(a1_sigmas)));
     Serial.println("Average Total time: " + String((float)total_time/num_experiments));
     Serial.println("Average RMSE: " + String((float)mean_vec(rmse_all_exp)));
}

void loop(){}
