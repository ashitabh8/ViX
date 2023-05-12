// Discrete DARE infinite horizon 

#undef abs
#undef max
#undef min
#undef round

#include <random>
// #include <iostream>
// #include<fstream>
#include <array>



//#include "../../include/ios.hpp"
// // #include "../../include/fixed.hpp"
// // #include "../../include/math.hpp"

#include "distributions_latest.hpp"


//#define T fpm::fixed<std::int32_t, std::int64_t, 11>

//#define T fpm::fixed<std::int16_t, std::int32_t, 5>

//#define T float
#define T double
#define SIGMA_LB 0.1
#define SIGMA_UB 5
std::mt19937 gen(0);

void transpose(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &AT)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            AT[i][j] = A[j][i];
        }
    }
}

T determinant(std::array<std::array<T, 3>, 3> &A)
{
    T det = 0;
    for(int i = 0; i < 3; i++)
    {
        det += (A[0][i] * (A[1][(i+1)%3] * A[2][(i+2)%3] - A[1][(i+2)%3] * A[2][(i+1)%3]));
    }
    return det;
}

void inverse(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &A_inv)
{
    T det = determinant(A);
    std::array<std::array<T, 3>, 3> A_adj;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            A_adj[i][j] = (A[(i+1)%3][(j+1)%3] * A[(i+2)%3][(j+2)%3] - A[(i+1)%3][(j+2)%3] * A[(i+2)%3][(j+1)%3]);
        }
    }
    // std::cout << "det: " << det << std::endl;
    std::array<std::array<T, 3>, 3> AT;
    transpose(A_adj, AT);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            A_inv[i][j] = AT[i][j] / det;
        }
    }
}

// Add matrices

void add(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B, std::array<std::array<T, 3>, 3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Multiply matrices

void multiply(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B, std::array<std::array<T, 3>, 3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = 0;
            for(int k = 0; k < 3; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Subtract matrices

void subtract(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B, std::array<std::array<T, 3>, 3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Print matrix
void print(std::array<std::array<T, 3>, 3> &A)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            // std::cout << A[i][j] << " ";
            Serial.print((float)A[i][j]);
            Serial.print(" ");
        }
        Serial.println();
        // std::cout << std::endl;
    }
}


void print(std::array<T, 3> &A)
{
    for(int i = 0; i < 3; i++)
    {
        // std::cout << A[i] << " ";
        Serial.print((float) A[i]);
    }
    // std::cout << std::endl;
    Serial.println();
}

// copy elements of matrix A to matrix B
void copy(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            B[i][j] = A[i][j];
        }
    }
}

// Matrix vector multiplication
void multiply(std::array<std::array<T, 3>, 3> &A, std::array<T, 3> &x, std::array<T, 3> &y)
{
    for(int i = 0; i < 3; i++)
    {
        y[i] = 0;
        for(int j = 0; j < 3; j++)
        {
            y[i] += A[i][j] * x[j];
        }
    }
}


// Infinite horizon discrete DARE
void DARE(std::array<std::array<T,3>,3> &A, std::array<std::array<T,3>,3> &B, std::array<std::array<T,3>,3> &Q, 
        std::array<std::array<T,3>,3> &R, std::array<std::array<T,3>,3> &Pk, 
        std::array<std::array<T,3>,3> &Pk_1)
{
    std::array<std::array<T,3>,3> AT, BT, AT_Pk, AT_Pk_A, Q_add_At_P_A,  AT_Pk_B,BT_Pk, BT_Pk_B, R_add_Bt_P_B, R_add_Bt_P_B_inv, BT_Pk_A,
     R_add_Bt_P_B_inv_mul_BT_Pk_A, AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A,
     At_Pk_A_sub_AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A;

    transpose(A, AT);
    transpose(B, BT);
    multiply(AT, Pk, AT_Pk);
    multiply(AT_Pk, A, AT_Pk_A);
    add(Q, AT_Pk_A, Q_add_At_P_A);
    multiply(AT_Pk, B, AT_Pk_B);
    multiply(BT, Pk, BT_Pk);
    multiply(BT_Pk, B, BT_Pk_B);
    
    add(R, BT_Pk_B, R_add_Bt_P_B);
    inverse(R_add_Bt_P_B, R_add_Bt_P_B_inv);
    multiply(BT_Pk, A, BT_Pk_A);
    multiply(R_add_Bt_P_B_inv, BT_Pk_A, R_add_Bt_P_B_inv_mul_BT_Pk_A);
    multiply(AT_Pk_B, R_add_Bt_P_B_inv_mul_BT_Pk_A, AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A);
    subtract(AT_Pk_A, AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A, At_Pk_A_sub_AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A);
    add(Q, At_Pk_A_sub_AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A, Pk_1);

}

// multiply scalar and matrix

void scalar_mat_mul(T scalar, std::array<std::array<T,3>,3> &A, std::array<std::array<T,3>,3> &B)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            B[i][j] = scalar * A[i][j];
        }
    }
}

// add two vectors

void add(std::array<T,3> &A, std::array<T,3> &B, std::array<T,3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        C[i] = A[i] + B[i];
    }
}


void LQR(std::array<std::array<T,3>,3> &A, std::array<std::array<T,3>,3> &B, std::array<std::array<T,3>,3> &Q, 
        std::array<std::array<T,3>,3> &R, std::array<std::array<T,3>,3> &Pk, 
        std::array<std::array<T,3>,3> &Pk_1, std::array<std::array<T,3>,3> &N, std::array<T,3> &X,
         std::array<std::array<T,3>,3> &K, std::array<T,3> &U, int steps = 1, bool init = true)
{
    if (init)
    {
        copy(Q, Pk);
    }
    for(int i = 0; i < steps; i++)
    {
        DARE(A, B, Q, R, Pk, Pk_1);
        copy(Pk_1, Pk);
    }
    copy(Pk, Pk_1);
    // DARE(A, B, Q, R, Pk, Pk_1);
    std::array<std::array<T,3>,3> Nt, Bt, Bt_Pk_1, Bt_Pk_1_B, R_add_Bt_Pk_1_B, R_add_Bt_Pk_1_B_inv, Bt_Pk_1_A, N_add_Bt_Pk_1_A, F;
    transpose(B, Bt);
    transpose(N, Nt);
    multiply(Bt, Pk_1, Bt_Pk_1);
    multiply(Bt_Pk_1, B, Bt_Pk_1_B);
    add(R, Bt_Pk_1_B, R_add_Bt_Pk_1_B);
    inverse(R_add_Bt_Pk_1_B, R_add_Bt_Pk_1_B_inv);
    multiply(Bt_Pk_1, A, Bt_Pk_1_A);
    add(Nt, Bt_Pk_1_A, N_add_Bt_Pk_1_A);
    multiply(R_add_Bt_Pk_1_B_inv, N_add_Bt_Pk_1_A, F);
    scalar_mat_mul(-1, F, K);
    multiply(K, X, U);

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

// Simulating the Environment - only pass the means of the distributions
void env(std::array<T,3> &X, std::array<T,3> &U, std::array<T,3> &X_true, std::array<std::array<T,3>,3> &A, 
        std::array<std::array<T,3>,3> &B, std::array<std::array<T,3>,3> &U_noise,T& acc_obs,T& gps_position_obs, bool call_gps = false, bool init = false)
{
    std::array<T,3> Ax, Bu, X_mu;
    if(init)
    {
        X_mu[0] = X[0];
        X_mu[1] = X[1];
        X_mu[2] = X[2];
    }
    else
    {
        
        multiply(A, X, Ax);
        multiply(B, U, Bu);
        add(Ax, Bu, X_mu);
    }
    

    // std::cout << "X_mu: " << X_mu[0] << " " << X_mu[1] << " " << X_mu[2] << std::endl;

    // Adding noise to the control input - U_noise is the covariance matrix, 
    // Since its a diagnol matrix, we can just add the noise to the diagonal elements
    
    T X_position = sample(X_mu[0], U_noise[0][0]);
    T X_velocity = sample(X_mu[1], U_noise[1][1]);
    T X_acceleration = sample(X_mu[2], U_noise[2][2]);
    T noise_variance = (T) 0.1;
    acc_obs = sample(X_acceleration, noise_variance);
    if(call_gps)
    {
        gps_position_obs = sample(X[0], noise_variance);
    }

    X_true[0] = X_position;
    X_true[1] = X_velocity;
    X_true[2] = X_acceleration;
}


// Make sure to only pass mean of the distribution in X
void perform_LTI(std::array<std::array<T,3>,3> &A, std::array<std::array<T, 3>, 3> &B, std::array<T,3> &X, 
                std::array<T,3> &U, std::array<T,3> &X_mu)
{
    std::array<T,3> Ax, Bu;
    multiply(A, X, Ax);
    multiply(B, U, Bu);
    add(Ax, Bu, X_mu);
}

// copy vector to vector

void copy(std::array<T,3> &X, std::array<T,3> &X_next)
{
    for(int i = 0; i < 3; i++)
    {
        X_next[i] = X[i];
    }
}



template<typename Obs_ty, typename prior_T, typename guide_T, typename mu_learn_ty, typename std_learn_ty>
auto Infer(Obs_ty &N, prior_T& T_prior, guide_T& T_var,mu_learn_ty& mu_var, std_learn_ty& sigma_var, T learning_rate, int  NUM_SAMPLES, int NUM_ITERS = 1000)
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
            T_prior(T_var.sample());
            second_term_diff_elbo = N.log_pdf().value() + T_prior.log_pdf().value() - T_var.log_pdf().value();
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

void setup()
{
    Serial.begin(9600);
    while (!Serial) {
      ; // wait for serial port to connect. Needed for native USB port only
    }
    
    std::array<std::array<T,3>,3> A{{{1,0.1,0},{0,1,0.1},{0,0,0}}};

    std::array<std::array<T,3>,3> A_approx{{{1,0.1,0},{0,1,0.1},{0,0,0.0001}}};

    std::array<std::array<T,3>,3> B{{{1,0,0},{0,1,0},{0,0,1}}};

    std::array<std::array<T,3>,3> Q{{{1,0,0},{0,0.1,0},{0,0,0.1}}};

    std::array<std::array<T,3>,3> R{{{100,0,0},{0,100,0},{0,0,1}}}; // changed from the original.

    std::array<std::array<T,3>,3> N{{{0,0,0},{0,0,0},{0,0,0}}};

    std::array<T,3> x_init{100, 0, 0.5};
    // std::array<T,3> u{0,0,0};
    std::array<T,3> x_next{0,0,0};

    std::array<std::array<T,3>,3> noise_covariance{{{0.1,0,0},{0,0.1,0},{0,0,0.1}}};

    std::array<std::array<T,3>,3> Pk, Pk_1;
    const int time_steps = 50;
    std::array<std::array<T,3>, time_steps> X_true;
    std::array<T, 3> X_predicted_prior; // X predicted via the dynamic system

    // Probabilistic program - Since the covariance matrix is a diagnal matrix the my_gaussian can be
    // represented as individual univariate gaussian distributions

    // Observed Position
    Constant<T> X_pos_prior_mu(x_init[0]), noise_pos_prior(0.1), X_position_sample(0);
    auto X_pos_prior = normal_dist<T>(X_pos_prior_mu, noise_pos_prior, X_position_sample);

    Constant<T> noise_position(0.01), X_position_data(0);
    auto X_pos_obs = normal_dist<T>(X_pos_prior,noise_position, X_position_data);

    // Variational guide of X_pos_prior
    Variable<T> X_pos_prior_mu_guide(x_init[0]), noise_pos_prior_guide(1);
    auto X_pos_guide = normal_dist<T>(X_pos_prior_mu_guide, noise_pos_prior_guide, X_position_sample);

    // Observed Acceleration
    Constant<T> X_acc_prior_mu(x_init[2]), noise_acc_prior(0.1), X_accelerations_sample(0);
    auto X_acc_prior = normal_dist<T>(X_acc_prior_mu, noise_acc_prior, X_accelerations_sample);

    Constant<T> noise_acceleration(0.1), X_acceleration_data(0);
    auto X_acc_obs = normal_dist<T>(X_acc_prior,noise_acceleration, X_acceleration_data);

    // Variational guide of X_acc_prior
    Variable<T> X_acc_prior_mu_guide(x_init[2]), noise_acc_prior_guide(1);
    auto X_acc_guide = normal_dist<T>(X_acc_prior_mu_guide, noise_acc_prior_guide, X_accelerations_sample);

    // Velocty - no variational guide no inference on this parameter as it is part of control
    Constant<T> X_velocity_prior_mu(x_init[2]), noise_velocity_prior(0.1), X_velocity_sample(0);
    auto X_velocity_prior = normal_dist<T>(X_velocity_prior_mu, noise_velocity_prior, X_velocity_sample);
    
    T acc_obs = (T) 0;
    T gps_position_obs = (T) 0;

    // Initial cmd is x_init
    std::array<T,3> cmd{0,0,0};std::array<std::array<T,3>,3> K;

    // call gps every 2 steps
    bool call_gps = false;
    // std::ofstream results("results.csv");
    // std::cout<< "time,true,infer" << std::endl;
    // results<< "time,true,infer" << std::endl;
    Serial.println("time,true,infer");
    std::array<T,time_steps> X_infer;
    unsigned long start_time = millis();
    for(int time_step = 0; time_step < time_steps; time_step++)
    {
        // std::cout << "Time step: " << time_step << std::endl;
        if(time_step % 2 == 0) // optimize this for fixedpoint
        {
            call_gps = true;
        }
        else
        {
            call_gps = false;
        }
        
        
        // Simulate environment for ground truth
        env(x_init, cmd, X_true[time_step],A, B, noise_covariance, acc_obs, gps_position_obs, true, true);
        // std::cout << "True: " << std::endl;
        // print(X_true[time_step]);
        // Kalman Filter begins here
        perform_LTI(A, B, x_init, cmd, X_predicted_prior);
        X_pos_prior_mu(sample(X_predicted_prior[0], noise_covariance[0][0]));
        X_acc_prior_mu(sample(X_predicted_prior[2], noise_covariance[2][2]));
        X_velocity_prior_mu(sample(X_predicted_prior[1], noise_covariance[1][1]));

        // Infer acceleration and position
        Infer(X_acc_obs, X_acc_prior, X_acc_guide,X_acc_prior_mu_guide,noise_acc_prior_guide,(T) 0.01, 4, 20);
        if(call_gps)
        {
            Infer(X_pos_obs, X_pos_prior, X_pos_guide,X_pos_prior_mu_guide,noise_pos_prior_guide,(T) 0.01, 4, 20);
            x_init[0] = X_pos_prior_mu_guide.value();
        }
        else
        {
            x_init[0] = X_pos_prior_mu.value();
        }
        X_infer[time_step] = x_init[0];
        // Serial.print(time_step);
        // Serial.print(",");
        // Serial.print((float)X_true[time_step][0]);
        // Serial.print(",");
        // Serial.println((float)x_init[0]);

        // std::cout << time_step << "," << X_true[time_step][0] << "," << x_init[0] << std::endl;
        // results << time_step << "," << X_true[time_step][0] << "," << x_init[0] << std::endl;
        // Update the state

        x_init[1] = X_velocity_prior_mu.value();
        x_init[2] = X_acc_prior_mu_guide.value();
        // std::cout << "Inferred: " << std::endl;
        // print(x_init);

        if(time_step == 0)
        {
            // Update the control
            LQR(A_approx, B, Q, R, Pk, Pk_1, N, x_init, K, cmd, true);
            // std::cout << "Control for t = 1" << std::endl;
            // print(cmd);
        }
        else
        {
            LQR(A_approx, B, Q, R, Pk, Pk_1, N, x_init, K, cmd, false);
        }
    }
    unsigned long end_time = millis();

    for(int i = 0; i < time_steps; i++)
    {
        Serial.print(i);
        Serial.print(",");
        Serial.print((float)X_true[i][0]);
        Serial.print(",");
        Serial.println((float)X_infer[i]);
    }

    Serial.print("Time taken: ");
    Serial.println((float)(end_time - start_time));

}

void loop()
{

}
