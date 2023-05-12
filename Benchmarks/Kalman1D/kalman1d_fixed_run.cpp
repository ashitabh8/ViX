//[ARD:REMOVE_BEGIN]
#include "../../include/distributions_latest.hpp"
#include "../../include/ios.hpp"
#include <iostream>
#include <fstream>
#include <random>
//[ARD:REMOVE_END]

//[ARD:UNDEFS]

//[ARD:ADD_HEADER]

// 14 fractional bits + 2^5 scaling factor gives good results.

// #define T double
#define T fpm::fixed<std::int32_t, std::int64_t, 10>
#define SIGMA_LB 0.1
#define SIGMA_UB 5
// std::mt19937 gen(3);
// gen

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


template< typename Obs_ty, typename prior_T, typename guide_T, typename mu_learn_ty, typename std_learn_ty >
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
    T init_x0 = 0;
    T init_sigma = 2500;
    T x0 = sample(init_x0, init_sigma); // this decides the "starting point" of the data
    T noise = 1;
    for(int i = 0; i < N; i++)
    {
        xtrue[i] = sample(x0, noise);
        yobs[i] = sample(xtrue[i], noise);
    }
}

//[ARD:REMOVE_BEGIN]
template <int N>
void print_data(std::array<T, N> &xtrue, std::array<T, N> &yobs)
{
    for(int i = 0; i < N; i++)
    {
        std::cout << xtrue[i] << "," << yobs[i] << std::endl;
    }
}
//[ARD:REMOVE_END]

//[ARD:REMOVE_BEGIN]
#include <chrono>
//[ARD:REMOVE_END]

int main(int argc, char ** argv)//[ARD:SETUP_DEF]
{
    //[ARD:SERIAL_MONITOR]

    const int NUM_DATA_POINTS = 500;
    // std::array<float, NUM_DATA_POINTS> true_positions{287.081, 287.608, 287.939, 288.961, 288.637, 288.043, 288.6, 287.818, 289.386, 288.673, 289.59, 289.806, 290.304, 289.771, 288.789, 289.301, 287.806, 286.292, 285.647, 285.306, 286.689, 285.872, 285.595, 284.411, 283.067, 283.237, 283.788, 284.386, 284.224, 283.735, 284.035, 285.175, 285.768, 285.361, 283.53, 284.064, 282.665, 282.231, 282.621, 281.087, 280.664, 281.248, 282.266, 281.568, 281.108, 282.613, 281.189, 281.343, 280.151, 279.142, 278.326, 279.084, 279.118, 279.625, 279.352, 279.049, 279.104, 278.86, 279.347, 280.607, 281.838, 282.452, 283.447, 284.069, 284.883, 283.862, 283.034, 282.95, 283.135, 282.638, 283.303, 283.063, 282.952, 283.49, 282.487, 283.727, 283.694, 283.391, 285.096, 284.83, 284.336, 284.624, 285.088, 286.23, 288.223, 287.909, 288.386, 289.226, 288.659, 288.193, 287.507, 287.991, 287.909, 286.996, 287.884, 288.131, 287.537, 286.674, 285.508, 285.005, 284.525, 284.969, 284.908, 287.031, 286.926, 287.98, 287.237, 288.399, 288.493, 287.997, 289.747, 289.537, 288.037, 286.227, 286.275, 285.779, 283.498, 283.867, 285.718, 286.141, 286.912, 286.963, 286.777, 285.855, 285.418, 286.509, 286.429, 284.674, 284.209, 284.36, 284.379, 284.731, 286.888, 286.024, 285.306, 286.473, 285.925, 287.142, 285.276, 284.48, 286.551, 287.729, 287.969, 288.02, 288.217, 287.468, 287.163, 287.329, 287.594, 286.355, 286.387, 286.25, 284.948, 284.897, 284.349, 283.114, 283.574, 285.575, 284.656, 284.47, 284.558, 283.1, 283.823, 283.218, 283.58, 284.345, 285.469, 284.215, 284.23, 284.089, 284.683, 284.599, 284.316, 283.281, 282.984, 283.194, 281.455, 280.825, 283.008, 285.048, 284.613, 282.834, 283.383, 280.945, 279.286, 278.536, 276.805, 276.908, 276.934, 278.333, 277.88, 278.204, 277.837, 277.148, 276.632, 276.769, 277.496, 277.296, 276.658, 276.622, 276.815, 275.76, 275.297, 275.408, 277.874, 277.514, 277.487, 276.487, 276.934, 277.542, 277.173, 278.085, 278.28, 277.672, 278.379, 277.593, 278.923, 278.343, 277.266, 276.988, 277.315, 278.29, 278.478, 278.77, 277.175, 277.172, 277.398, 276.798, 276.75, 277.555, 277.909, 276.848, 274.815, 275.464, 274.97, 274.778, 273.624, 273.429, 273.745, 275.042, 275.109, 275.751, 275.444, 274.553, 274.995, 275.639, 277.307, 277.122, 277.217, 275.698, 277.217, 275.913, 277.657, 276.281, 279.778, 279.836, 279.762, 279.946, 278.165, 277.645, 276.487, 276.637, 276.614, 275.724, 275.417, 274.236, 276.036, 277.137, 277.188, 276.825, 278.892, 277.896, 278.066, 278.674, 277.141, 277.418, 278.012, 276.393, 276.218, 275.396, 275.416, 276.334, 277.735, 279.392, 278.858, 279.041, 279.028, 278.355, 280.228, 279.761, 279.883, 280.102, 280.881, 282.992, 279.972, 279.657, 281.449, 281.869, 282.268, 284.034, 285.297, 286.539, 287.653, 289.52, 288.265, 286.954, 289.089, 288.757, 288.435, 288.173, 287.204, 287.673, 285.68, 284.368, 285.269, 285.479, 285.413, 284.226, 284.115, 285.586, 285.95, 285.35, 285.783, 284.48, 284.305, 284.418, 283.95, 282.179, 282.428, 282.301, 282.91, 281.311, 282.656, 282.739, 282.79, 281.057, 281.71, 280.406, 281.091, 280.959, 281.832, 283.15, 283.108, 282.895, 283.091, 284.902, 285.218, 284.476, 282.392, 281.875, 281.505, 281.459, 282.806, 283.354, 283.577, 284.073, 283.294, 282.988, 283.659, 284.683, 284.096, 285.455, 284.903, 285.381, 284.257, 283.683, 284.806, 285.406, 284.885, 285.108, 285.51, 286.209, 286.104, 286.504, 286.112, 285.67, 285.025, 284.067, 284.092, 285.351, 286.266, 286.081, 286.565, 286.067, 287.036, 286.773, 287.137, 287.888, 290.044, 290.709, 290.678, 290.09, 291.755, 291.599, 291.242, 290.829, 289.88, 290.715, 290.053, 290.593, 289.378, 290.909, 291.841, 291.694, 290.372, 288.199, 288.069, 289.491, 290.026, 291.153, 291.354, 291.301, 291.476, 292.237, 292.411, 290.398, 291.893, 290.266, 289.94, 290.583, 290.075, 289.45, 289.826, 290.537, 291.911, 292.408, 292.76, 292.889, 293.52, 293.85, 293.987, 293.246, 293.024, 293.901, 294.877, 293.92, 292.054, 292.779, 292.214, 292.791, 292.347, 292.536, 292.938, 293.089, 293.372, 291.926, 293.421, 293.503, 294.111, 293.572, 293.141, 292.338, 292.455, 293.184, 294.011, 293.4, 294.142, 293.957, 295.075, 295.015, 295.034, 294.302, 295.188, 294.712, 295.49, 296.843, 296.609, 297.154, 294.495, 295.07, 295.835, 294.55, 294.486, 294.931, 294.272, 294.7, 295.681, 295.178, 296.211, 295.586, 295.121, 294.551, 296.92, 297.201, 297.164, 295.429, 296.959, 297.681, 299.189, 299.786, 301.228, 302.226, 300.456, 300.506, 300.569, 300.151, 300.284, 299.438, 300.46, 300.904};
    
    std::array<float, NUM_DATA_POINTS> true_positions {29.4646, 29.4982, 31.6229, 31.4951, 30.5592, 28.5439, 28.3593, 28.7362, 28.4009, 28.1158, 27.6449, 27.2393, 26.2511, 25.2794, 26.6976, 27.2132, 25.7774, 25.3595, 26.1056, 26.1245, 27.2267, 27.3536, 28.2836, 30.2884, 31.6337, 31.0535, 29.8123, 28.8869, 28.5035, 28.054, 28.7515, 28.8217, 28.9794, 31.4499, 29.3164, 30.4904, 29.4174, 29.2631, 30.3193, 30.3276, 31.7731, 31.1244, 33.3711, 31.5641, 31.82, 30.8718, 31.1166, 31.3491, 32.2426, 32.999, 33.6472, 33.3363, 33.1864, 32.4412, 32.1704, 32.9693, 31.3917, 31.5362, 30.7689, 29.1346, 30.2562, 30.5156, 30.3372, 30.7675, 31.3595, 30.6505, 30.1041, 29.7211, 29.5969, 28.3528, 28.9282, 29.0416, 31.1905, 31.0621, 31.3096, 31.6266, 31.9997, 32.7926, 32.5764, 32.068, 33.3382, 33.1708, 34.9301, 34.8973, 34.8544, 34.7756, 35.1066, 34.0286, 36.2153, 35.3206, 36.0573, 36.4866, 37.5757, 36.6371, 34.7194, 33.7407, 33.8462, 35.0728, 33.3694, 31.9068, 33.4145, 32.9373, 33.145, 31.7672, 32.3287, 32.4125, 31.3731, 32.3429, 32.9473, 33.6961, 33.0231, 32.6663, 32.0411, 30.4925, 29.3154, 30.4605, 30.8083, 30.9827, 30.2217, 30.9753, 30.3911, 30.1019, 30.4132, 29.3987, 29.1009, 30.9488, 32.473, 31.834, 29.8232, 28.6966, 31.1461, 33.7561, 34.9481, 36.4476, 36.1643, 36.0037, 35.2963, 36.4797, 35.8327, 34.4813, 33.5594, 30.7126, 31.533, 31.6548, 33.2521, 32.9552, 32.0624, 32.7785, 32.4673, 33.5726, 32.7274, 33.6303, 34.298, 34.883, 35.283, 35.9305, 36.7624, 36.5678, 35.6956, 36.7098, 36.1854, 35.2334, 35.6791, 37.4765, 37.8766, 37.6886, 38.4773, 37.3799, 36.9528, 36.2822, 35.2294, 36.4908, 36.7347, 37.0532, 38.1574, 38.6995, 37.9811, 35.3941, 37.295, 36.6004, 36.4641, 37.729, 39.8472, 40.3668, 40.8749, 39.1633, 39.0808, 39.1961, 38.3428, 38.9972, 38.2822, 40.0184, 38.4741, 36.3986, 37.118, 36.1996, 33.984, 35.172, 34.6989, 33.8707, 32.641, 32.7247, 31.8462, 32.808, 31.6315, 31.8171, 33.6149, 32.4376, 31.1064, 32.3746, 31.8666, 31.8875, 32.4764, 33.4015, 32.8636, 32.2011, 31.6272, 32.1776, 31.6077, 31.3552, 30.0314, 31.3773, 30.3803, 29.7335, 29.8065, 29.3368, 29.8915, 30.6952, 30.363, 29.5674, 28.5714, 28.3986, 27.1299, 26.9043, 26.5123, 27.1483, 27.2206, 27.4842, 26.5622, 27.1067, 26.6035, 26.2441, 25.5688, 26.1074, 25.7356, 26.2178, 25.7225, 25.8571, 26.8136, 26.7825, 26.3855, 24.7645, 21.0153, 19.1159, 20.0546, 19.4588, 20.564, 21.3889, 21.2162, 20.4007, 18.7415, 17.9098, 16.4997, 16.1531, 15.1867, 14.2175, 13.9616, 14.3181, 12.0829, 10.7968, 12.2952, 12.995, 10.7717, 11.5309, 11.8913, 10.343, 9.88753, 9.86059, 9.69187, 8.50071, 7.62095, 7.91549, 7.70944, 6.59857, 5.70014, 7.01319, 8.99725, 9.41087, 10.0025, 9.91056, 9.61363, 9.96204, 9.21228, 12.2128, 11.5369, 10.4222, 11.6663, 10.6131, 9.73121, 9.78106, 9.34659, 11.236, 9.34832, 9.95273, 9.94321, 9.35065, 10.3377, 10.7272, 11.4227, 11.5046, 10.3933, 10.1834, 10.4748, 9.88301, 9.24031, 9.20161, 9.09899, 10.7721, 12.0314, 11.1309, 9.98625, 9.97182, 7.89221, 6.62299, 9.24312, 8.08708, 8.16431, 7.48046, 7.77852, 7.63689, 6.86036, 7.54517, 7.61618, 9.3093, 11.2025, 10.7092, 10.3087, 10.9588, 11.7486, 13.196, 13.0615, 15.3967, 15.4161, 13.9714, 12.3749, 10.6649, 11.6394, 12.0544, 11.8019, 12.8292, 13.1048, 11.265, 11.6575, 11.2986, 12.1869, 14.5539, 14.2637, 14.3191, 14.8866, 14.9048, 14.4213, 13.1159, 12.8448, 13.6182, 13.6162, 14.0466, 14.0482, 14.4565, 16.2924, 16.4561, 16.0656, 17.7855, 19.4752, 20.0758, 18.0294, 17.6696, 17.6273, 19.294, 19.6845, 19.853, 18.233, 18.499, 19.4886, 19.3104, 17.7847, 18.026, 18.7509, 19.311, 19.8326, 20.5006, 21.1633, 21.7591, 22.4312, 22.9209, 23.4271, 22.9037, 22.6685, 21.8462, 20.977, 19.507, 18.5205, 19.2473, 17.6388, 18.1687, 18.1384, 18.7421, 17.5052, 17.909, 18.2537, 19.9594, 18.3704, 18.7445, 18.6425, 18.555, 18.6402, 17.4063, 16.4544, 16.3595, 15.885, 15.8969, 16.3349, 17.0014, 16.7217, 17.3199, 17.9769, 17.0439, 16.341, 15.5929, 16.4602, 17.3027, 14.5007, 15.6563, 14.8266, 14.1332, 13.093, 11.507, 10.9706, 10.7766, 12.537, 13.4538, 14.0454, 12.6446, 12.8719, 12.7192, 14.0304, 12.9092, 14.181, 14.8051, 15.2732, 14.1944, 13.6876, 15.2432, 14.9831, 15.0825, 14.6982, 15.5227, 14.7277, 14.692, 14.5246, 14.9582, 15.7612, 15.5744, 14.8908, 16.9209, 18.0822, 18.2775, 18.3898, 18.6968, 20.4929, 20.1078, 20.7935, 19.8329, 21.8891, 22.2214, 21.5215, 20.3302, 20.9738, 21.8711, 22.5634, 21.658, 20.2881, 19.747, 19.9425, 21.4264, 21.8097, 20.5021, 21.573, 19.992, 19.9787, 19.5033, 20.3711, 20.7118, 21.3852, 22.7086, 21.2997, 21.2245, 22.122, 21.9382, 20.7654, 20.8879};
    std::array<float, NUM_DATA_POINTS> yobs_positions {29.4979, 29.9608, 32.6439, 30.8079, 29.7865, 27.6493, 28.2855, 30.1385, 29.6125, 26.8016, 29.3181, 26.6323, 27.3558, 25.4381, 27.2655, 28.3299, 26.7184, 25.2247, 26.2015, 25.9666, 27.3593, 28.3442, 27.5417, 31.0642, 31.139, 31.9687, 30.3644, 28.1826, 28.1461, 29.2518, 27.0963, 30.4787, 27.1128, 29.4973, 29.1611, 32.5293, 29.7277, 30.8335, 30.5663, 28.8576, 30.9449, 30.6284, 33.6252, 31.5533, 31.635, 30.3767, 33.0984, 31.0851, 31.4879, 32.8797, 32.3418, 33.7403, 33.0854, 31.7201, 32.9714, 32.7872, 29.7144, 31.9541, 29.728, 30.4022, 28.7376, 29.7963, 30.4201, 31.1948, 31.6561, 31.8498, 29.7258, 29.6001, 29.9522, 29.4042, 29.7153, 29.0655, 31.1143, 30.0451, 31.8328, 32.1612, 33.2202, 34.1981, 31.3711, 30.2843, 33.4577, 32.8108, 35.5005, 36.3976, 33.7811, 34.6091, 33.3029, 33.5797, 35.6636, 34.0809, 34.3308, 35.2852, 36.5817, 35.9799, 33.7786, 34.7703, 34.6269, 33.0895, 33.8054, 32.3729, 32.9118, 32.1201, 33.7929, 30.8475, 32.9983, 32.418, 30.3165, 32.0619, 32.6424, 34.9836, 32.3465, 32.5225, 32.6998, 31.0738, 29.933, 31.3823, 28.8679, 31.5227, 29.9552, 30.5851, 30.8897, 30.1255, 31.9043, 29.5299, 29.6255, 28.5599, 31.6668, 32.6665, 29.5097, 27.2883, 30.7195, 34.6492, 36.1604, 36.3399, 35.3655, 36.6874, 35.42, 35.9833, 35.8019, 35.2443, 34.0834, 30.5934, 31.6825, 32.3897, 32.8403, 33.4538, 31.3668, 32.6242, 31.2776, 31.8963, 33.8862, 35.0064, 34.117, 34.7901, 36.7759, 35.9304, 36.2244, 36.6771, 35.1248, 38.3524, 37.1973, 35.1634, 34.2195, 35.8308, 38.4103, 36.1841, 38.8118, 36.53, 36.4513, 37.1647, 35.3642, 35.9693, 36.2358, 36.847, 38.3278, 37.3748, 37.3932, 37.4638, 35.996, 36.4086, 36.1847, 39.1088, 38.9711, 41.4816, 40.3731, 39.6975, 38.4766, 38.0798, 37.6313, 39.5316, 38.6799, 39.1331, 39.1994, 35.9477, 37.2475, 36.867, 33.48, 35.563, 34.3435, 35.6797, 34.5921, 35.2455, 30.6904, 32.1301, 30.6195, 33.5224, 34.5828, 33.526, 31.2702, 33.5855, 32.5131, 30.7706, 32.0885, 34.6061, 31.1459, 31.8297, 31.0755, 32.7936, 31.5925, 31.9882, 29.7182, 29.4739, 30.8972, 29.3137, 32.0339, 29.1717, 30.1651, 29.3795, 32.8835, 29.7089, 29.1916, 28.7541, 26.209, 27.3635, 25.6758, 27.5294, 26.0789, 26.435, 24.6541, 26.5237, 26.2593, 26.8738, 24.6071, 25.8036, 27.1183, 28.0442, 24.9628, 25.0722, 26.9935, 25.4102, 26.2476, 24.7877, 21.5086, 19.5097, 17.7482, 19.637, 19.4931, 21.5867, 20.6631, 21.3204, 18.139, 18.9733, 16.3649, 16.5618, 13.654, 15.3445, 14.676, 14.6172, 10.2732, 11.0683, 11.4118, 12.2912, 10.3982, 9.84423, 10.8363, 12.0468, 9.26656, 9.03236, 8.89917, 8.91903, 7.8472, 7.97408, 7.90789, 5.12643, 4.37943, 8.65293, 7.43278, 11.2351, 9.61165, 9.13899, 9.91627, 9.90443, 9.32086, 11.0805, 12.6367, 9.12952, 13.4912, 11.0708, 9.82074, 9.75883, 9.33089, 9.0653, 8.95507, 10.2333, 11.2434, 11.0785, 10.6951, 10.3573, 11.3916, 11.2922, 11.9162, 11.483, 9.8752, 8.25918, 9.46713, 10.5734, 9.07221, 11.5114, 14.5497, 10.6435, 10.7645, 9.72252, 7.76859, 5.73202, 9.03871, 8.21999, 7.73974, 7.8437, 7.35791, 8.32072, 7.40355, 7.60487, 7.44106, 9.82976, 11.7777, 10.4127, 10.5695, 10.292, 12.837, 15.4574, 11.885, 15.3046, 15.7468, 15.3553, 12.9523, 10.8791, 9.35644, 12.7533, 10.9666, 12.2838, 11.7144, 11.9626, 11.3854, 11.3913, 11.2229, 16.9294, 12.5245, 14.5059, 15.2471, 15.8278, 13.4121, 13.405, 12.9298, 13.5499, 12.6683, 12.4647, 13.9817, 16.5166, 16.9359, 15.6645, 16.4925, 17.5834, 19.7949, 19.1812, 19.1086, 18.4761, 17.2277, 19.6388, 20.013, 19.7101, 18.1677, 18.61, 21.2384, 17.9369, 18.8975, 18.4484, 17.4519, 18.2224, 18.826, 19.6714, 20.6711, 19.4589, 22.9716, 22.6518, 24.045, 21.174, 22.4474, 20.9047, 21.7171, 18.2394, 17.7438, 20.1356, 18.3581, 18.3872, 18.8814, 17.5873, 16.8479, 17.9293, 17.7369, 18.2228, 20.0322, 16.8337, 18.0344, 18.5745, 19.359, 20.2284, 17.9546, 15.7081, 16.6565, 15.4485, 15.9142, 15.3228, 15.5131, 15.7084, 18.1278, 15.1512, 16.9397, 17.106, 17.8756, 17.2166, 13.9273, 16.2843, 15.0525, 13.7097, 14.8887, 10.2052, 10.6275, 13.1328, 10.8698, 14.2911, 13.5755, 11.5298, 13.7139, 12.3536, 11.8729, 12.6628, 14.3182, 13.4734, 16.0686, 12.686, 12.5885, 15.9382, 15.694, 14.1017, 15.295, 14.9821, 16.0475, 16.4492, 15.091, 14.0167, 15.6664, 15.5943, 14.2108, 16.2628, 18.4418, 19.3296, 18.9387, 20.3985, 20.1452, 18.59, 22.0637, 21.5533, 21.4334, 22.6126, 22.6163, 21.7772, 20.2394, 22.1092, 23.1412, 21.9838, 20.4504, 20.321, 20.1744, 20.856, 21.0599, 21.1755, 21.8298, 18.7578, 20.3592, 20.355, 19.4457, 20.1422, 22.3821, 22.0716, 20.5751, 21.0904, 20.109, 21.6175, 21.5086, 21.3913};
    
    // add 10 to true_positions
    for(int i = 0; i < NUM_DATA_POINTS; i++)
    {
        true_positions[i] += 70;
    }

    // add 10 to yobs_positions
    for(int i = 0; i < NUM_DATA_POINTS; i++)
    {
        yobs_positions[i] += 70;
    }

    // std::array<T, NUM_DATA_POINTS> yobs_positions{286.604, 287.156, 288.433, 288.594, 289.706, 287.998, 288.407, 288.365, 290.628, 288.488, 288.775, 288.885, 291.16, 289.184, 289.307, 290.322, 288.817, 285.313, 286.724, 286.905, 286.703, 286.62, 284.701, 283.428, 283.217, 282.225, 285.942, 283.992, 284.905, 285.068, 285.272, 285.354, 287.664, 284.391, 283.158, 283.545, 281.968, 283.268, 282.391, 282.577, 280.983, 279.564, 282.793, 282.602, 281.349, 282.492, 280.508, 280.979, 280.611, 278.839, 279.03, 278.9, 278.065, 278.244, 279.223, 278.776, 279.242, 278.155, 279.624, 281.053, 282.804, 283.277, 284.535, 281.324, 285.37, 284.395, 282.774, 282.426, 282.364, 282.28, 285.691, 282.712, 286.226, 281.323, 283.72, 282.441, 285.092, 283.899, 286.545, 285.757, 283.895, 283.074, 284.826, 285.965, 287.427, 287.801, 287.59, 288.722, 289.174, 288.579, 288.003, 288.465, 288.719, 287.985, 287.109, 286.303, 288.585, 285.836, 285.518, 284.332, 285.227, 283.377, 285.311, 288.358, 288.316, 290.259, 286.501, 287.293, 290.197, 288.336, 289.843, 291.346, 287.057, 287.057, 287.621, 286.116, 282.419, 283.196, 285.908, 286.79, 287.404, 285.549, 286.351, 285.84, 285.244, 287.152, 288.254, 283.436, 285.149, 284.764, 283.668, 284.058, 287.268, 286.107, 285.418, 284.404, 284.254, 285.895, 284.555, 285.688, 287.217, 286.793, 288.171, 287.849, 287.95, 287.449, 287.646, 288.131, 287.289, 286.805, 286.275, 287.05, 283.143, 285.76, 281.776, 284.217, 281.939, 284.876, 285.957, 286.402, 283.311, 283.198, 284.198, 282.299, 284.733, 285.035, 284.571, 285.387, 283.225, 284.176, 284.929, 287.055, 285.11, 282.509, 282.718, 284.567, 281.484, 281.792, 281.738, 284.838, 285.481, 283.873, 282.792, 279.481, 278.946, 278.262, 277.295, 275.893, 276.026, 278.795, 277.392, 279.248, 277.918, 278.484, 275.926, 277.071, 277.521, 278.812, 278.225, 277.152, 277.924, 274.681, 276.299, 275.793, 278.765, 279.074, 277.998, 277.18, 276.643, 277.818, 277.058, 279.802, 278.898, 276.343, 277.609, 276.455, 279.178, 278.758, 276.223, 278.636, 278.684, 278.744, 277.535, 279.808, 277.419, 275.865, 277.844, 276.1, 276.627, 276.766, 275.625, 275.451, 273.633, 276.709, 275.354, 274.727, 272.538, 275.483, 274.788, 276.473, 275.25, 274.494, 276.402, 273.196, 274.649, 274.234, 279.106, 276.024, 278.287, 276.496, 278.327, 275.6, 278.292, 275.886, 281.275, 278.863, 279.603, 279.368, 278.21, 277.952, 275.668, 276.44, 277.814, 275.501, 275.665, 272.63, 275.582, 279.016, 276.395, 278.009, 280.281, 277.278, 278.626, 278.76, 276.913, 278.004, 278.483, 274.391, 276.035, 274.743, 277.004, 274.636, 278.141, 278.003, 280.988, 279.414, 279.236, 279.443, 278.019, 278.985, 279.379, 279.969, 279.515, 284.277, 279.791, 280.206, 280.896, 282.307, 284.119, 282.181, 286.443, 287.344, 288.36, 289.2, 286.945, 285.779, 289.211, 289.47, 288.728, 287.955, 286.882, 287.777, 285.796, 284.487, 284.775, 285.767, 284.724, 282.812, 284.739, 287.192, 284.815, 283.953, 286.537, 284.642, 285.098, 284.479, 284.478, 281.197, 284.045, 284.058, 283.066, 281.271, 283.442, 282.355, 283.929, 281.033, 282.053, 282.356, 281.315, 279.746, 281.63, 283.422, 283.628, 281.951, 282.833, 283.972, 283.341, 283.938, 284.332, 283.049, 281.437, 279.143, 281.826, 283.537, 284.206, 283.307, 283.343, 282.925, 283.894, 282.968, 284.775, 284.588, 284.938, 284.593, 283.035, 285.103, 285.367, 284.739, 284.12, 283.545, 283.159, 286.299, 286.769, 287.74, 285.431, 285.165, 283.677, 283.992, 282.901, 286.014, 288.465, 284.671, 284.899, 285.992, 287.05, 287.824, 288.434, 288.004, 289.524, 291.206, 290.519, 291.456, 291.056, 293.071, 292.721, 291.755, 290.737, 292.395, 290.935, 290.904, 288.468, 290.578, 290.579, 292.508, 290.108, 285.364, 287.929, 289.38, 289.618, 292.225, 290.253, 291.068, 292.131, 292.942, 291.771, 289.749, 292.554, 291.657, 289.093, 290.093, 289.351, 287.511, 289.264, 288.13, 292.294, 293.347, 292.47, 291.114, 291.593, 295.439, 294.433, 293.61, 293.573, 291.021, 293.528, 294.449, 291.12, 292.258, 290.312, 291.864, 290.955, 293.204, 292.802, 293.66, 292.466, 292.193, 293.093, 292.596, 294.701, 293.006, 291.032, 292.831, 292.015, 292.041, 294.675, 293.546, 293.878, 295.052, 296.79, 295.224, 295.923, 292.308, 295.344, 294.542, 296.184, 297.308, 297.426, 297.477, 294.471, 296.111, 294.881, 294.917, 293.617, 295.634, 295.8, 294.739, 295.725, 294.621, 296.062, 294.528, 293.784, 293.981, 298.876, 295.888, 297.3, 295.086, 296.43, 297.204, 299.222, 299.058, 300.979, 301.701, 299.864, 303.434, 299.73, 300.68, 298.622, 299.765, 301.769, 299.94};
    
    const int num_experiments = 1;
    // std::array<float, num_experiments> position_infer;
    std::array<float, num_experiments> rmse_all_exp;
    std::array<float, num_experiments> geomean_all_exp;

    //[ARD:REMOVE_BEGIN]
    std::cout << "rmse,geomean,num_iters,num_data_samples,num_obs_samples" << std::endl;
    //[ARD:REMOVE_END]


    //[ARD:UNCOMMENT_BEGIN]
    // Serial.println("rmse,geomean,num_iters,time_ms,num_data_samples,num_obs_samples");
    //[ARD:UNCOMMENT_END]

    for(int curr_exp = 0; curr_exp < num_experiments; curr_exp++)
    {

         //[ARD:UNCOMMENT_BEGIN]
         // randomSeed(analogRead(0));
        // unsigned long total_time = 0;
         // gen.seed(random(100));
         //[ARD:UNCOMMENT_END]

        //  [ARD:START_TIME]

        //[ARD:REMOVE_BEGIN]
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //[ARD:REMOVE_END]
        
        // // Initialising the prior
        // T init_mu = 0;
        // T init_sigma = 250;
        // 20 original mu
        Constant<T> mu_T(90), sigma_T(5), data_T(0);
        auto T_prior = normal_dist<T>(mu_T, sigma_T,data_T);

        Constant<T> sigma_N(1), data_N(20);
        auto N = normal_dist<T>(T_prior, sigma_N,data_N);

        // Varaitional assignment of T (the posterior)
        Variable<T> mu_var(90), sigma_var(1);
        auto T_var = normal_dist<T>(mu_var, sigma_var, data_T);

        // // Learning
        int NUM_SAMPLES = 4;
        int NUM_ITERS = 200;
        // // auto results = Infer(N, T_prior, T_var, mu_var, sigma_var, 0.01, NUM_SAMPLES, NUM_ITERS);
        T learning_rate = (T)0.01;
        std::array<T, NUM_DATA_POINTS> predicted_positions;

        //[ARD:START_TIME]
        for(int i = 0; i < yobs_positions.size(); i++)
        {
            data_N(yobs_positions[i]);
            // mu_T(mu_T.value() + time_step*velocity_control_with_noise[i]);
            auto results = Infer(N, T_prior, T_var, mu_var, sigma_var,learning_rate, NUM_SAMPLES, NUM_ITERS);

            // std::cout << "ELBO: " << results.elbo << std::endl;
            // std::cout << "Mean: " << results.mean << std::endl;
            // std::cout << "Sigma: " << results.sigma << std::endl;

            mu_T(results.mean);
            sigma_T(1);
            predicted_positions[i] =  results.mean;
            // sigma_T(results.sigma);
            // data_N(25);
            mu_var(results.mean);
            // sigma_var(results.sigma);
        }
        //[ARD:END_TIME]


        //[ARD:UNCOMMENT_BEGIN]
        // total_time+=curr_total_time;
        // Serial.println("Average Total time: " + String((float)total_time/num_experiments));
        //[ARD:UNCOMMENT_END]


        // convert predicted position to float
        std::array<float, NUM_DATA_POINTS> predicted_positions_float;
        for(int i = 0; i < predicted_positions.size(); i++)
        {
            predicted_positions_float[i] = (float)predicted_positions[i];
        }
        

        float rmse_i = rmse(true_positions, predicted_positions_float,0);
        float geomean_i = geomean(true_positions, predicted_positions_float,0);
        rmse_all_exp[curr_exp] = rmse_i;
        geomean_all_exp[curr_exp] = geomean_i;

        // print predicted positions float
        // std::cout << "timestep,true_position,predicted_position,relative_error" << std::endl;

        // for(int i = 0; i < true_positions.size(); i++)
        // {
        //     std::cout << i << "," << true_positions[i] << "," << predicted_positions_float[i] << ",      "<< std::abs(predicted_positions_float[i] - true_positions[i])/true_positions[i] << std::endl;
        // }
        

        

        //[ARD:REMOVE_BEGIN]
        std::cout <<rmse_i << "," << geomean_i << "," << NUM_ITERS << "," << NUM_SAMPLES << "," << NUM_DATA_POINTS << std::endl;
        //[ARD:REMOVE_END]

        break;

        //[ARD:UNCOMMENT_BEGIN]
        // Serial.println(String(float(rmse_i),4) + "," + String(float(geomean_i),4) + "," + String(NUM_ITERS) + ","+String(curr_total_time)+","+String(NUM_SAMPLES) + "," + String(NUM_DATA_POINTS));
        //[ARD:UNCOMMENT_END]
    }

}

//[ARD:ADD_LOOP]