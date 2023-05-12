// #include <type_traits>
// #include <random>
// #include <vector>
// #include <cmath>
// #include <bits/stdc++.h>
// #include <string>

#include "/home/ashitabh/Documents/fpm_test_code/include/ad_engine.hpp"
#include<vector>
#include <random>

class DistributionBase{};

template <typename lb_ty, typename ub_ty,typename input_ty, typename T,typename EXPRESSION_TYPE, typename LOG_TYPE, typename RefTy = ByRefClass>
class Uniform : public DistributionBase, ADBase{
    lb_ty& lb;
    ub_ty& ub;
    input_ty& x_in;
    // std::random_device r;
    // std::seed_seq seed{r(), r()};
    // std::mt19937 gen{seed};
    std::default_random_engine gen;
    T range;
    EXPRESSION_TYPE expression = T{1} / (ub - lb);
    LOG_TYPE log_expression = T{0} -log(ub - lb);

    T init_log_pdf_value;

    public:
        
        using type_ = T;
        using ref_type = RefTy;
        Uniform(lb_ty& lb, ub_ty& ub, input_ty& x_in) : lb(lb), ub(ub), x_in(x_in)
        {
            range = ub.value() - lb.value();
            init_log_pdf_value = log_expression.value();
        }

        auto pdf()
        {
            return expression;
        }

        auto lb_value()
        {
            return lb.value();
        }

        auto ub_value()
        {
            return ub.value();
        }

        // auto lb()
        // {
        //     return lb.value();
        // }

        // auto ub()
        // {
        //     return ub.value();
        // }

        auto log_pdf()
        {
            return log_expression;
        }

        auto log_pdf_value()
        {
            if constexpr(std::is_base_of<fpm::fixedpoint_base, type_>::value)
            {
                type_ x_value = x_in.value();
                if(x_value < lb.value() || x_value > ub.value())
                {
                    return std::numeric_limits<type_>::lowest();
                }
                else
                {
                    return init_log_pdf_value;
                }
            }
            if constexpr(std::is_arithmetic<type_>::value)
            {
                type_ x_value = x_in.value();
                if(x_value < lb.value() || x_value > ub.value())
                {
                    return -std::numeric_limits<type_>::infinity();
                }
                else
                {
                    return init_log_pdf_value;
                }
            }  
        }

        auto score() // same as log_pdf_value
        {
            if constexpr(std::is_base_of<fpm::fixedpoint_base, type_>::value)
            {
                type_ x_value = x_in.value();
                if(x_value < lb.value() || x_value > ub.value())
                {
                    return std::numeric_limits<type_>::lowest();
                }
                else
                {
                    return init_log_pdf_value;
                }
            }
            if constexpr(std::is_arithmetic<type_>::value)
            {
                type_ x_value = x_in.value();
                if(x_value < lb.value() || x_value > ub.value())
                {
                    return -std::numeric_limits<type_>::infinity();
                }
                else
                {
                    return init_log_pdf_value;
                }
            }  
        }

        T value() const
        {
            return x_in.value();
        }

        template <typename Tc = T, typename std::enable_if<std::is_base_of<fpm::fixedpoint_base, Tc>::value>::type* = nullptr>
        Tc sample()
        {
            auto sign_and_int_bits = Tc::integer_bits;

            uint32_t random32 = gen();
            random32 = random32 << sign_and_int_bits;
            random32 = random32 >> sign_and_int_bits;
            Tc uniform_0_1_sample = Tc::from_raw_value(random32);

            return uniform_0_1_sample * range + lb.value();
        }

        template <typename Tc = T, typename std::enable_if<std::is_arithmetic<Tc>::value>::type* = nullptr>
        Tc sample() 
        {
            std::uniform_real_distribution dist(lb.value() , ub.value()); 
            return dist(gen);
        }

};

template<typename input, typename T, typename RefTy = ByValueClass>
class Sigmoid: public DistributionBase, ADBase, AD_OP_Base // ONLY GIVE GAUSSIAN FOR NOW.
{

    template <class T_c =  T, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>
    T_c exp_mem(T_c x) const
    {
      // std::cout << "Running std..." << std::endl;
      return std::exp(x);
    }

    template <class T_c = T, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>
    T_c exp_mem(T_c x) const
    {
      // std::cout << "Running fixed..." << std::endl;
      return fpm::exp(x);
    }
    input& x_in;
    T sigmoid_value;

    public:
    using type_ = T;
    using ref_type = RefTy;
    Sigmoid(input& x_in) : x_in(x_in)
    {
    }

    T value() const
    {
        T temp = x_in.pdf_value();
        return T{1} / (T{1} + exp_mem(-temp));
        // get the pdf value of the normal here and apply the sigmoid to it
    }
};

template<typename T, typename input>
auto sigmoid(input& x_in)
{
    return Sigmoid<input, T>(x_in);
}


template<typename P_ty, typename input_ty, typename T, typename EXPRESSION_TYPE, typename RefTy = ByRefClass>
class Bernoulli: public DistributionBase, ADBase // ONLY PASS SIGMOID FOR NOW.
{
    template <class T_c =  T, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>
    T_c log_mem(T_c x) const
    {
      // std::cout << "Running std..." << std::endl;
      return std::log(x);
    }

    template <class T_c = T, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>
    T_c log_mem(T_c x) const
    {
      // std::cout << "Running fixed..." << std::endl;
      return fpm::log(x);
    }
    P_ty& P;
    input_ty& x_in;
    EXPRESSION_TYPE expression = P * x_in + (T{1} - P) * (T{1} - x_in);

    public:;
    Bernoulli(P_ty& P, input_ty& x_in) : P(P), x_in(x_in)
    {
    }

    auto score()
    {
        T x_value = x_in.value();
        T P_value = P.value();
        if(x_value == T{1})
        {
            return log_mem(P_value);
        }
        else if(x_value == T{0})
        {
            return log_mem(T{1} - P_value);
        }
        else
        {
            return std::numeric_limits<T>::lowest();
        }
    }

    T value() const
    {
        return expression.value();
    }
};

template<typename T, typename P_ty, typename input_ty, typename RefTy = ByRefClass>
auto bernoulli(P_ty& P, input_ty& x_in)
{
    auto expression = P * x_in + (T{1} - P) * (T{1} - x_in);
    return Bernoulli<P_ty, input_ty, T, decltype(expression), RefTy>(P, x_in);
}



template <typename N_ty, typename P_ty, typename input_ty,  typename T, typename EXPRESSION_TYPE, typename LOG_TYPE, typename RefTy = ByRefClass>
class Binomial : public DistributionBase, ADBase{
    N_ty& N;
    P_ty& P;
    input_ty& x_in;
    // K_ty& k;

    LOG_TYPE log_expression = x_in * log(P) + (N - x_in) * log( T{1} - P);
    public: 
    Binomial(N_ty& N, P_ty& P, input_ty& x_in) : N(N), P(P), x_in(x_in){}

    auto log_pdf()
    {
        return log_expression;
    }

    T score()
    {
        return log_expression.value();
    }

    T value() const
    {
        return x_in.value();
    }
};

template<typename T, typename N_ty, typename P_ty, typename input_ty>
inline auto binomial_dist(N_ty& N, P_ty& P, input_ty& x_in){
//   using main_ty = T;
  
  using log_type = decltype(x_in * log(P) + (N - x_in) * log(T{1} - P));

  return Binomial<N_ty, P_ty,input_ty,  T,void, log_type>(N, P,x_in);
}

template <typename mu_ty, typename stddev_ty,typename input_ty, typename T, typename EXPRESSION_TYPE,  typename LOG_TYPE, typename RefTy = ByRefClass>
class Normal : public DistributionBase, ADBase{
    // Data
    
    std::random_device r;
    std::seed_seq seed{r()};
    std::mt19937 gen{seed}; // seed 1 works for bayesian filter

    // std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution;


    // Constant<T> c1(0.3989422804014327), one(1), minus_half(-0.5);
    // div_ty<T>(one, stddev)
    // sub_ty<T>(x_in, mu)
    // mul_ty<T>(stddev, stddev)
    // mul_ty<T>()
    bool change_value = false;
    typename mu_ty::type_ mu_value;
    typename stddev_ty::type_ stddev_value;
    // mul_ty<T>(0.3989422804014327,mul_ty<T>(div_ty<>) )
    // EXPRESSION_TYPE expression = 0.3989422804014327 * (1/stddev) * exp(-0.5 * (x_in - mu) * (x_in - mu) / (stddev * stddev));
    // EXPRESSION_TYPE expression = T{0.3989422804014327} * (T{1}/stddev) * exp(T{-0.5} * (x_in - mu) * (x_in - mu) / (stddev * stddev));
    // // NEED TO USE THIS LOG EXPRESSION INSTEAD (note 1/stddev^2 is written out differently, 
    // //that is because for small values of stddev fpm was rounding it to 0 and throwing error. So, I have written it out explicitly)
    // LOG_TYPE log_expression = T{-0.9189385332046727}  - log(stddev) - T{0.5} * (T{1}/stddev) *(T{1}/stddev) * (x_in - mu) * (x_in - mu);

    // LOG_TYPE log_expression = (T{1}/stddev) *(T{1}/stddev) * (mu);
    //  LOG_TYPE log_expression = T{-0.9189385332046727}  - log(stddev) - (T{0.5}/(stddev * stddev)) * (x_in - mu) * (x_in - mu);
    // LOG_TYPE log_expression = -0.9189385332046727  - log(stddev) - 0.5 * (1/(stddev * stddev)) * (x_in - mu) * (x_in - mu);
    
    // std::string name;

    public:

        mu_ty& mu;
        stddev_ty& stddev;
        input_ty& x_in; 

        EXPRESSION_TYPE expression = T{0.3989422804014327} * (T{1}/stddev) * exp(T{-0.5} * (x_in - mu) * (x_in - mu) / (stddev * stddev));
    // NEED TO USE THIS LOG EXPRESSION INSTEAD (note 1/stddev^2 is written out differently, 
    //that is because for small values of stddev fpm was rounding it to 0 and throwing error. So, I have written it out explicitly)
        LOG_TYPE log_expression = T{-0.9189385332046727}  - log(stddev) - T{0.5} * (T{1}/stddev) *(T{1}/stddev) * (x_in - mu) * (x_in - mu);
        using type_ = T;
        using ref_type = RefTy;
        Normal(mu_ty& mu, stddev_ty& stddev, input_ty& x_in) : mu(mu), stddev(stddev), x_in(x_in)
        {
            distribution = std::uniform_real_distribution<float>(0,1);
        }

        auto inline pdf()
        {
            return expression;
        }

        auto inline pdf_value()
        {
            return expression.value();
        }

        auto log_pdf()
        {
            return log_expression;
        }

        // auto inline pdf(T x_in)
        // {
        //     x_in.set_value(x_in);
        //     return pdf().value();
        // }
        // auto inline log_pdf(T x_in)
        // {
        //     x_in.set_value(x_in);
        //     return log_pdf().value();
        // }

        T value() const
        {
            return x_in.value();
            // return expression.value();
        }

        void operator()(type_ vec)
        {
            this->x_in(vec);
        }

        constexpr inline T diff(int wrt) const
        {
            return T{expression.diff(wrt)};
        }

        template<typename DIFFTY>
        constexpr inline T diff(Variable<DIFFTY> &wrt_diff)
        {
            return this->diff(wrt_diff.Var_ID);
        }

        T lb()
        {
            return mu.value() - (T) 3 * stddev.value();
        }

        T ub()
        {
            return mu.value() + (T) 3 * stddev.value();
        }

        // template<typename Tc = T, typename std::enable_if<is_array_std<Tc>::value>::type* = nullptr>
        // T score() const
        // {
        //     T score_val = 0;
        //     if constexpr(std::is_base_of<DistributionBase, mu_ty>::value)
        //     {
        //         score_val += T{mu.score()};
        //     }

        //     if constexpr(std::is_base_of<DistributionBase, stddev_ty>::value)
        //     {
        //         score_val += T{stddev.score()};
        //     }
        //     auto log_prob_vec = log_expression.value();
        //     if constexpr(is_array_std<decltype(log_prob_vec)>::value)
        //     {
        //         for(auto i : log_prob_vec)
        //         {
        //             score_val += i;
        //         }
        //     }
        //     else
        //     {
        //         score_val += T{log_prob_vec};
        //     }
        //     return score_val;
        // }


        T score()
        {
            return log_expression.value();
        }

        T diff_log(int wrt) const
        {
            auto answer = log_expression.diff(wrt);
            if constexpr(is_array_std<decltype(answer)>::value)
            {
                T score_val = 0;
                for(auto i : answer)
                {
                    score_val += i;
                }
                return score_val;
            }
            else
            {
                return answer;
            }
            // return T{log_expression.diff(wrt)};
        }



        template<typename diff_ty>
        T diff_log(Variable<diff_ty> &wrt_diff)
        {
            return this->diff_log(wrt_diff.Var_ID);
        }
        // template<typename Tc = T, typename std::enable_if<!is_array_std<Tc>::value>::type* = nullptr>
        // Tc score() const
        // {
        //     T score_val = 0;
        //     if constexpr(std::is_base_of<DistributionBase, mu_ty>::value)
        //     {
        //         score_val += mu.score();
        //     }

        //     if constexpr(std::is_base_of<DistributionBase, stddev_ty>::value)
        //     {
        //         score_val += stddev.score();
        //     }

        //     auto log_prob = log_expression.value();
        //     if constexpr(is_array_std<decltype(log_prob)>::value)
        //     {
        //         for(auto i : log_prob)
        //         {
        //             score_val += i;
        //         }
        //     }
        //     else
        //     {
        //         score_val += log_prob;
        //     }
        //     // score_val += log_prob;
        //     return score_val;
        // }




        template <typename Tc = T, typename std::enable_if<std::is_arithmetic<Tc>::value>::type* = nullptr> inline
        Tc sample()
        {
            Tc mu_value, stddev_value;
            if constexpr(std::is_base_of<ADBase, mu_ty>::value) {
                mu_value = Tc{mu.value()};
            }
            if constexpr(std::is_base_of<ADBase, stddev_ty>::value) {
                stddev_value = Tc{stddev.value()};
            }
            if constexpr(std::is_base_of<DistributionBase, mu_ty>::value)
            {
                mu_value = Tc{mu.sample()};
            }
            if constexpr(std::is_base_of<DistributionBase, stddev_ty>::value)
            {
                stddev_value = Tc{stddev.sample()};
            }
            float r1 = distribution(gen);
            float r2 = distribution(gen);
            float r3 = distribution(gen);

            auto sum = r1 + r2 + r3;
            sum = sum * 2;
            auto shifted_mean = sum - 3;
            return (shifted_mean *stddev_value) + mu_value;
        }
        // Tc sample()
        // {
        //     Tc u,v,s;

        //     Tc mu_value, stddev_value;
        //     if constexpr(std::is_base_of<ADBase, mu_ty>::value) {
        //         mu_value = Tc{mu.value()};
        //     }
        //     if constexpr(std::is_base_of<ADBase, stddev_ty>::value) {
        //         stddev_value = Tc{stddev.value()};
        //     }
        //     if constexpr(std::is_base_of<DistributionBase, mu_ty>::value)
        //     {
        //         mu_value = Tc{mu.sample()};
        //     }
        //     if constexpr(std::is_base_of<DistributionBase, stddev_ty>::value)
        //     {
        //         stddev_value = Tc{stddev.sample()};
        //     }
        //     std::normal_distribution<float> distr_normal(mu_value, stddev_value);
        //     return distr_normal(gen);
        // }

        template < typename Tc = T, typename std::enable_if<std::is_base_of<fpm::fixedpoint_base, Tc>::value>::type* = nullptr> inline
    Tc sample()
    {
      auto sign_and_int_bits = Tc::integer_bits;

      Tc mu_value, stddev_value;
      if constexpr(std::is_base_of<ADBase, mu_ty>::value) {
        mu_value = Tc{mu.value()};
      }
      if constexpr(std::is_base_of<ADBase, stddev_ty>::value) {
        stddev_value = Tc{stddev.value()};
      }
      if constexpr(std::is_base_of<DistributionBase, mu_ty>::value)
      {
        mu_value = Tc{mu.sample()};
      }
      if constexpr(std::is_base_of<DistributionBase, stddev_ty>::value)
      {
        stddev_value = Tc{stddev.sample()};
      }

      uint32_t random32_1 = gen();
      random32_1 = random32_1 << sign_and_int_bits;
      random32_1 = random32_1 >> sign_and_int_bits;

      uint32_t random32_2 = gen();
      random32_2 = random32_2 << sign_and_int_bits;
      random32_2 = random32_2 >> sign_and_int_bits;

      uint32_t random32_3 = gen();
      random32_3 = random32_3 << sign_and_int_bits;
      random32_3 = random32_3 >> sign_and_int_bits;

    //   uint32_t random32_4 = gen();
    //   random32_4 = random32_4 << sign_and_int_bits;
    //   random32_4 = random32_4 >> sign_and_int_bits;

    //   uint32_t random32_5 = gen();
    //   random32_5 = random32_5 << sign_and_int_bits;
    //   random32_5 = random32_5 >> sign_and_int_bits;

    //   uint32_t random32_6 = gen();
    //   random32_6 = random32_6 << sign_and_int_bits;
    //   random32_6 = random32_6 >> sign_and_int_bits;

      auto sum = random32_1 + random32_2 + random32_3 ;
      sum = sum << 1;

      auto shifted_mean = Tc::from_raw_value(sum) - Tc(3);
      return (shifted_mean *stddev_value) + mu_value;


        // float r1 = distribution(gen);
        // float r2 = distribution(gen);
        // float r3 = distribution(gen);

        // auto sum = r1 + r2 + r3;
        // sum = sum * 2;
        // auto shifted_mean = sum - 3;
        // return (Tc) ((shifted_mean *(float)stddev_value) + (float)mu_value);
    }

    template<int N>
    auto samples()
    {
        std::array<T, N> samples_;
        for(int i = 0; i < N; i++)
        {
            samples_[i] = sample();
            // samples.push_back(sample());
        }
        return samples_;
    }




    // inline void set(T mu_in, T stddev_in)
    // {
    //   mu.set_value(mu_in);
    //   std_dev.set_value(stddev_in);
    // }

    // void set_input(T input_)
    // {
    //   std::cout << "value to be inserted: " << input_ << "\n";
    //   x.set_value(input_);
    // }

    // void set_input(Constant<T> & input)
    // {
    //   x.set_value(input.value());
    // }

    // void set_input(Variable<T> & input)
    // {
    //   x.set_value(input.value());
    // }

    // inline void set_mu(T mu_in)
    // {
    //   mu.set_value(mu_in);
    // }

    // inline void set_stddev(T stddev_in)
    // {
    //   std_dev.set_value(stddev_in);
    // }

    // void see_input()
    // {
    //     std::cout << x.value() <<"\n";
    // }

    // void see_params()
    // {
    //   std::cout << "Params\n";
    //   std::cout << "Input: " << x.value() << "\n";
    //   std::cout << "mu: " << mu.value() << "\n";
    //   std::cout << "sd: " << std_dev.value() << "\n";
    // }
    
};



// Need to add universal forwarding

template<typename T, typename mu_ty, typename stddev_ty, typename input_ty>
inline auto normal_dist(mu_ty& mu_in, stddev_ty& stddev_in, input_ty& x_in){

  using main_ty = T;
  using exp_type = decltype(main_ty{0.3989422804014327} * (main_ty{1}/stddev_in) * exp(main_ty{-0.5} * (x_in - mu_in) * (x_in - mu_in) / (stddev_in * stddev_in)));
  using log_type = decltype(main_ty{-0.9189385332046727}  - log(stddev_in) - main_ty{0.5} * (main_ty{1}/stddev_in) *( main_ty{1}/stddev_in) * (x_in - mu_in) * (x_in - mu_in));

  using test_type = decltype((main_ty{1}/stddev_in) *( main_ty{1}/stddev_in) * (mu_in) );
                            //   run_ty{-0.9189385332046727} - log(stddev) - run_ty{0.5} * (run_ty{1}/stddev) * (run_ty{1}/stddev) * (x_in - mu) * (x_in - mu);
//   using log_type = decltype(main_ty{-0.9189385332046727}  - log(stddev_in) - (main_ty{0.5}/(stddev_in * stddev_in)) * (x_in - mu_in) * (x_in - mu_in));

  return Normal<mu_ty, stddev_ty,input_ty, T,exp_type, log_type>(mu_in, stddev_in,x_in);
}


template<typename T, typename lb_ty, typename ub_ty, typename input_ty >
inline auto uniform_dist(lb_ty& lb_in, ub_ty& ub_in, input_ty& x_in){
  using main_ty = T;
  using exp_type = decltype(main_ty{1} / (ub_in - lb_in));
  using log_type = decltype(main_ty{0}-log(ub_in - lb_in));

  return Uniform<lb_ty, ub_ty,input_ty, T,exp_type, log_type>(lb_in, ub_in,x_in);
}


template<typename opt_ty, typename joint_type, typename variational_type>
typename joint_type::type_ elbo_one_latent(joint_type& joint, variational_type& variational, int num_samples = 5)
{
    opt_ty elbo_val = 0;
    for(int i = 0; i < num_samples; i++)
    {
        variational(variational.sample());
        elbo_val += opt_ty{joint.score()} - opt_ty{variational.score()};
    }
    return elbo_val / opt_ty{num_samples};
}


template<typename run_ty>
run_ty only_positive(run_ty x)
{
    if(x < run_ty{0})
    {
        // abort();
        return run_ty{0.01};
    }
    else
    {
        return x;
    }
}

template <typename T>
T hard_lb(T x, T lb)
{
    if(x < lb)
    {
        return lb;
    }
    else
    {
        return x;
    }
}

template<typename run_ty>
run_ty grad_clipping(run_ty x, run_ty LB, run_ty UB)
{
    if(x < LB)
    {
        return LB;
    }
    else if(x > UB)
    {
        return UB;
    }
    else
    {
        return x;
    }
}


template<typename K, int EARLY_STOPPING, typename std::enable_if_t<std::is_arithmetic<K>::value>* = nullptr>
K average_diff(std::array<K, EARLY_STOPPING>& arr)
{
    K diff_sum = 0;
    for(int i = 0; i < EARLY_STOPPING -1; i++)
    {
        diff_sum += std::abs(arr[i] - arr[i+1]);
    }
    return diff_sum / EARLY_STOPPING;
}


template<typename K, int EARLY_STOPPING, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, K>::value>* = nullptr>
K average_diff(std::array<K, EARLY_STOPPING>& arr)
{
    K diff_sum = (K) 0;
    for(int i = 0; i < EARLY_STOPPING -1; i++)
    {
        diff_sum += fpm::abs(arr[i] - arr[i+1]);
    }
    return diff_sum;
}

template<typename K, int EARLY_STOPPING, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, K>::value>* = nullptr>
K change_in_consecutive(std::array<K, EARLY_STOPPING>& arr)
{
   return fpm::abs((arr[0] - arr[1]));
    // return diff_sum;
}




template<typename K = float, int N>
std::array<float, N> linspace(K start, K end)
{
  std::array<float, N> linspaced;

  float delta = (end - start) / (N - 1);

  for(int i=0; i < N-1; ++i)
    {
      linspaced[i] = (start + delta * i);
    }
  linspaced[N-1] = (end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}


template<int BITSHIFT = 0, typename D, typename cont_ty, std::enable_if_t<BITSHIFT == 0> * = nullptr>
typename D::type_ total_score_w_data(D& dist, cont_ty& data)
{
    typename D::type_ score = 0;
    for(auto& d : data)
    {
        dist(d);
        score += dist.log_pdf().value();
        // std::cout << "Score: " << score << std::endl;
    }
    return score;
}



template<int BITSHIFT = 0, typename D, typename cont_ty, std::enable_if_t<BITSHIFT != 0> * = nullptr>
typename D::type_ total_score_w_data(D& dist, cont_ty& data)
{
    typename D::type_ score = 0;
    using M = typename D::type_;
    for(auto& d : data)
    {
        dist(d);
        score += M::from_raw_value(dist.log_pdf().value().raw_value() >> BITSHIFT);
    }
    return score;
}

template<int EARLY_STOPPING, typename T>
void inline store_past_values(int iter, std::array<T, EARLY_STOPPING> &arr, T container)
{
    arr[0] = arr[1];
    arr[1] = container;
}

template<typename T>
void all_type_info()
{
    std::cout << "Max: " << std::numeric_limits<T>::max() << std::endl;
    std::cout << "Min: " << std::numeric_limits<T>::min() << std::endl;
    std::cout << "Epsilon: " << std::numeric_limits<T>::epsilon() << std::endl;
}



template<typename T>
float rmse(T &inferred, T &target)
{
    float temp = 0;
    for(int i = 0; i < target.size(); i++)
    {
        temp += (inferred[i] - target[i]) * (inferred[i] - target[i]);
    }
    return std::sqrt(temp / target.size());
}




template<typename T>
float rmse(T &inferred, T &target, int BURNIN)
{
    float temp = 0;
    for(int i = BURNIN; i < target.size(); i++)
    {
        temp += (inferred[i] - target[i]) * (inferred[i] - target[i]);
    }
    return std::sqrt(temp / target.size());
}

// geomean of relative error
template<typename T>
float geomean(T &inferred, T &target)
{
    float temp = 0;
    for(int i = 0; i < target.size(); i++)
    {
        temp += std::log(std::abs(inferred[i] - target[i]) / target[i]);
    }
    return std::exp(temp / target.size());
}

template<typename T>
float geomean(T &inferred, T &target, int BURNIN)
{
    float temp = 0;
    for(int i = BURNIN; i < target.size(); i++)
    {
        temp += std::log(std::abs(inferred[i] - target[i]) / target[i]);
    }
    return std::exp(temp / target.size());
}

template<typename T>
class Results{
    public:
    T mean;
    T sigma;
    T elbo;
// 
    Results(T mean, T sigma, T elbo){
        this->mean = mean;
        this->sigma = sigma;
        this->elbo = elbo;
    }
};