#pragma once
// #include "ad_engine.hpp"
#include<vector>
#include <random>
#include "distributions_latest.hpp"



template <typename mu_ty, typename stddev_ty,typename input_ty, typename EXPRESSION_TYPE,  typename LOG_TYPE, typename RefTy = ByRefClass>
class NormalInterval : public DistributionBase , ADBase{
    mu_ty& mu;
    stddev_ty& stddev;
    input_ty& x_in; // Data

   // typename mu_ty::type_ mu_value;
   // typename stddev_ty::type_ stddev_value;

    // EXPRESSION_TYPE expression = 0.3989422804014327 * (1/stddev) * exp(-0.5 * (x_in - mu) * (x_in - mu) / (stddev * stddev));
    EXPRESSION_TYPE expression = float{0.3989422804014327}* (float{1}/stddev) * exp(float{-0.5} * (x_in - mu) * (x_in - mu) / (stddev * stddev));

    // NEED TO USE THIS LOG EXPRESSION INSTEAD (note 1/stddev^2 is written out differently, 
    //that is because for small values of stddev fpm was rounding it to 0 and throwing error. So, I have written it out explicitly)
    LOG_TYPE log_expression = float{-0.9189385332046727}  - log(stddev) - float{0.5} * (float{1}/stddev) * (float{1}/stddev) * (x_in - mu) * (x_in - mu);

    public:
        using T = interval;
        using type_ = T;
        using ref_type = RefTy;

        NormalInterval(mu_ty& mu, stddev_ty& stddev, input_ty& x_in) : mu(mu), stddev(stddev), x_in(x_in)
        {

        }
  
        auto inline pdf()
        {
            return expression;
        }
  
        auto log_pdf()
        {
            return log_expression;
        }



        interval value() const
        {
            return x_in.value();
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


        T diff_log(int wrt) const
        {
	    return T{log_expression.diff(wrt)};
        }


        template<typename diff_ty>
        T diff_log(Variable<diff_ty> &wrt_diff)
        {
            return this->diff_log(wrt_diff.Var_ID);
        }


        interval sample()
        {
			float mu_l = mu.value().lb;
			float mu_u = mu.value().ub;
			float sigma_u = stddev.value().ub;
			return interval(mu_l-3*sigma_u,mu_u+3*sigma_u);
        }


    
};

// Need to add universal forwarding


template<typename mu_ty, typename stddev_ty, typename input_ty >
inline auto normal_dist_INTERVAL(mu_ty& mu_in, stddev_ty& stddev_in, input_ty& x_in){

  using main_ty = float;
  using exp_type = decltype(main_ty{0.3989422804014327} * (main_ty{1}/stddev_in) * exp(main_ty{-0.5} * (x_in - mu_in) * (x_in - mu_in) / (stddev_in * stddev_in)));
  using log_type = decltype(main_ty{-0.9189385332046727}  - log(stddev_in) - main_ty{0.5} * (main_ty{1}/stddev_in) *( main_ty{1}/stddev_in) * (x_in - mu_in) * (x_in - mu_in));
 
  return NormalInterval<mu_ty, stddev_ty,input_ty, exp_type, log_type>(mu_in, stddev_in,x_in);
}



template <typename lb_ty, typename ub_ty,typename input_ty, typename EXPRESSION_TYPE,  typename LOG_TYPE, typename RefTy = ByRefClass>
class UniformInterval : public DistributionBase , ADBase{
    lb_ty& lb;
    ub_ty& ub;
    input_ty& x_in; // Data


   // typename mu_ty::type_ mu_value;
   // typename stddev_ty::type_ stddev_value;

    EXPRESSION_TYPE expression = float{1} / (ub - lb);
    
    LOG_TYPE log_expression = float{0} -log(ub - lb);

    public:
        using T = interval;
        using type_ = T;
        using ref_type = RefTy;

        UniformInterval(lb_ty& lb, ub_ty& ub, input_ty& x_in) : lb(lb), ub(ub), x_in(x_in)
        {

        }
  
        auto inline pdf()
        {
            return expression;
        }
  
        auto log_pdf()
        {
            return log_expression;
        }



        interval value() const
        {
            return x_in.value();
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


        T diff_log(int wrt) const
        {
	    return T{log_expression.diff(wrt)};
        }


        template<typename diff_ty>
        T diff_log(Variable<diff_ty> &wrt_diff)
        {
            return this->diff_log(wrt_diff.Var_ID);
        }


        interval sample()
        {
			float lower = lb.value().lb;
			float upper = ub.value().ub;
			return interval(lower,upper);
        }


    
};



template<typename lb_ty, typename ub_ty, typename input_ty >
inline auto uniform_dist_INTERVAL(lb_ty& lb, ub_ty& ub, input_ty& x_in){

  using main_ty = float;
  using exp_type = decltype(main_ty(1) / (ub - lb));
  using log_type = decltype(main_ty(0) -log(ub - lb));
 
  return UniformInterval<lb_ty, ub_ty,input_ty, exp_type, log_type>(lb, ub, x_in);
}



Constant<interval> max_with_0(interval x)
{
    return Constant<interval>(interval(std::max(x.lb,0.0f),std::max(x.ub,0.0f)));
}