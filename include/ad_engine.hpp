#include <cmath>
#include <type_traits>
#include "fixed.hpp"
#include "math.hpp"
#include <array>
#include <iostream>


constexpr int counter = 0;

 int var_id_counter = 0;


class ADBase{}; //

template <typename M, typename T>
class Node2 {
  public:
    using type_ = T;
    using child_type = M;
    constexpr const M& cast() const {return static_cast<const M&>(*this);}
    constexpr type_ value() const {return cast().value();}
};


class Intervaldummy{};

class ByRefClass{};
class ByValueClass{}; // Introduced for Constant management

template<typename Child_type, typename ret_type>
class Node_INT{
  public:
  constexpr const Child_type& cast() const {return static_cast<const Child_type&>(*this);}
  constexpr ret_type value() const {return cast().value(); } // making copy of everything for now - lets see if this needs to be changed.
};

// Is it forward or reverse LOL 
class AD_OP_Base{};

class Primitive{};

template<typename T=float> // Does not need to be templatised tbh. 
class Interval : public Intervaldummy{
  
  public:
  // using type_ = T;
  T lb;
  T ub;
  bool has_zero;
  Interval() {}
  Interval(T lb_in, T ub_in): lb(lb_in), ub(ub_in){
    if(lb <= T{0} && ub >= T{0}){
      has_zero = true;
    }
    else
    {
      has_zero = false;
    }
  }

  void see()
  {
      std::cout << "[ " << lb << ", " << ub << " ] \n";
  }

  Interval<T> value () const // making copy of everything for now lets see if this needs to be changed.
  {
    return *this;
  }
  // friend std::ostream& operator<<(std::ostream& os, const Interval<T>& it);
};
using interval=Interval<float>; //hacky fix

// factory method 
template<typename T=float>
Interval<T> make_interval(float lb, float ub) //more hacky fixes
{
  if(lb>ub)
  {
    // std::cout << "exiting: invalid interval\n";
    exit(EXIT_FAILURE);
  }
  return Interval<T>(lb, ub);
}






// template<class T>
// struct is_array_std {
//     static constexpr bool value = false;
// };

// template<class T>
// struct is_array_std<std::vector<T>> {
//     static constexpr bool value =
//         std::is_arithmetic<T>::value ||
//         std::is_base_of<fpm::fixedpoint_base, T>::value;
// };


// template<class T>
// struct is_array_std{
//     static constexpr bool value = false;
// };

// template<class T, int N>
// struct is_array_std<std::array<T, N>> {
//     static constexpr bool value =
//         std::is_arithmetic<T>::value ||
//         std::is_base_of<fpm::fixedpoint_base, T>::value;
// };

template<typename T>
struct is_array_std : std::false_type {};

template<typename T, std::size_t N>
struct is_array_std<std::array<T, N>> : std::true_type {};

template<typename T>
struct is_float{
  public:
  static constexpr bool value = std::is_floating_point<T>::value;
};


template<class T>
struct is_interval {
    static constexpr bool value = std::is_same<T , Interval<float>>::value;
};

// template<class T>
// struct is_interval<std::vector<T>> {
//     static constexpr bool value =
//         std::is_arithmetic<T>::value ||
//         std::is_base_of<fpm::fixedpoint_base, T>::value;
// };

template <typename T, typename RefTy = ByRefClass>
struct Variable : public ADBase, Intervaldummy, Primitive{

  public:
    using type_ = T;
    using ref_type = RefTy;
    mutable type_ value_;
    const int Var_ID;
    Interval<float> ones;
    Interval<float> zeros;
    type_ ones_array;
    type_ zeros_array;
    size_t size;


    constexpr Variable ():value_(0), Var_ID(var_id_counter++){}

    constexpr Variable( const type_ val_):value_(val_), Var_ID(var_id_counter++)
    {
      if constexpr (std::is_same<T, Interval<float>>::value)
      {
        ones = make_interval<float>(1,1);
        zeros = make_interval<float>(0,0);
      }

      if constexpr (is_array_std<type_>::value)
      {
        // ones_vec = std::vector<typename T::value_type>(val_.size(), 1);
        // zeros_vec = std::vector<typename T::value_type>(val_.size(), 0);
        this->size = value_.size();
        // std::cout << "size: " << size << "\n";
        for(int i = 0; i < this->size; i++)
        {
          ones_array[i] = 1;
        }
        for(int i = 0; i < this->size; i++)
        {
          zeros_array[i] = 0;
        }
      }
    }

    constexpr Variable(const float v1, const float v2):value_(make_interval<float>(v1,v2)), Var_ID(var_id_counter++)
    {
      if constexpr (std::is_same<T, Interval<float>>::value)
      {
        ones = make_interval<float>(1,1);
        zeros = make_interval<float>(0,0);
      }
    }
    
    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    void operator()(type_& vec)
    {
      // if constexpr(is_array_std<type_>::value)
      // {
        for(int i = 0; i < this->size; i++)
        {
          this->value_[i] = vec[i];
        }
      // }
      // else
      // {
      //   this->value_ = vec;
      // }
      // this->value_ = vec;
    }


    template<typename Tc = type_, typename std::enable_if_t< !is_array_std<Tc>::value && !std::is_same<Tc, Interval<float>>::value && (  std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    void operator()(type_ val)
    {
      this->value_ = val;
    }

    // Non-vector value functions
    template<typename Tc = type_, typename std::enable_if_t< !is_array_std<Tc>::value && ( std::is_same<Tc, Interval<float>>::value || std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    constexpr type_ value() const
    {
      // std::cout <<" value in variable : " << value_ << "\n";
      return value_;
    }

    // Vector value functions
    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    constexpr type_& value() const
    {
      // std::cout <<" value in variable : " << value_ << "\n";
      return value_;
    }

    // constexpr void set_value(type_ in_)
    // {
    //   // std::cout << "variable set value \n";
    //     value_ = in_;
    // }

    template<typename Tc = type_, typename std::enable_if_t< (std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    void set_value(type_ in_)
    {
      // std::cout << "constant set value : " << in_  << "\n";
      value_ = in_;
    }

    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    void set_value(type_& in_)
    {
      for(size_t i = 0; i < this->size ; i++)
      {
        value_[i] = in_[i];
      }
    }
    // typename std::enable_if_t<std::is_base_of<Intervaldummy, T>::value>* = nullptr
    template<typename Tc = type_, typename std::enable_if_t< (std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    constexpr type_ diff(int wrt_v) const
    {
      if(wrt_v == Var_ID)
      {
        return type_{1};
      }
      else
      {
        return type_{0};
      }
    }

    template<typename Tc = type_, typename std::enable_if_t<std::is_same<Tc, Interval<float>>::value>* = nullptr>
    constexpr type_ diff(int wrt_v) const
    {
      if(wrt_v == Var_ID)
      {
        return ones;
      }
      else
      {
        return zeros;
      }
    }

    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    constexpr const type_& diff(int wrt_v) const
    {
      if(wrt_v == Var_ID)
      {
        return ones_array;
      }
      else
      {
        return zeros_array;
      }
    }
};

template <typename T, typename K = ByRefClass>
struct Constant : public ADBase, Intervaldummy, Primitive{

  public:
    using type_ = T;
    using ref_type = K;
    mutable type_ value_;
    type_ zeros_array;
    Interval<float> zeros;
    size_t size;
    constexpr Constant ():value_(0){}
    constexpr Constant(type_ val_): value_(val_) {
    if constexpr (std::is_same<T, Interval<float>>::value)
    {
      zeros = make_interval<float>(0,0);
    }

    // if constexpr (is_array_std<type_>::value)
    // {
    //   this->size = val_.size();
    //   zeros_vec = std::vector<typename T::value_type>(val_.size(), 0);
    // }

    if constexpr (is_array_std<type_>::value)
      {
        // ones_vec = std::vector<typename T::value_type>(val_.size(), 1);
        // zeros_vec = std::vector<typename T::value_type>(val_.size(), 0);
        this->size = value_.size();
        // std::cout << "size: " << size << "\n";
        for(int i = 0; i < this->size; i++)
        {
          zeros_array[i] = 0;
        }
      }
    }
    constexpr Constant(const float v1, const float v2):value_(make_interval<float>(v1,v2)){}
    
    void operator()(type_ vec)
    {
      if constexpr(is_array_std<type_>::value)
      {
        for(int i = 0; i < this->size; i++)
        {
          this->value_[i] = vec[i];
        }
      }
      else
      {
        this->value_ = vec;
      }
    }

    template<typename Tc = type_, typename std::enable_if_t< ( std::is_same<Tc, Interval<float>>::value || std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    constexpr type_ value() const
    {
      return value_;
    }

    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    constexpr type_& value() const
    {
      return value_;
    }

    template<typename Tc = type_, typename std::enable_if_t< (std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    constexpr type_ diff(int wrt_v) const
    {
      return type_{0};
    }

    template<typename Tc = type_, typename std::enable_if_t<std::is_same<Tc, Interval<float>>::value>* = nullptr>
    constexpr type_ diff(int wrt_v) const
    {
      return zeros;
    }

    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    constexpr const type_& diff(int wrt_v) const
    {
      return zeros_array;
    }

    template<typename Tc = type_, typename std::enable_if_t< (std::is_arithmetic<Tc>::value || std::is_base_of<fpm::fixedpoint_base, Tc>::value)  >* = nullptr>
    void set_value(type_ in_)
    {
      // std::cout << "constant set value : " << in_  << "\n";
      value_ = in_;
    }

    // template<typename Tc = type_, typename std::enable_if_t<std::is_same<Tc, std::vector<typename Tc::value_type> >::value>* = nullptr>
    // void set_value(type_ in_)
    // {
    //   for(size_t i = 0; i < in_.size(); i++)
    //   {
    //     value_.at(i) = in_.at(i);
    //   }
    // }

    template<typename Tc = type_, typename std::enable_if_t<is_array_std<Tc>::value>* = nullptr>
    void set_value(type_& in_)
    {
      for(size_t i = 0; i < this->size ; i++)
      {
        value_[i] = in_[i];
      }
    }
};

auto invert(Interval<float> a)
{
  return make_interval<float>(1/a.ub, 1/a.lb);
}

Interval<float> add(Interval<float> &i1, Interval<float> &i2) // Instant result functions - do not create any expression graph.
{
  float lb = i1.lb + i2.lb;
  float ub = i1.ub + i2.ub;
  return make_interval(lb, ub);
}

Interval<float> sub(Interval<float> &i1, Interval<float> &i2)
{
  float lb = i1.lb - i2.ub;
  float ub = i1.ub - i2.lb;
  return make_interval(lb, ub);
}

Interval<float> mul(Interval<float> &i1, Interval<float> &i2)
{
  float lb = std::min({i1.lb*i2.lb, i1.lb*i2.ub, i1.ub*i2.lb, i1.ub*i2.ub});
  float ub = std::max({i1.lb*i2.lb, i1.lb*i2.ub, i1.ub*i2.lb, i1.ub*i2.ub});
  // float ub = i1.ub * i2.ub;
  return make_interval(lb, ub);
}

Interval<float> div(Interval<float> &l_temp, Interval<float> &r_temp)
{
  // const Interval<float> l_temp = l_param.value(); 
  //   const Interval<float> r_temp = r_param.value();
    if(r_temp.has_zero){
      auto answer = make_interval<float>(-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());

      return answer;
    }
    else{
      auto invert_r = invert(r_temp);
      // std::cout << "Check invert: ";
      // invert_r.see(); 
      // auto answer = make_interval<float>(l_temp.lb * invert_r.lb, l_temp.ub * invert_r.ub);
      float lb = std::min({l_temp.lb * invert_r.lb, l_temp.lb * invert_r.ub, l_temp.ub * invert_r.lb, l_temp.ub * invert_r.ub});
      float ub = std::max({l_temp.lb * invert_r.lb, l_temp.lb * invert_r.ub, l_temp.ub * invert_r.lb, l_temp.ub * invert_r.ub});
      return make_interval<float>(lb,ub);
    }
}

template<typename T,typename std::enable_if_t<std::is_base_of<Intervaldummy, T>::value>* = nullptr >
Interval<float> exp_instant(const T &i1)
{
  auto curr_interval = i1.value();
  float lb = std::exp(curr_interval.lb);
  float ub = std::exp(curr_interval.ub);
  return make_interval(lb, ub);
}

template<typename T,typename std::enable_if_t<std::is_base_of<Intervaldummy, T>::value>* = nullptr >
Interval<float> sqrt_instant(const T &i1)
{
  auto curr_interval = i1.value();
  float lb = std::sqrt(curr_interval.lb);
  float ub = std::sqrt(curr_interval.ub);
  return make_interval(lb, ub);
}

template<typename T,typename std::enable_if_t<std::is_base_of<Intervaldummy, T>::value>* = nullptr >
Interval<float> log_instant(const T& i1)
{
  float lb = std::log(i1.lb);
  float ub = std::log(i1.ub);
  return make_interval(lb, ub);
}

// log10 of an interval
template<typename T,typename std::enable_if_t<std::is_base_of<Intervaldummy, T>::value>* = nullptr >
Interval<float> log10_instant(const T& i1)
{
  float lb = std::log10(i1.lb);
  float ub = std::log10(i1.ub);
  return make_interval(lb, ub);
}

template<typename L, typename R, typename RefTy = ByValueClass>
struct ADD_INT: public AD_OP_Base, Intervaldummy {

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

  Ltype L_int;
  Rtype R_int;

  public:
  using type_ = Interval<float>;
  using ref_type = RefTy;

  ADD_INT( Ltype Lin,  Rtype Rin) : L_int{Lin}, R_int{Rin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
    auto l_temp = L_int.value();
    auto r_temp = R_int.value();
    auto answer = add(l_temp, r_temp);
    return answer;
  }

  constexpr type_ diff(int wrt) const
    {
      auto l_diff = L_int.diff(wrt);
      auto r_diff = R_int.diff(wrt);
      return add(l_diff, r_diff);
    }

    template<typename DIFFTY>
    constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};

Interval<float> negate(const Interval<float>& a)
{
  return make_interval(-a.ub, -a.lb);
}

template<typename L, typename R, typename RefTy = ByValueClass>
struct SUB_INT: public AD_OP_Base, Intervaldummy {

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass, typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

  Ltype L_int;
  Rtype R_int;

  public:
  // using type_ = typename L::type_;
  using type_ = Interval<float>;
  using ref_type = RefTy;
  SUB_INT( Ltype Lin,  Rtype Rin) : L_int{Lin}, R_int{Rin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
     Interval<float> l_temp = L_int.value(); 
    
    Interval<float> r_temp = R_int.value(); 
    auto answer = sub(l_temp, r_temp);
    return answer;
  }
  constexpr type_ diff(int wrt) const
  {
    auto l_diff = L_int.diff(wrt);
    auto r_diff = R_int.diff(wrt);
    return sub(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }
};

template<typename L, typename R, typename RefTy = ByValueClass>
struct MUL_INT: public AD_OP_Base, Intervaldummy {

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

  Ltype L_int;
  Rtype R_int;

  public:
  // using type_ = typename L::type_;
   using type_ = Interval<float>;
   using ref_type = RefTy;
  MUL_INT( Ltype Lin,  Rtype Rin) : L_int{Lin}, R_int{Rin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
    Interval<float> l_temp = L_int.value(); 
    
    Interval<float> r_temp = R_int.value(); 
    auto answer = mul(l_temp, r_temp);
    return answer;
  }

  constexpr type_ diff(int wrt) const
  {
    auto l_diff = L_int.diff(wrt);
    auto l_value = L_int.value();
    auto r_diff = R_int.diff(wrt);
    auto r_value = R_int.value();

    auto pt_1 = mul(l_diff, r_value);
    auto pt_2 = mul(l_value, r_diff);
    return add(pt_1,pt_2 ); // Will need to add universal forwarding to this to fix it.
    // return mul(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }
};


template<typename L, typename R, typename RefTy = ByValueClass>
struct DIV_INT: public AD_OP_Base, Intervaldummy {

   typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

  Ltype L_int; // NUM
  Rtype R_int; // DENOM

  public:
  // using type_ = typename L::type_;
   using type_ = Interval<float>;
   using ref_type = RefTy;
  DIV_INT( Ltype Lin,  Rtype Rin) : L_int{Lin}, R_int{Rin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
  // std::cout << "check\n";
    Interval<float> l_temp = L_int.value(); 
    Interval<float> r_temp = R_int.value();
    auto answer = div(l_temp, r_temp);
    return answer;
  }

  constexpr type_ diff(int wrt) const
  {
    auto num_diff = L_int.diff(wrt);
    // std::cout << "num_diff: "; num_diff.value().see();
    auto num_value = L_int.value();
    // std::cout << "num_value: "; num_value.value().see();
    auto den_diff = R_int.diff(wrt);
    // std::cout << "den_diff: "; den_diff.value().see();
    auto den_value = R_int.value();
    // std::cout << "den_value: "; den_value.value().see();

    auto pt_1 = mul(den_value, num_diff);

    // std::cout << "gx . f'x = "; pt_1.value().see();
    auto pt_2 = mul(num_value, den_diff);
    // std::cout << "g'x . fx = "; pt_2.value().see();
    auto final_num = sub(pt_1, pt_2);
    // std::cout << "final num: "; final_num.value().see();
    auto final_denom = mul(den_value, den_value);
    // std::cout << "final denom: "; final_denom.value().see();

    auto answer = div(final_num, final_denom);
    return answer; // Will need to add universal forwarding to this to fix it.
    // return mul(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }

};

template<typename L,typename RefTy = ByValueClass>
struct EXP_INT: public AD_OP_Base, Intervaldummy {
  
  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
  const L& >::type Ltype;

  Ltype L_int;

  public:
  // using type_ = typename L::type_;
   using type_ = Interval<float>;
   using ref_type =RefTy;
  EXP_INT( Ltype Lin) : L_int{Lin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
  // std::cout << "check\n";
    Interval<float> l_temp = L_int.value();
    auto answer = exp_instant(l_temp);
    return answer;
  }

  constexpr type_ diff(int wrt) const
  {
    auto x = L_int.diff(wrt);
    // x.see();
    auto temp = L_int.value();
    auto e = exp_instant(temp);
    auto answ = mul(e,x);
    // e.see();
    return answ; // Will need to add universal forwarding to this to fix it.
    // return mul(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }
};


template<typename L,typename RefTy = ByValueClass>
struct LOG_INT: public AD_OP_Base, Intervaldummy {
  
  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
  const L& >::type Ltype;

  Ltype L_int;

  public:
   using type_ = Interval<float>;
   using ref_type =RefTy;
  LOG_INT( Ltype Lin) : L_int{Lin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
  // std::cout << "check\n";
    auto answer = log_instant(L_int.value());
    return answer;
  }
  constexpr type_ diff(int wrt) const
  {
    auto pt_1 = invert(L_int.value());
    auto pt_2 = L_int.diff(wrt);
    auto answer = mul(pt_1, pt_2);
    return answer; // Will need to add universal forwarding to this to fix it.
    // return mul(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }
};



template<typename L,typename RefTy = ByValueClass>
struct LOG10_INT: public AD_OP_Base, Intervaldummy {
  
  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
  const L& >::type Ltype;

  Ltype L_int;

  public:
  // using type_ = typename L::type_;
   using type_ = Interval<float>;
   using ref_type =RefTy;
  LOG10_INT( Ltype Lin) : L_int{Lin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
  // std::cout << "check\n";
    Interval<float> l_temp = L_int.value();
    auto answer = log10_instant(l_temp);
    return answer;
  }

  constexpr type_ diff(int wrt) const
  {
    auto curr_int = L_int.value();
    auto pt_1 = invert(curr_int);
    auto pt_2 = L_int.diff(wrt);
    auto answer = mul(pt_1, pt_2);
    return answer; // Will need to add universal forwarding to this to fix it.
    // return mul(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }
};


template<typename L,typename RefTy = ByValueClass>
struct SQRT_INT: public AD_OP_Base, Intervaldummy {
  
  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
  const L& >::type Ltype;

  Ltype L_int;

  public:
  // using type_ = typename L::type_;
   using type_ = Interval<float>;
   using ref_type =RefTy;
  SQRT_INT( Ltype Lin) : L_int{Lin} {}

  Interval<float> value() const{ // Little ineffecient but can be fixed later.
  // std::cout << "check\n";
    Interval<float> l_temp = L_int.value();
    auto answer = sqrt_instant(l_temp);
    return answer;
  }

  constexpr type_ diff(int wrt) const
  {
    auto curr_int = L_int.value();
    auto sqrt_curr = sqrt_instant(curr_int);
    auto pt_1 = invert(sqrt_curr);
    auto half = Interval<float>(0.5,0.5);
    auto pt_2 = L_int.diff(wrt);
    auto first_part = mul(half, pt_1);
    auto answer = mul(first_part, pt_2);
    return answer; // Will need to add universal forwarding to this to fix it.
    // return mul(l_diff, r_diff);
  }

  template<typename DIFFTY>
  constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
  {
    return this->diff(wrt_diff.Var_ID);
  }
};

template<typename T, typename std::enable_if_t<std::is_same<typename T::type_, Interval<float>>::value>* = nullptr >
auto exp(const T& input) // Will add to tree - Lazy evaluation.
{
  return EXP_INT<T>(input);
}

template<typename T, typename std::enable_if_t<std::is_same<typename T::type_, Interval<float>>::value>* = nullptr >
auto sqrt(const T& input) // Will add to tree - Lazy evaluation.
{
  return SQRT_INT<T>(input);
}



//log of a number
template<typename T, typename std::enable_if_t<std::is_same<typename T::type_, Interval<float>>::value>* = nullptr >
auto log(const T& input) // Will add to tree - Lazy evaluation.
{
  return LOG_INT<T>(input);
}

//log10 of a number
template<typename T, typename std::enable_if_t<std::is_same<typename T::type_, Interval<float>>::value>* = nullptr >
auto log10(const T& input) // Will add to tree - Lazy evaluation.
{
  return LOG10_INT<T>(input);
}

template<typename L, typename R, typename std::enable_if_t<std::is_same<typename L::type_, Interval<float>>::value 
          && std::is_same<typename R::type_, Interval<float>>::value>* = nullptr>
auto operator+(const L& l, const R& r)
{
  // std::cout << "Runnin interval dummy op\n";
  return ADD_INT<L,R>(l,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)
          && std::is_same<Interval<float>, typename R::type_>::value>* = nullptr>
auto operator+(L l, const R& r)
{
  Interval<float> temp(float{l},float{l});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  // const_temp.value().see();
  // std::cout << "Runnin interval dummy op\n";
  return ADD_INT<Constant<Interval<float>,ByValueClass>,R>(const_temp,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value) && std::is_same<Interval<float>, typename L::type_>::value>* = nullptr>
auto operator+(const L& l, R r)
{
  Interval<float> temp(float{r},float{r});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return ADD_INT<L,Constant<Interval<float>,ByValueClass>>(l,const_temp);
}

template<typename L, typename R, typename std::enable_if_t<std::is_same<typename L::type_, Interval<float>>::value 
          && std::is_same<typename R::type_, Interval<float>>::value>* = nullptr>
auto operator-(const L& l, const R& r)
{
  return SUB_INT<L,R>(l,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)
          && std::is_same<Interval<float>, typename R::type_>::value>* = nullptr>
auto operator-(L l, const R& r)
{
  Interval<float> temp(float{l},float{l});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return SUB_INT<Constant<Interval<float>,ByValueClass>,R>(const_temp,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value) && std::is_same<Interval<float>, typename L::type_>::value>* = nullptr>
auto operator-(const L& l, R r)
{
  Interval<float> temp(float{r},float{r});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return SUB_INT<L,Constant<Interval<float>,ByValueClass>>(l,const_temp);
}

template<typename L, typename R, typename std::enable_if_t<std::is_same<typename L::type_, Interval<float>>::value 
          && std::is_same<typename R::type_, Interval<float>>::value>* = nullptr>
auto operator*(const L& l, const R& r)
{
  return MUL_INT<L,R>(l,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)
          && std::is_same<Interval<float>, typename R::type_>::value>* = nullptr>
auto operator*(L l, const R& r)
{
  Interval<float> temp(float{l},float{l});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return MUL_INT<Constant<Interval<float>,ByValueClass>,R>(const_temp,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value) && std::is_same<Interval<float>, typename L::type_>::value>* = nullptr>
auto operator*(const L& l, R r)
{
  Interval<float> temp(float{r},float{r});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return MUL_INT<L,Constant<Interval<float>,ByValueClass>>(l,const_temp);
}
/////////


template<typename L, typename R, typename std::enable_if_t<std::is_same<typename L::type_, Interval<float>>::value 
          && std::is_same<typename R::type_, Interval<float>>::value>* = nullptr>
auto operator/(const L& l, const R& r)
{
  return DIV_INT<L,R>(l,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)
          && std::is_same<Interval<float>, typename R::type_>::value>* = nullptr>
auto operator/(L l, const R& r)
{
  Interval<float> temp(float{l},float{l});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return DIV_INT<Constant<Interval<float>,ByValueClass>,R>(const_temp,r);
}

template<typename L, typename R, typename std::enable_if_t<(std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value) && std::is_same<Interval<float>, typename L::type_>::value>* = nullptr>
auto operator/(const L& l, R r)
{
  Interval<float> temp(float{r},float{r});
  Constant<Interval<float>,ByValueClass> const_temp(temp);
  return DIV_INT<L,Constant<Interval<float>,ByValueClass>>(l,const_temp);
}

template<typename L, typename R, typename output,typename RefTy = ByValueClass>
struct ADD :  public AD_OP_Base, public ADBase{

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

   Ltype LHS;
   Rtype RHS;
   size_t size;
  public:
    static const int check = 8 ;
    using type_ = output;
    using ref_type = ByValueClass;
   
    ADD( Ltype Lin,  Rtype Rin) : LHS{Lin}, RHS{Rin} 
    {
      if constexpr(is_array_std<type_>::value)
      {
        size = LHS.size;
      }
      else
      {
        size = 1;
      }
    }
    
    

    constexpr  type_ value() const {
      // std::cout << "calling value in ADD: " <<  "\n";
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; // the type of the array, will use it as the functioning type of this class
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} + cast_type{R_temp[i]};
            
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} + cast_type{R_temp};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} + typename type_::value_type{R_temp});
          }
          return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < R_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp} + cast_type{R_temp[i]};
          }
          return answer;
        }
      }
      else{

        if constexpr(std::is_same<type_,typename L::type_>::value && std::is_same<type_,typename R::type_>::value)
        {
          return LHS.value() + RHS.value();
        }
        if constexpr(std::is_same<type_,typename L::type_>::value && !std::is_same<type_,typename R::type_>::value)
        {
          return LHS.value() + type_{RHS.value()};
        }
        if constexpr(!std::is_same<type_,typename L::type_>::value && std::is_same<type_,typename R::type_>::value)
        {
          return type_{LHS.value()} + RHS.value();
        }
        if constexpr(!std::is_same<type_,typename L::type_>::value && !std::is_same<type_,typename R::type_>::value)
        {
          return type_{LHS.value()} + type_{RHS.value()};
        }
      }
      // return type_{LHS.value()} + type_{RHS.value()};
    }

    constexpr type_ diff(int wrt) const
    {
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; // the type of the array, will use it as the functioning type of this class
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
          typename L::type_ L_temp = LHS.diff(wrt);
          typename R::type_ R_temp = RHS.diff(wrt);
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} + cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} + typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.diff(wrt);
          typename R::type_ R_temp = RHS.diff(wrt);
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} + cast_type{R_temp};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} + typename type_::value_type{R_temp});
          }
          return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.diff(wrt);
          typename R::type_ R_temp = RHS.diff(wrt);
          
          for(size_t i = 0; i < R_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp} + cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp} + typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
      }
      else
      {
        return type_{LHS.diff(wrt)} + type_{RHS.diff(wrt)};
      }
    }

    template<typename DIFFTY>
    constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};


template< typename I, typename output, typename RefTy = ByValueClass>
struct EXP : public AD_OP_Base, public ADBase{
  private:

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, I>::value,
  I,
    typename std::conditional <
      std::is_same<ByRefClass,typename I::ref_type>::value,
      const I&,
      I>::type
  >::type Itype;

    Itype input_obj;
    template <class T_c =  output, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>
    T_c exp_mem(T_c x) const
    {
      // std::cout << "Running std..." << std::endl;
      return std::exp(x);
    }

    template <class T_c = output, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>
    T_c exp_mem(T_c x) const
    {
      // std::cout << "Running fixed..." << std::endl;
      return fpm::exp(x);
    }

  public:
    constexpr EXP(Itype in_): input_obj(in_) {}
    using type_ = output;
    using ref_type = RefTy;
    constexpr type_ value() const {
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; // the type of the array, will use it as the functioning type of this class
        type_ answer;
        typename I::type_ input_temp = input_obj.value();
        for(size_t i = 0; i < input_temp.size(); i++)
        {
          answer[i] = exp_mem(cast_type{input_temp[i]});
        }
        return answer;
      }
      if constexpr (!is_array_std<type_>::value)
      {
        if constexpr(std::is_same<type_, typename Itype::type_>::value)
        {
          return exp_mem(input_obj.value());
        }
        else{
          return exp_mem(type_{input_obj.value()});
        }
      }
    }

    constexpr type_ diff(int wrt) const
    {
      if constexpr(is_array_std<type_>::value)
      {
        type_ answer;
        typename I::type_ input_temp_diff = input_obj.diff(wrt);
        typename I::type_ input_temp = input_obj.value();
        for(size_t i = 0; i < input_temp_diff.size(); i++)
        {
          answer.push_back(exp_mem(typename type_::value_type{input_temp.at(i)}) * typename type_::value_type{input_temp_diff.at(i)});
        }
        return answer;
      }
      if constexpr (!is_array_std<type_>::value)
      {
        return exp_mem(type_{input_obj.value()}) * type_{input_obj.diff(wrt)};
      }
    }

    template<typename DIFFTY>
    constexpr   type_ diff(Variable<DIFFTY> wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};

template< typename I, typename output, typename RefTy = ByValueClass>
struct LOG10 : public Node2<LOG10<I,output>, output>, public AD_OP_Base, public ADBase{
  private:
    typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, I>::value,
  I,
    typename std::conditional <
      std::is_same<ByRefClass,typename I::ref_type>::value,
      const I&,
      I>::type
  >::type Itype;


    Itype input_obj;
    template <class T_c = output, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>
    T_c log10_mem(T_c x) const
    {
        return std::log10(x);
    }

    template <class T_c = output, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>
    T_c log10_mem(T_c x) const
    {
        return fpm::log10(x);
    }

  public:
    constexpr LOG10(Itype in_): input_obj(in_) {}
    using type_ = output;
    using ref_type = RefTy;
    // constants
    type_ one = type_{1};
    // type_ one = type_{1};
    constexpr type_ value() const {
      // auto answer = log10_mem(type_{input_obj.value()});

      if constexpr(is_array_std<type_>::value)
      {
        type_ answer;
        typename I::type_ input_temp = input_obj.value();
        for(size_t i = 0; i < input_temp.size(); i++)
        {
          answer.push_back(log10_mem(typename type_::value_type{input_temp.at(i)}));
        }
        return answer;
      }

      if constexpr (!is_array_std<type_>::value)
      {
        return log10_mem(type_{input_obj.value()});
      }
      // std::cout << " log answer: " <<answer << "\n";
      // return answer;
    }


    constexpr type_ diff(int wrt) const
    {

      if constexpr(is_array_std<type_>::value)
      {
        type_ answer;
        typename I::type_ input_temp_diff = input_obj.diff(wrt);
        typename I::type_ input_temp = input_obj.value();
        typename type_::value_type one{1};
        for(size_t i = 0; i < input_temp_diff.size(); i++)
        {
          answer.push_back((one/typename type_::value_type{input_temp.at(i)}) * typename type_::value_type{input_temp_diff.at(i)});
        }
        return answer;
      }

      if constexpr (!is_array_std<type_>::value)
      {
        return (type_{1}/type_{input_obj.value()}) * type_{input_obj.diff(wrt)};
      }
      // auto answer = (one/type_{input_obj.value()})*type_{input_obj.diff(wrt)};
      // auto input = type_{input_obj.value()};
      // auto input_diff = type_{input_obj.diff(wrt)};
      // return answer;
    }

    template<typename DIFFTY>
    constexpr   type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};


template< typename I, typename output, typename RefTy = ByValueClass>
struct LOG : public Node2<LOG10<I,output>, output>, public AD_OP_Base, public ADBase{
  private:
    typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, I>::value,
  I,
    typename std::conditional <
      std::is_same<ByRefClass,typename I::ref_type>::value,
      const I&,
      I>::type
  >::type Itype;


    Itype input_obj;
    template <class T_c = output, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>
    T_c log_mem(T_c x) const
    {
        return std::log(x);
    }

    template <class T_c = output, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>
    T_c log_mem(T_c x) const
    {
        return fpm::log(x);
    }

  public:
  size_t size;
    constexpr LOG(Itype in_): input_obj(in_) 
    {
      if constexpr(is_array_std<type_>::value)
      { // Add error checking
        size = input_obj.size;
      }
      else
      {
        size = 1;
      }
    }
    using type_ = output;
    using ref_type = RefTy;
    // constants
    type_ one = type_{1};
    // type_ one = type_{1};
    constexpr type_ value() const {
      if constexpr(is_array_std<type_>::value)
      {
        type_ answer;
        typename I::type_ input_temp = input_obj.value();
        for(size_t i = 0; i < input_temp.size(); i++)
        {
          answer.push_back(log_mem(typename type_::value_type{input_temp.at(i)}));
        }
        return answer;
      }
      if constexpr(!is_array_std<type_>::value)
      {
        if constexpr(std::is_same<type_, typename std::remove_reference<Itype>::type::type_>::value)
        {
          return log_mem(input_obj.value());
        }
        if constexpr(!std::is_same<type_,typename std::remove_reference<Itype>::type::type_>::value)
        {
          return log_mem(type_{input_obj.value()});
        }
      }
    }


    constexpr type_ diff(int wrt) const
    {
      if constexpr(is_array_std<type_>::value)
      {
        type_ answer;
        typename I::type_ input_temp_diff = input_obj.diff(wrt);
        typename I::type_ input_temp = input_obj.value();
        typename type_::value_type one{1};
        for(size_t i = 0; i < input_temp_diff.size(); i++)
        {
          answer.push_back((one/typename type_::value_type{input_temp.at(i)}) * typename type_::value_type{input_temp_diff.at(i)});
        }
        return answer;
      }
      if constexpr(!is_array_std<type_>::value)
      {

        return (type_{1}/type_{input_obj.value()}) * type_{input_obj.diff(wrt)};
      }
    }

    template<typename DIFFTY>
    constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};


template< typename I>
struct SIN : public Node2<SIN<I>, typename I::type_>,public AD_OP_Base{
  private:
    const I& input_obj;
    template <class T_c = typename I::type_, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>   
    T_c sin_mem(T_c x) const
    {
        return std::sin(x);
    }

    template <class T_c = typename I::type_, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>   
    T_c sin_mem(T_c x) const
    {
        return fpm::sin(x);
    }

  public:
    constexpr SIN(const I& in_): input_obj(in_) {}
    using type_ = typename I::type_;
    constexpr   type_ value() const {
      return sin_mem(input_obj.value());
    }
};


template< typename I>
struct COS : public Node2<COS<I>, typename I::type_>,public AD_OP_Base{
  private:
    const I& input_obj;
    template <class T_c = typename I::type_, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>   
    T_c cos_mem(T_c x) const
    {
        return std::cos(x);
    }

    template <class T_c = typename I::type_, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>   
    T_c cos_mem(T_c x) const
    {
        return fpm::cos(x);
    }

  public:
    constexpr COS(const I& in_): input_obj(in_) {}
    using type_ = typename I::type_;
    constexpr   type_ value() const {
      return cos_mem(input_obj.value());
    }
};

template< typename I>
struct TAN : public Node2<TAN<I>, typename I::type_>,public AD_OP_Base{
  private:
    const I& input_obj;
    template <class T_c = typename I::type_, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>   
    T_c tan_mem(T_c x) const
    {
        return std::tan(x);
    }

    template <class T_c = typename I::type_, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>   
    T_c tan_mem(T_c x) const
    {
        return fpm::tan(x);
    }

  public:
    constexpr TAN(const I& in_): input_obj(in_) {}
    using type_ = typename I::type_;
    constexpr   type_ value() const {
      return tan_mem(input_obj.value());
    }
};


template< typename I, typename output, typename RefTy = ByValueClass>
struct SQRT : public AD_OP_Base, public ADBase{
  private:
    typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, I>::value,
  I,
    typename std::conditional <
      std::is_same<ByRefClass,typename I::ref_type>::value,
      const I&,
      I>::type
  >::type Itype;


    Itype input_obj;
    template <class T_c = output, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>
    T_c sqrt_mem(T_c x) const
    {
        return std::sqrt(x);
    }

    template <class T_c = output, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>
    T_c sqrt_mem(T_c x) const
    {
        return fpm::sqrt(x);
    }

  public:
    constexpr SQRT(Itype in_): input_obj(in_) {}
    using type_ = output;
    using ref_type = RefTy;
    // constants
    type_ one = type_{1};
    type_ point_five = type_{0.5};
    // type_ one = type_{1};
    constexpr type_ value() const {
      // auto answer = log10_mem(type_{input_obj.value()});

      if constexpr(is_array_std<type_>::value)
      {
        type_ answer;
        typename I::type_ input_temp = input_obj.value();
        for(size_t i = 0; i < input_temp.size(); i++)
        {
          answer.push_back(sqrt_mem(typename type_::value_type{input_temp.at(i)}));
        }
        return answer;
      }

      if constexpr (!is_array_std<type_>::value)
      {
        return sqrt_mem(type_{input_obj.value()});
      }
      // std::cout << " log answer: " <<answer << "\n";
      // return answer;
    }


    constexpr type_ diff(int wrt) const
    {

      // if constexpr(is_array_std<type_>::value)
      // {
      //   type_ answer;
      //   typename I::type_ input_temp_diff = input_obj.diff(wrt);
      //   typename I::type_ input_temp = input_obj.value();
      //   typename type_::value_type one{1};
      //   for(size_t i = 0; i < input_temp_diff.size(); i++)
      //   {
      //     answer.push_back((one/typename type_::value_type{input_temp.at(i)}) * typename type_::value_type{input_temp_diff.at(i)});
      //   }
      //   return answer;
      // }

      if constexpr (!is_array_std<type_>::value)
      {
        return (point_five/sqrt_mem(type_{input_obj.value()})) * type_{input_obj.diff(wrt)};
      }
      // auto answer = (one/type_{input_obj.value()})*type_{input_obj.diff(wrt)};
      // auto input = type_{input_obj.value()};
      // auto input_diff = type_{input_obj.diff(wrt)};
      // return answer;
    }

    template<typename DIFFTY>
    constexpr   type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};

// template< typename I>
// struct SQRT : public Node2<SQRT<I>, typename I::type_>, public AD_OP_Base, public ADBase{
//   private:
//     const I& input_obj;
//     template <class T_c = typename I::type_, typename std::enable_if_t<std::is_arithmetic<T_c>::value>* = nullptr>   
//     T_c sqrt_mem(T_c x) const
//     {
//         return std::sqrt(x);
//     }

//     template <class T_c = typename I::type_, typename std::enable_if_t<std::is_base_of<fpm::fixedpoint_base, T_c>::value>* = nullptr>   
//     T_c sqrt_mem(T_c x) const
//     {
//         return fpm::sqrt(x);
//     }

//   public:
//     constexpr SQRT(const I& in_): input_obj(in_) {}
//     using type_ = typename I::type_;
//     constexpr   type_ value() const {
//       return sqrt_mem(input_obj.value());
//     }
// };

template<typename L, typename R, typename output, typename RefTy = ByValueClass>
struct MUL : public AD_OP_Base, public ADBase{

  // typedef typename std::conditional<
  // std::is_base_of<AD_OP_Base, L>::value,
  // L,
  // const L& >::type Ltype;

  // typedef typename std::conditional<
  // std::is_base_of<AD_OP_Base, R>::value,
  // R,
  // const R& >::type Rtype;


  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

   Ltype LHS;
   Rtype RHS;

  public:
    size_t size;

    constexpr MUL(Ltype Lin, Rtype Rin): LHS(Lin), RHS(Rin)
    {
      if constexpr(is_array_std<type_>::value)
      { // Add error checking
        size = Lin.size;
      }
      else
      {
        size = 1;
      }
    }
    using type_ = output;
    using ref_type = ByValueClass;
    constexpr   type_ value() const {

      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; 
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} * cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} * typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} * cast_type{R_temp};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} * typename type_::value_type{R_temp});
          }
          return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < R_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp} * cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp} * typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
      }
      else
      {
        if constexpr(std::is_same<type_, typename L::type_>::value && std::is_same<type_, typename R::type_>::value)
        {
          return LHS.value() * RHS.value();
        }
        if constexpr(std::is_same<type_, typename L::type_>::value && !std::is_same<type_, typename R::type_>::value)
        {
          return LHS.value() * type_{RHS.value()};
        }
        if constexpr(!std::is_same<type_, typename L::type_>::value && std::is_same<type_, typename R::type_>::value)
        {
          return type_{LHS.value()} * RHS.value();
        }
        if constexpr(!std::is_same<type_, typename L::type_>::value && !std::is_same<type_, typename R::type_>::value)
        {
          return type_{LHS.value()} * type_{RHS.value()};
        }
      }
    }


    constexpr type_ diff(int wrt) const
    {
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; 
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp_val = LHS.value();
          typename R::type_ R_temp_val = RHS.value();
          typename L::type_ L_temp_diff = LHS.diff(wrt);
          typename R::type_ R_temp_diff = RHS.diff(wrt);

          for(size_t i = 0; i < L_temp_val.size(); i++)
          {
            answer[i] = cast_type{L_temp_diff[i]} * cast_type{R_temp_val[i]} + cast_type{L_temp_val[i]} * cast_type{R_temp_diff[i]};
            // answer.push_back(typename type_::value_type{L_temp_val.at(i)} * typename type_::value_type{R_temp_diff.at(i)} + typename type_::value_type{R_temp_val.at(i)} * typename type_::value_type{L_temp_diff.at(i)});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp_val = LHS.value();
          typename R::type_ R_temp_val = RHS.value();
          typename L::type_ L_temp_diff = LHS.diff(wrt);
          typename R::type_ R_temp_diff = RHS.diff(wrt);
          
          for(size_t i = 0; i < L_temp_val.size(); i++)
          {
            answer[i] = cast_type{L_temp_diff[i]} * cast_type{R_temp_val} + cast_type{L_temp_val[i]} * cast_type{R_temp_diff};
            // answer.push_back(typename type_::value_type{L_temp_val.at(i)} * typename type_::value_type{R_temp_diff} + typename type_::value_type{R_temp_val} * typename type_::value_type{L_temp_diff.at(i)});
          }
          return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp_val = LHS.value();
          typename R::type_ R_temp_val = RHS.value();
          typename L::type_ L_temp_diff = LHS.diff(wrt);
          typename R::type_ R_temp_diff = RHS.diff(wrt);
          
          for(size_t i = 0; i < R_temp_val.size(); i++)
          {
            answer[i] = cast_type{L_temp_val} * cast_type{R_temp_diff[i]} + cast_type{L_temp_diff} * cast_type{R_temp_val[i]};
            // answer.push_back(typename type_::value_type{L_temp_val} * typename type_::value_type{R_temp_diff.at(i)} + typename type_::value_type{R_temp_val.at(i)} * typename type_::value_type{L_temp_diff});
          }
          return answer;
        }
      }
      else
      {
        return type_{LHS.value()} * type_{RHS.diff(wrt)} + type_{RHS.value()} * type_{LHS.diff(wrt)};
      }
      // return type_{LHS.value()}*type_{RHS.diff(wrt)} + type_{RHS.value()}*type_{LHS.diff(wrt)};
    }

    template<typename DIFFTY>
    constexpr   type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};

template<typename L, typename R, typename output,typename RefTy = ByValueClass>
struct DIV : public AD_OP_Base, public ADBase{

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;

   Ltype NUM;
   Rtype DEN;

  public:
    size_t size;
    constexpr DIV(Ltype Lin, Rtype Rin): NUM(Lin), DEN(Rin)
    {
      if constexpr(is_array_std<type_>::value)
      { // Add error checking
        size = Lin.size;
      }
      else
      {
        size = 1;
      }
    }
    using type_ = output;
    using ref_type = RefTy;
    constexpr type_ value() const {
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>;
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
          typename L::type_ L_temp = NUM.value();
          typename R::type_ R_temp = DEN.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} / cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} / typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = NUM.value();
          typename R::type_ R_temp = DEN.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} / cast_type{R_temp};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} / typename type_::value_type{R_temp});
          }
          return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = NUM.value();
          typename R::type_ R_temp = DEN.value();
          
          for(size_t i = 0; i < R_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp} / cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp} / typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
      }
      else
      {
        if constexpr (std::is_same<type_, typename L::type_>::value && std::is_same<type_, typename R::type_>::value)
        {
          return NUM.value() / DEN.value();
        }
        if constexpr (std::is_same<type_, typename L::type_>::value && !std::is_same<type_, typename R::type_>::value)
        {
          return NUM.value() / type_{DEN.value()};
        }
        if constexpr (!std::is_same<type_, typename L::type_>::value && std::is_same<type_, typename R::type_>::value)
        {
          return type_{NUM.value()} / DEN.value();
        }
        if constexpr (!std::is_same<type_, typename L::type_>::value && !std::is_same<type_, typename R::type_>::value)
        {
          return type_{NUM.value()} / type_{DEN.value()};
        }
      }
    }

    constexpr type_ diff(int wrt) const
    {
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>;
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
          typename L::type_ L_temp_val = NUM.value();
          typename R::type_ R_temp_val = DEN.value();
          typename L::type_ L_temp_diff = NUM.diff(wrt);
          typename R::type_ R_temp_diff = DEN.diff(wrt);
            
          for(size_t i = 0; i < L_temp_val.size(); i++)
          {
            answer[i] = (cast_type{L_temp_diff[i]} * cast_type{R_temp_val[i]} - cast_type{L_temp_val[i]} * cast_type{R_temp_diff[i]}) / (cast_type{R_temp_val[i]} * cast_type{R_temp_val[i]});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp_val = NUM.value();
          typename R::type_ R_temp_val = DEN.value();
          typename L::type_ L_temp_diff = NUM.diff(wrt);
          typename R::type_ R_temp_diff = DEN.diff(wrt);
          
          for(size_t i = 0; i < L_temp_val.size(); i++)
          {
            answer[i] = (cast_type{L_temp_diff[i]} * cast_type{R_temp_val} - cast_type{L_temp_val[i]} * cast_type{R_temp_diff}) / (cast_type{R_temp_val} * cast_type{R_temp_val});
            return answer;
          }
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
         type_ answer;
          typename L::type_ L_temp_val = NUM.value();
          typename R::type_ R_temp_val = DEN.value();
          typename L::type_ L_temp_diff = NUM.diff(wrt);
          typename R::type_ R_temp_diff = DEN.diff(wrt);
            
          for(size_t i = 0; i < R_temp_val.size(); i++)
          {
            answer[i] = (cast_type{L_temp_diff} * cast_type{R_temp_val[i]} - cast_type{L_temp_val} * cast_type{R_temp_diff[i]}) / (cast_type{R_temp_val[i]} * cast_type{R_temp_val[i]});
}
          return answer;
        }
      }
      else
      {
        return (type_{NUM.diff(wrt)} * type_{DEN.value()} - type_{NUM.value()} * type_{DEN.diff(wrt)})/(type_{DEN.value()} * type_{DEN.value()});
      }
    }

    template<typename DIFFTY>
    constexpr   type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};

template<typename L, typename R, typename output, typename RefTy = ByValueClass>
struct SUB : public AD_OP_Base, public ADBase{

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, L>::value,
  L,
    typename std::conditional <
      std::is_same<ByRefClass,typename L::ref_type>::value,
      const L&,
      L>::type
  >::type Ltype;

  typedef typename std::conditional<
  std::is_base_of<AD_OP_Base, R>::value,
  R,
    typename std::conditional <
      std::is_same<ByRefClass,typename R::ref_type>::value,
      const R&,
      R>::type
  >::type Rtype;


   Ltype LHS;
   Rtype RHS;

  public:
    size_t size;
    constexpr SUB(Ltype Lin, Rtype Rin): LHS(Lin), RHS(Rin)
    {
      if constexpr(is_array_std<type_>::value)
      { // Add error checking
        size = Lin.size;
      }
      else
      {
        size = 1;
      }
    }
    using type_ = output;
    using ref_type = RefTy;
    constexpr type_ value() const {
      // std::cout << "Value lhs: " << LHS.value() << "\n";

      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; 
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} - cast_type{R_temp[i]};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} - typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < L_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp[i]} - cast_type{R_temp};
            // answer.push_back(typename type_::value_type{L_temp.at(i)} - typename type_::value_type{R_temp});
          }
          return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          type_ answer;
          typename L::type_ L_temp = LHS.value();
          typename R::type_ R_temp = RHS.value();
          
          for(size_t i = 0; i < R_temp.size(); i++)
          {
            answer[i] = cast_type{L_temp} - cast_type{R_temp.at(i)};
            // answer.push_back(typename type_::value_type{L_temp} - typename type_::value_type{R_temp.at(i)});
          }
          return answer;
        }
      }
      else
      {
        if constexpr(std::is_same<type_, typename L::type_>::value && std::is_same<type_, typename R::type_>::value)
        {
          return LHS.value() - RHS.value();
        }
        if constexpr(std::is_same<type_, typename L::type_>::value && !std::is_same<type_, typename R::type_>::value)
        {
          return LHS.value() - type_{RHS.value()};
        }
        if constexpr(!std::is_same<type_, typename L::type_>::value && std::is_same<type_, typename R::type_>::value)
        {
          return type_{LHS.value()} - RHS.value();
        }
        if constexpr(!std::is_same<type_, typename L::type_>::value && !std::is_same<type_, typename R::type_>::value)
        {
          return type_{LHS.value()} - type_{RHS.value()};
        }
        // return type_{LHS.value()} - type_{RHS.value()};
      }
      // return type_{LHS.value()} - type_{RHS.value()};
    }

    constexpr type_ diff(int wrt) const
    {
      if constexpr(is_array_std<type_>::value)
      {
        using cast_type = std::remove_reference_t<decltype(*std::begin(std::declval<type_&>()))>; // the type of the array, will use it as the functioning type of this class
        if constexpr(is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
        
          type_ answer;
        
          typename L::type_ L_temp_diff = LHS.diff(wrt);
          typename R::type_ R_temp_diff = RHS.diff(wrt);

          for(size_t i = 0; i < L_temp_diff.size(); i++)
          {
            answer[i] = cast_type{L_temp_diff[i]} - cast_type{R_temp_diff[i]};
            // answer.push_back(typename type_::value_type{L_temp_diff.at(i)} - typename type_::value_type{R_temp_diff.at(i)});
          }
          return answer;
        }
        if constexpr(is_array_std<typename L::type_>::value && !is_array_std<typename R::type_>::value)
        {
          type_ answer;
        
        typename L::type_ L_temp_diff = LHS.diff(wrt);
        typename R::type_ R_temp_diff = RHS.diff(wrt);

        for(size_t i = 0; i < L_temp_diff.size(); i++)
        {
          answer[i] = cast_type{L_temp_diff[i]} - cast_type{R_temp_diff};
          // answer.push_back(typename type_::value_type{L_temp_diff.at(i)} - typename type_::value_type{R_temp_diff});
        }
        return answer;
        }

        if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
        {
          // std::cout << "LHS is not a vector, RHS is a vector" << std::endl;
          type_ answer;
          
          typename L::type_ L_temp_diff = LHS.diff(wrt);
          typename R::type_ R_temp_diff = RHS.diff(wrt);

          for(size_t i = 0; i < R_temp_diff.size(); i++)
          {
            answer[i] = cast_type{L_temp_diff} - cast_type{R_temp_diff[i]};
            // answer.push_back(typename type_::value_type{L_temp_diff} - typename type_::value_type{R_temp_diff.at(i)});
          }
          return answer;
        }
      }
      else
      {
        
        return type_{LHS.diff(wrt)} - type_{RHS.diff(wrt)};
      }
    }

    template<typename DIFFTY>
    constexpr type_ diff(Variable<DIFFTY> &wrt_diff)
    {
      return this->diff(wrt_diff.Var_ID);
    }
};

template <typename out_t, typename L, typename R,
          typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>
auto add_ty(const L& lhs_in, const R&rhs_in)
{
    return ADD<L,R,out_t>(lhs_in, rhs_in);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>  
auto operator+(const L& lhs , const R& rhs) { // TODO: Implement precedence of types
  if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
  {
    return ADD<L,R,typename R::type_>(lhs,rhs);
  }
  else
  {
    return ADD<L,R,typename L::type_>(lhs,rhs);
  }
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, R>::value && !std::is_same<typename R::type_, Interval<float>>::value
          && (std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)>* = nullptr>  
auto operator+(L lhs, const R& rhs)
{

    Constant<L,ByValueClass> temp(lhs);
    return ADD<Constant<L,ByValueClass>,R,typename R::type_>(temp, rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && !std::is_same<typename L::type_, Interval<float>>::value
          && (std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value)>* = nullptr>
auto operator+(const L& lhs, R rhs)
{
    Constant<R,ByValueClass> temp(rhs);
    return ADD<L,Constant<R,ByValueClass>,typename L::type_>(lhs, temp);
}

template <typename out_t, typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>     
constexpr const EXP<I_t, out_t> exp_ty(const I_t& in_)
{
    return EXP<I_t, out_t>(in_); // out_ty is the computation type and the return type of the object.
}

template <typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>    
constexpr auto exp(const I_t& in_)
{
    return EXP<I_t, typename I_t::type_>(in_);
}


template <typename I>  
constexpr const TAN<I> tan(const I& in_)
{
    return TAN<I>(in_);
}


template <typename out_t, typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value  && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>     
auto log10_ty(const I_t& in_)
{
    return LOG10<I_t, out_t>(in_);
}

template <typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>
auto log10(const I_t& in_)
{
    return LOG10<I_t, typename I_t::type_>(in_);
}



template <typename out_t, typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value  && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>     
auto log_ty(const I_t& in_)
{
    return LOG<I_t, out_t>(in_);
}


template <typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>
auto log(const I_t& in_)
{
    return LOG<I_t, typename I_t::type_>(in_);
}


template <typename I>  
constexpr const SIN<I> sin(const I& in_)
{
    return SIN<I>(in_);
}

template <typename I>  
constexpr const COS<I> cos(const I& in_)
{
    return COS<I>(in_);
}

template <typename I_t, typename std::enable_if_t<std::is_base_of<ADBase, I_t>::value && 
            !std::is_same<typename I_t::type_, Interval<float>>::value>* = nullptr>    
constexpr auto sqrt(const I_t& in_)
{
    return SQRT<I_t, typename I_t::type_>(in_);
}

template <typename out_t, typename L, typename R,typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>   
auto mul_ty(const L& lhs , const R& rhs){
  return MUL<L,R,out_t>(lhs,rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>  
constexpr auto operator*(const L& lhs , const R& rhs){
  if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
  {
    return MUL<L,R,typename R::type_>(lhs,rhs);
  }
  else
  {
    return MUL<L,R,typename L::type_>(lhs,rhs);
  }
  // return MUL<L,R, typename L::type_>(lhs,rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, R>::value && !std::is_same<typename R::type_, Interval<float>>::value
          && (std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)>* = nullptr>  
auto operator*(L lhs, const R& rhs)
{
  Constant<L,ByValueClass> temp(lhs);
  return MUL<Constant<L,ByValueClass>,R,typename R::type_>(temp, rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && !std::is_same<typename L::type_, Interval<float>>::value
          && (std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value)>* = nullptr>  
auto operator*(const L& lhs, R rhs)
{
  Constant<R,ByValueClass> temp(rhs);
  return MUL<L,Constant<R,ByValueClass>,typename L::type_>(rhs, temp);
}


template <typename out_t, typename L, typename R,typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>    
auto div_ty(const L& lhs , const R& rhs){
  return DIV<L,R,out_t>(lhs,rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>  
constexpr auto operator/(const L& lhs , const R& rhs){
  if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
  {
    return DIV<L,R,typename R::type_>(lhs,rhs);
  }
  else
  {
    return DIV<L,R,typename L::type_>(lhs,rhs);
  }

}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, R>::value && !std::is_same<typename R::type_, Interval<float>>::value
          && (std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)>* = nullptr>  
auto operator/(L lhs, const R& rhs)
{
    Constant<L,ByValueClass> temp(lhs);
    return DIV<Constant<L,ByValueClass>,R,typename R::type_>(temp, rhs);
  // Constant<L,ByValueClass> temp(lhs);
  // return DIV<Constant<L,ByValueClass>,R,typename R::type_>(temp, rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && !std::is_same<typename L::type_, Interval<float>>::value
          && (std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value)>* = nullptr>  
auto operator/(const L& lhs, R rhs)
{
    Constant<R,ByValueClass> temp(rhs);
    return DIV<L,Constant<R,ByValueClass>,typename L::type_>(lhs, temp);
  // Constant<R,ByValueClass> temp(rhs);
  // return DIV<L,Constant<R,ByValueClass>,typename L::type_>(lhs, temp);
}

template <typename out_t, typename L, typename R,typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>   
auto sub_ty(const L& lhs , const R& rhs){
  return SUB<L,R,out_t>(lhs,rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && std::is_base_of<ADBase, R>::value
          && !std::is_same<typename L::type_, Interval<float>>::value>* = nullptr>  
constexpr auto operator-(const L& lhs , const R& rhs){
  if constexpr(!is_array_std<typename L::type_>::value && is_array_std<typename R::type_>::value)
  {
    return SUB<L,R,typename R::type_>(lhs,rhs);
  }
  else
  {
    return SUB<L,R,typename L::type_>(lhs,rhs);
  }
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, R>::value && !std::is_same<typename R::type_, Interval<float>>::value
          && (std::is_arithmetic<L>::value || std::is_base_of<fpm::fixedpoint_base, L>::value)>* = nullptr>  
auto operator-(L lhs, const R& rhs)
{
  // using r_type = typename R::type_;
  // if constexpr(is_array_std<r_type>::value)
  // {
  //   r_type rhs_vec = r_type(rhs.size, lhs);
  //   Constant<r_type,ByValueClass> temp(rhs_vec);
  //   return SUB<Constant<r_type,ByValueClass>,R,r_type>(temp, rhs);
  // }

  // if constexpr(!is_array_std<r_type>::value)
  // {
    // std::cout << "Correct operator called" << std::endl;
    Constant<L,ByValueClass> temp(lhs);
    return SUB<Constant<L,ByValueClass>,R,typename R::type_>(temp, rhs);
  // }
  // Constant<L,ByValueClass> temp(lhs);
  // // std::cout << "Oper: " << temp.value() << "\n";
  // return SUB<Constant<L,ByValueClass>,R,typename R::type_>(temp, rhs);
}

template <typename L, typename R, typename std::enable_if_t<std::is_base_of<ADBase, L>::value && !std::is_same<typename L::type_, Interval<float>>::value
          && (std::is_arithmetic<R>::value || std::is_base_of<fpm::fixedpoint_base, R>::value)>* = nullptr>  
auto operator-(const L& lhs, R rhs)
{
    Constant<R,ByValueClass> temp(rhs);
    return SUB<L,Constant<R,ByValueClass>,typename L::type_>(lhs, temp);
}

template<typename T>
void print_ad(T& input)
{
  if constexpr(is_array_std<typename T::type_>::value)
  {
    for(auto i : input.value())
    {
      std::cout << i << "\n";
    }
  }
  if constexpr(!is_array_std<typename T::type_>::value)
  {
    std::cout << input.value() << "\n";
  }
}


// Some useful constants

// 1/sqrt(2*pi)

#define one_by_sqrt_2_pi 0.3989422804014327

// Constant<float> one_by_sqrt_2_pi(0.3989422804014327);
