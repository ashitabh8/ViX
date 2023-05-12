import interval_arithmetic as ia
import numpy as np
import math

def get_integer_bits(input, shift = 0.9):
    return math.ceil(math.log2(input) * (shift))

def get_virtual_bits(input, shift = 0.9):
    I = get_integer_bits(input, shift=shift)
    K = math.ceil(math.log2(input)) - I
    F = 31 - I
    if F < 8:
        print("Original K: {}".format(K))
        print("Original F: {}".format(F))
        K = K + (8 - F)
        print("New K: {}".format(K))
        F = 8
        
        print("Warning: F < 8 - Resizing to 8")
        if K > 8:
            print("Warning: K > 8 - Resizing to 8")
            K = 8
    if K > 8:
        print("Warning: K > 8 - Resizing to 8")
        K = 8
    I = 31 - F
    # T = math.ceil(math.log2(input))
    return I, K, F
    

class Node:
    def __init__(self, Interval):
        self.Interval = Interval
        pass

    def __mul__(self, other):
        return Mul(self, other)
    
    def __add__(self, other):
        return Add(self, other)
    

    def __sub__(self, other):
        return Sub(self, other)
    
    def __truediv__(self, other):
        return Div(self, other)
    
    def __rsub__(self, other):
        return Sub(other, self)
    
    def __rtruediv__(self, other):
        return Div(other, self)
    
    def __rmul__(self, other):
        return Mul(other, self)
    
    def set_interval(self, interval):
        self.Interval = interval
    
    def get_interval(self):
        return self.Interval

    def set_name(self, name):
        self.name = name
    
    def get_name(self):
        return self.name
    
    def get_lb(self):
        return self.Interval.lb

    def get_ub(self):
        return self.Interval.ub

    
    # def set_string(self, string):
    #     self.string = string
    
    __radd__ = __add__

class Variable(Node):
    def __init__(self, init = None, range = None, name = None):
        self.range = range
        self.name = name
        self.init = init
        print("NAME: ", self.name)
        self.is_variable = True
        if type(range) is list:
            if range is not None:
                if range[0] > range[1]:
                    raise Exception("Lower bound is greater than upper bound")
                
                Node.__init__(self,ia.Interval(range[0], range[1]))
    
    def set_range(self, range):
        self.range = range
        if type(range) is list:
            if range is not None:
                if range[0] > range[1]:
                    raise Exception("Lower bound is greater than upper bound")
                
                Node.__init__(self,ia.Interval(range[0], range[1]))
    
    def get_string(self):
        return self.name
    
    def get_midpoint(self):
        return (self.Interval.lb + self.Interval.ub) / 2
    
    def set_name(self, name):
        self.name = name
    
    def get_complete_definition(self, running_type):
        return f"Variable<{running_type}> {self.name} ({self.value});\n"
    
    def complete_definition(self, running_type):
        if type(self.range) is not list:
            print("range: ", self.range)
            return f"Variable<{running_type}> {self.name} ({self.range});\n"
        else:
            return f"Variable<{running_type}> {self.name} ({self.init});\n"
        # return f"Variable<{running_type}> {self.name} ({self.range});\n"

    # def get_interval(self):
    #     return self.Interval
    
    # def set_interval(self, interval):
    #     self.Interval = interval


class Constant(Node):
    def __init__(self, range, name = None) -> None:
        self.range = range
        self.name = name
        self.is_variable = True

        if range is not None and type(range) is list:
            if range[0] > range[1]:
                raise Exception("Lower bound is greater than upper bound")
            
            Node.__init__(self,ia.Interval(range[0], range[1]))
        elif range is not None and type(range) is not list:
            Node.__init__(self,ia.Interval(range, range))
    
    def get_string(self):
        return self.name
    
    def get_assignment(self, running_type):
        if type(self.range) is not list:
            return f"Constant<{running_type}> {self.name} ({self.range});\n"
    
    def complete_definition(self, running_type):
        if type(self.range) is not list:
            return f"Constant<{running_type}> {self.name} ({self.range});\n"
        # return f"Constant<{running_type}> {self.name} ({self.value});\n"

    def get_midpoint(self):
        if type(self.range) is list:
            return (self.range[0] + self.range[1]) / 2
        else:
            self.range
        # return (self.Interval.lb + self.Interval.ub) / 2
    
    def get_value(self):
        #possible to return list or constant
        return self.range
    def get_complete_definition(self, running_type):
        return f"Constant<{running_type}> {self.name} ({self.value});\n"

    def set_name(self, name):
        super().set_name(name)
    
class Distribution(Node):

    def __init__(self, Interval):
        super().__init__(Interval)
        # self.name = None
        # self.is_distribution = True



class Mul(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
        self.string = f" {self.a.get_string()} * {self.b.get_string()}"
        Node.__init__(self, self.a.Interval * self.b.Interval)
    
    def get_string(self):
        return self.string
        # print("{} * {}".format(self.a.name, self.b.name))
    
    def get_interval(self):
        return self.Interval
    

class Log(Node):

    def __init__(self, a):
        self.a = a
        self.string = f" log({self.a.get_string()})"
        # print("type of a: {}".format(type(a)))

        Node.__init__(self, ia.Interval(np.log(self.a.Interval.lb), np.log(self.a.Interval.ub)))
    
    def get_string(self):
        return self.string


class Add(Node):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.string = f" {self.a.get_string()} + {self.b.get_string()}"
        Node.__init__(self, self.a.Interval + self.b.Interval)
    
    def get_string(self):
        return self.string
    
    def get_interval(self):
        return self.Interval
    

class Sub(Node):
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.string = f" {self.a.get_string()} - {self.b.get_string()}"
        Node.__init__(self, self.a.Interval - self.b.Interval)
    
    def get_string(self):
        return self.string
    

class Div(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.string = f" {self.a.get_string()} / {self.b.get_string()}"
        Node.__init__(self, self.a.Interval / self.b.Interval)
    
    def get_string(self):
        return self.string
    
        
        
        

class Data(Node):
    def __init__(self, data,name = None):
        self.string = name
        self.data = data
        Node.__init__(self, ia.Interval(min(data), max(data)))
    
    def get_string(self):
        return self.string

    def get_size(self):
        return len(self.data)
    
    def get_posterior_string(self):
        return self.string

    def get_constant_container(self, running_type):
        return f"Constant<{running_type}> {self.string}(0);\n"

   
    def get_array_name(self):
        return f"{self.string}_array"
    
    def get_container_name(self):
        return f"{self.string}_container"
     
class Uniform (Distribution, Node):


    def __init__(self, lb, ub,learnable_params,learnable_names = ['mean','stddev'], name = None):

        if lb > ub:
            raise Exception("Lower bound is greater than upper bound")
        
        self.lb = lb
        self.ub = ub
        self.string = name
        self.sample_value_name = f"{self.string}_sample_value"
        self.prior_string = f"{self.string}_prior"
        self.sample_container_name = f"{self.string}_sample"
        self.learnable_params = learnable_params
        self.learnable_names = learnable_names
        Distribution.__init__(self, ia.Interval(lb, ub))
        # print("Self get interval: ", self.get_interval())
        # print("Setting interval: ", ia.Interval(lb, ub))
        learnable_params[0].set_interval(ia.Interval(lb, ub))
        # learnable_params[0].set_name(self.string + "_mean")
        # learnable_params[1].set_name(self.string + "_stddev")
        print("self.lb+ self.ub/2", (self.lb+ self.ub)/2)
        learnable_params[0].set_range((self.lb+self.ub)/2)
        print("learnable_params[0] name: ", learnable_params[0].name)
        print("learnable_params[1] name: ", learnable_params[1].name)
        # print("learnable_params[0].get_interval()", learnable_params[0].get_interval())
        self.posterior = Normal(learnable_params[0], learnable_params[1], name= self.string + "_posterior")
        self.posterior.set_interval(self.get_interval())
        # print("self.get_interval()", self.get_interval())
        # print("self.posterior.get_interval()", self.posterior.get_interval())
        # Node.__init__(ia.Interval(lb, ub))
        # set mu interval learnable_params[0]
        mu = learnable_params[0]
        mu.set_interval(self.get_interval())

    def get_learnable_params(self):
        return self.learnable_params
        
    def get_posterior(self):
        return self.posterior 
    
    def likelihood(self, x):
        if self.lb <= x and x <= self.ub:
            return 1 / (self.ub - self.lb)
        else:
            return 0
    
    def log_likelihood(self, x):
        if self.lb <= x and x <= self.ub:
            return -np.log(self.ub - self.lb)
        else:
            return -float('inf')


    def prior_log_likelihood_interval(self):
        return ia.Interval(-np.log(self.ub - self.lb), -np.log(self.ub - self.lb))
    
    def posterior_log_likelihood_interval(self, x_interval):
        return self.posterior.log_likelihood_interval(x_interval)

    def log_likelihood_interval(self,interval = None):
        if isinstance(interval, ia.Interval):
            if self.lb <= interval.lb and interval.ub <= self.ub:
                return ia.Interval(-np.log(self.ub - self.lb), -np.log(self.ub - self.lb))
            else:
                return ia.Interval(-float('inf'), -float('inf'))
        
        if isinstance(interval, (int, float)):
            if self.lb <= interval and interval <= self.ub:
                return ia.Interval(-np.log(self.ub - self.lb), -np.log(self.ub - self.lb))
            else:
                return -float('inf')
        
        if isinstance(interval, Node):
            return ia.Interval(-np.log(self.ub - self.lb), -np.log(self.ub - self.lb))
    
    def get_string(self):
        return self.string



class Normal(Distribution):

    def __init__(self, mu, sigma, name = None, posterior = False):
        self.mu = mu
        self.string = name
        # self.mu_value = mu
        print("Self mu: ", self.mu)
        if isinstance(mu, (int, float)):
            self.mu_interval = ia.Interval(mu, mu)
            self.mu = Constant(mu, name = f"{name}_mean_container")
        if isinstance(mu, Node):
            self.mu_interval = mu.get_interval()
            self.mu.set_name(f"{name}_mean_container")
        self.sigma = sigma

        if isinstance(sigma, (int, float)):
            self.sigma_interval = ia.Interval(sigma, sigma)
            self.sigma = Constant(sigma, name = f"{name}_sigma_container")
        if isinstance(sigma, Node):
            self.sigma_interval = sigma.get_interval()
            self.sigma.set_name(f"{name}_sigma_container")
        
        
        
        

        interval_1 = self.mu_interval -  3 * self.sigma_interval
        
        interval_2 = self.mu_interval + 3 * self.sigma_interval
        

        final_interval = ia.Interval(min(interval_1.lb,interval_2.lb), max(interval_1.ub, interval_2.ub))

        Distribution.__init__(self, final_interval)
        
    
    def likelihood(self, x):
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(- (x - self.mu) ** 2 / (2 * self.sigma ** 2))
    
    def get_mean_name(self):
        return self.mu.name
    
    def get_stddev_name(self):
        return self.sigma.name
    

    def complete_definition(self, running_type = "double"):
        # print("SIGMA:", self.sigma)
        # print("SIGMA NAME: ", self.sigma.name)
        main_str = [self.mu.complete_definition(running_type), self.sigma.complete_definition(running_type)]
        # print("main_str: ", main_str)
        return main_str + [f"auto {self.string} = normal_dist<{running_type}>({self.get_mean_name()}, {self.get_stddev_name()}, {self.get_sample_name()} );\n"]
        # return f"normal_dist<{running_type}>({self.get_mean_name()}, {self.get_stddev_name()},{self.get_sample_name()} );\n"
    
    def get_prior_name(self):
        if "posterior" in self.string:
            #remove posterior from string
            prior_string = self.string.replace("posterior", "")
            return prior_string
        else:
            return self.string
    
    def get_sample_name(self):
        if "posterior" in self.string:
            #remove posterior from string
            sample_string = self.string.replace("posterior", "sample")
            return sample_string
        else:
            return self.string+"_sample"
    

    def log_likelihood(self, x):
        return -np.log(np.sqrt(2 * np.pi * self.sigma ** 2)) - (x - self.mu) ** 2 / (2 * self.sigma ** 2)

    def get_string(self):
        return self.string
    
    def set_interval(self, interval):
        Node.set_interval(self, interval)

    
    def log_likelihood_interval(self, interval):
        

        const_2pi  = Constant(-0.9189385, name = "const_2pi")
        neg_half = Constant(-0.5, name = "neg_half")
        one_int = Constant(1, name = "one_int")
        if not isinstance(interval, Node):
            interval_input = Variable(range = [interval.lb, interval.ub], name = "interval")
        else:
            interval_input = interval
        
        
        log_lh  = const_2pi  - Log(self.sigma) + neg_half * (one_int / self.sigma) * (one_int / self.sigma) * (interval_input - self.mu) * (interval_input - self.mu)
        
        return log_lh.get_interval()

    def diff_log_likelihood(self,interval, wrt = 'mean'):
        one_int = Constant(1, name = "one_int")
        if not isinstance(interval, Node):
            interval_input = Variable(range = [interval.lb, interval.ub], name = "interval")
        else:
            interval_input = interval
        if wrt == 'mean':
            return (self.mu - interval_input) / ((self.sigma) * (self.sigma))
        if wrt == 'stddev':
            return (((interval_input - self.mu) *(interval_input - self.mu)) / (self.sigma *self.sigma*self.sigma)) - one_int / self.sigma

    

class Param(Node):

    def __init__(self, prior, learnable_mu, learnable_sigma, name = None):
        self.prior = prior
        temp_interval = prior.Interval
        mid_point = (temp_interval.lb + temp_interval.ub) / 2
        self.guide = Normal(mu = mid_point, sigma = 1, name = prior.get_string() + "_guide")
        self.learnable_mu = learnable_mu
        self.learnable_sigma = learnable_sigma
        self.string = name
        self.posterior = None
        Node.__init__(self, self.guide.Interval)
    
    def get_string(self):
        return self.string

class Observed(Node):

    def __init__(self, dist, data, name = None):
        self.dist = dist
        self.data = data
        lb = min(data.data)
        ub = max(data.data)
        interval = ia.Interval(lb, ub)
        super().__init__(interval)
        # dist.set_interval(interval)
        self.string = name
        # Node.__init__(self, self.dist.Interval)
    
    def get_string(self):
        return self.string
    def get_array_name(self):
        return f"{self.string}_array"
    
    def get_posterior_string(self):
        return self.string
    
    def get_interval(self):
        return self.Interval * len(self.data.data)
    
    def log_likelihood_interval(self):
        return self.dist.log_likelihood_interval(self.data) * len(self.data.data)
    
    def get_assignment(self, running_type = "double"):
        return f"auto {self.string} = normal_dist<{running_type}>(mean_{self.string}, {self.string}_sigma_container, {self.data.string});\n"


class Model:
    def __init__(self, all_distributions, params = None):
        self.all_distributions = all_distributions
        self.priors = self.get_priors()
        self.observed = self.get_observed()
        self.input_data = self.get_data()
        # self.num_data = params["num_data"]
        self.num_samples = params["num_samples"]
        self.num_experiments = params["num_experiments"]
        self.learning_rate = params["learning_rate"]
        self.iterations = params["iterations"]
        
    def get_range_distributions(self):
        range_distributions = {}
        for dist in self.all_distributions:
            range_distributions[dist.get_string()] = dist.get_interval()
            
        return range_distributions


    def get_data(self):
        for dist in self.all_distributions:
            if isinstance(dist, Data):
                return dist

    def get_priors(self):
            priors = []
            for dist in self.all_distributions:
                if isinstance(dist, Observed) or isinstance(dist, Data):
                    continue
                else:
                    priors.append(dist)
                
            return priors
    
    def get_observed(self):
            for dist in self.all_distributions:
                if isinstance(dist, Observed):
                    return dist

    def run_analysis(self, alpha = 0.9):

        def get_observed(all_distributions):
            for dist in all_distributions:
                if isinstance(dist, Observed):
                    return dist
        
        def get_priors(all_distributions):
            priors = []
            for dist in all_distributions:
                if isinstance(dist, Observed) or isinstance(dist, Data):
                    continue
                else:
                    priors.append(dist)
                
            return priors
        
        observed = get_observed(self.all_distributions)
        all_priors = get_priors(self.all_distributions)
        # print("Observed Variable Log Likelihood Interval:")
        # print(observed.log_likelihood_interval())

        # print("Range Analysis of Priors:")
        # for prior in all_priors:
        #     print(prior.get_string(), prior.get_interval())

        # print("Log Likelihood of Priors:")
        # for prior in all_priors:
        #     print(prior.get_string(), prior.prior_log_likelihood_interval())

        
        # print(" Log-lh of posterior:")
        # for prior in all_priors:
        #     print(prior.get_string(), prior.posterior_log_likelihood_interval(prior.get_interval()))

        # print("Range Analysis of Gradient of ELBO")
        

        # print("Abstract ELBO Common term: ")

        # print("Second Term of ELBO: ")
        # Sum of priors
        sum_of_priors = None
        for prior in all_priors:
            if sum_of_priors is None:
                sum_of_priors = prior.log_likelihood_interval(prior.get_interval())
            else:
                sum_of_priors = sum_of_priors +  prior.log_likelihood_interval(prior.get_interval())

        # Add observed variable
        # print("Observed Variable Log Likelihood Interval:")
        # print(observed.log_likelihood_interval())
        final_elbos = sum_of_priors + observed.log_likelihood_interval()

        # Subtract guide
        for prior in all_priors:
            final_elbos = final_elbos - prior.posterior_log_likelihood_interval(prior.get_interval())
        # print(sum_of_priors)

        # First term of ELBO 
        # print("First Term of ELBO: ")
        
        # for prior in all_priors:
        #     print(prior.get_string(),"wrt  mean ", prior.posterior.diff_log_likelihood(prior.get_interval(), wrt = 'mean').get_interval())
        #     print(prior.get_string(),"wrt stddev", prior.posterior.diff_log_likelihood(prior.get_interval(), wrt = 'stddev').get_interval())

        # print("Final ELBOs:")
        # final_elbos = {}
        # for prior in all_priors:
        #     for i in range(len(prior.learnable_names)):
        #         if prior.learnable_names[i] == 'mean':
        #             final_elbos[prior.get_string()+'_mean'] = sum_of_priors
        #         if prior.learnable_names[i] == 'stddev':
        #             final_elbos[prior.get_string()+'_stddev'] = sum_of_priors

        max_abs_value = max(abs(final_elbos.lb), abs(final_elbos.ub))

        return max_abs_value
        
        # for key in final_elbos:
        #     print(key, final_elbos[key])

        # get max abs value from dictionary
        # max_abs_value = 0
        # for key in final_elbos:
        #     if abs(final_elbos[key].lb) > max_abs_value:
        #         max_abs_value = abs(final_elbos[key].lb)
        #     if abs(final_elbos[key].ub) > max_abs_value:
        #         max_abs_value = abs(final_elbos[key].ub)
        
        # get virtual bits
        # I, K , F = get_virtual_bits(max_abs_value, alpha)

        # return I,K,F
    
    def get_bit_configuration(self, alpha = 0.9):
        max_value = self.run_analysis(alpha)
        I, K , F = get_virtual_bits(max_value, alpha)
        return I, K, F


if __name__ == "__main__":

    mu_0, sigma_0 = Variable(name = 'mu0'), Variable(name = 'sigma0', range = [1, 5])
    mu_1, sigma_1 = Variable(name = 'mu1'), Variable(name = 'sigma1', range = [1, 5])
    a0 = Uniform(0, 10, [mu_0, sigma_0], name = 'a0')
    a1 = Uniform(0, 10,[mu_1, sigma_1], name = 'a1')
    data = Data([1.2,3.4], name = 'data')
    temp = a0 * data + a1
    Y_obs_data = Data([5.45, 9.9], name = 'Y_obs_data')
    Y = Observed(dist = Normal(temp, 1), data = Y_obs_data, name = 'Y_obs')
    M = Model([Y, a0, a1])
    final_elbos = M.run_analysis()
    print("type of final_elbos", type(final_elbos))
    print(final_elbos)
    for key in final_elbos:
        print(key, final_elbos[key])
    # M.gen_code()
