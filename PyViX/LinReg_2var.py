# import distributions as 
# from distributions import *
# import distributions
# from distributions import Normal
# from distributions import Uniform
# from distributions import Data
# from distributions import Model
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from distributions import *
import interval_arithmetic as ia
from GenCode import gen_code


parameters = {
    'iterations': 1500,
    'learning_rate': 0.01,
    'num_experiments': 1,
    'num_samples': 4,
    'type': 'float'
}

# gen linreg data for 5x + 6

def gen_data(num_samples = 100):
    data_X = []
    data_Y = []
    np.random.seed(1236)
    for i in range(num_samples):
        x = np.random.uniform(-2,2)
        data_X.append(x)
        data_Y.append(15* x + 13 + np.random.normal(0, 0.5))
    return list(data_X), list(data_Y)




if __name__ == "__main__":

    data_X, data_Y = gen_data(num_samples=50)

    mu_0, sigma_0 = Variable(), Variable(init = 2, range=[0.1,5])
    mu_1, sigma_1 = Variable(), Variable(init = 2, range=[0.1,5])
    a0 = Uniform(0, 25, name = 'a0',learnable_params=[mu_0, sigma_0])
    a1 = Uniform(0, 25, name = 'a1',learnable_params=[mu_1, sigma_1])
    data = Data(data_X, name = 'data')
    Y_obs_data = Data(data_Y, name = 'Y_obs_data')
    Y = Observed(dist = Normal(a0 * data + a1, 1, name = 'Y'), data = Y_obs_data, name = 'Y')
    M = Model([Y, a0, a1, data], params = parameters)
    
    gen_code(M, file_name = "test_2v", type=parameters['type'])
    print("Filename test_2v.cpp generated.")
    print(f"Type: {parameters['type']}")