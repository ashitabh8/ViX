# ViX

ViX is a framework setup to run probabilistic programs on the edge. Currently, there are two ways to use this framework:

a) PyVix: A Python framework which generates c++ code that can be run both on the cpu and the arduino.

b) Cpp interface: The header files can directly be used to write the probabilistic models as well, as they are in Benchmarks/


# PyViX Syntax and Example

## Linear Regression Model
```python

    mu_0, sigma_0 = Variable(), Variable(init = 2, range=[0.1,5])
    a0 = Uniform(0, 30, name = 'a0',learnable_params=[mu_0, sigma_0])
    data = Data(data_X, name = 'data')
    Y_obs_data = Data(data_Y, name = 'Y_obs_data')
    Y = Observed(dist = Normal(a0 * data , 1, name = 'Y'), data = Y_obs_data, name = 'Y')
    M = Model([Y, a0, data], params = parameters)

```

Notes:

Benchmarks/run_benchmarks.py can be used to easily run all experiments, and generate the arduino code from the cpp files.
