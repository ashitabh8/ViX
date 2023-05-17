# ViX

ViX is a framework setup to run probabilistic programs on the edge. Currently, there are two ways to use this framework:

a) PyVix: A Python framework which generates c++ code that can be run both on the cpu and the arduino.

b) Cpp interface: The header files can directly be used to write the probabilistic models as well, as they are in Benchmarks/


# PyViX Syntax and Example

Workflow summary: PyVix is a python interface that generates the C++17 and Arduino code (compatible with ARM processors) with the appropriate interval analysis of the distributions, evidence lower bound and scores.

## Linear Regression Model
```python

    mu_0, sigma_0 = Variable(), Variable(init = 2, range=[0.1,5])
    a0 = Uniform(0, 30, name = 'a0',learnable_params=[mu_0, sigma_0]) # Each name should be unique
    data = Data(data_X, name = 'data')
    Y_obs_data = Data(data_Y, name = 'Y_obs_data')
    Y = Observed(dist = Normal(a0 * data , 1, name = 'Y'), data = Y_obs_data, name = 'Y')
    M = Model([Y, a0, data], params = parameters)

```

### Defining learnable parameters

```python
mu_0, sigma_0 = Variable(), Variable(init = 2, range=[0.1,5])
```

First, we define the parameters of the posterior using `Variable()` or `Constant()`. ViX only supports unimodal gaussian as a posterior so the parameters in this case are mean and variance. We can also provide an initial value for each parameter and a range if it suits the application.

### Defining Prior 

```python
a0 = Uniform(0, 30, name = 'a0',learnable_params=[mu_0, sigma_0])
```

Currently, ViX supports Uniform and Normal distributions as priors.

### Input Data

```python
data = Data(data_X, name = 'data')
Y_obs_data = Data(data_Y, name = 'Y_obs_data')
```
Input and Observed data is wrapped around the `Data()` object.

### Observed Variable

```python
Y = Observed(dist = Normal(a0 * data , 1, name = 'Y'), data = Y_obs_data, name = 'Y')
```

The observed variable is defined using `Observed()` object.


Notes:

Benchmarks/run_benchmarks.py can be used to easily run all experiments, and generate the arduino code from the cpp files.
