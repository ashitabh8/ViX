# ViX

ViX is a framework setup to run probabilistic programs on the edge. Currently, there are two ways to use this framework:

a) PyVix: A Python framework which generates c++ code that can be run both on the cpu and the arduino.

b) Cpp interface: The header files can directly be used to write the probabilistic models as well, as they are in Benchmarks/

Notes:

Benchmarks/run_benchmarks.py can be used to easily run all experiments, and generate the arduino code from the cpp files.
