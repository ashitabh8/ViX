from lib2to3.pytree import convert
import os
from pydoc import doc
import pandas as pd
# import math

import math

def get_integer_bits(input, shift = 0.7):
    return math.ceil(math.log2(input) * (shift))

# def get_virtual_bits(input, shift = 0.7):
#     I = get_integer_bits(input, shift=shift)
#     K = math.ceil(math.log2(input)) - I
#     F = 31 - I
#     if F < 8:
#         K = K + (8 - F)
#         F = 8
#         if K > 8:
#             K = 8
#     if K > 8:
#         K = 8
#     # T = math.ceil(math.log2(input))
#     return I, K, F

def get_virtual_bits(input, shift = 0.9):
    I = get_integer_bits(input, shift=shift)
    K = math.ceil(math.log2(input)) - I
    F = 31 - I
    if F < 8:
        K = K + (8 - F)
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

def compile(file_config):
    output_name =''
    if file_config["output_file"] is not None:
        output_name = file_config["output_file"]
    else:
        output_name = file_config["file_name"].split(".")[0]
    command = f"g++ -std=c++17 {file_config['dir']}/{file_config['file_name']} -o {file_config['dir']}/{output_name}"
    if "misc_flags" in file_config:
        command += " " + " ".join(file_config["misc_flags"])
    os.system(command)

def run(file_config, args):
    output_name =''
    if file_config["output_file"] is not None:
        output_name = file_config["output_file"]
    else:
        output_name = file_config["file_name"].split(".")[0]
    #args list of arguments
    command = f"{file_config['dir']}/{output_name} {' '.join(args)}"
    print(command)
    print(f"Running: {command}")
    os.system(command)


def see_results_old(file_config, args = None, top_n=10, params = None):
    #if file_config is a dict
    if isinstance(file_config, dict):

        print("Results:")
        df = pd.read_csv(f"{args['dir']}/{args['result_file']}")
        df = df.sort_values(by=['elbo'], ascending=False)
    # print all results if chenge_in_mean is less 0.01
    # df = df[df['change_in_mean'] < 0.01]
        print(df)
    #if file_config is a string
    if isinstance(file_config, str):
        print("Results:")
        df = pd.read_csv(file_config)
        if params is not None:
            for param in params:
                df[param[1]] = df[param[0]].diff(periods=1)
                df[param[1]] = df[param[1]].abs()
        df = df.sort_values(by=['elbo'], ascending=False)
        # df_new  = df.head(top_n)
        df['grad'] = df['grad'].abs()
        df = df[df['grad'] < 1]
        df_new = df.sort_values(by=['iter'], ascending=True)

        print(df_new)

def get_rows_less_than(df, column, threshold):
    return df[(df[column] < threshold)]

def see_results(file_config, params = None):
    print("Results:")
    df = pd.read_csv(f"{file_config['dir']}/{file_config['result_file']}")
    df = df.sort_values(by=['elbo'], ascending=False)
    # get first row
    main_result = df.head(1)
    for param in params:
        print(f"{param}: {main_result[param].values[0]}")
    return main_result
        # print(main_result[param].values[0])
    
    # print("For reference top 10 results: ")
    # print(df.head(10))

# params_to_compare must be a subset of params
def add_difference_of_params(file, params, params_to_compare = None, threshold_func = get_rows_less_than, threshold=0.01):

    df = pd.read_csv(file)
    print(f"Analysing Early stopping conditions, threshold = {threshold}")
    for param in params:
        df[param[1]] = df[param[0]].diff(periods=1)
        df[param[1]] = df[param[1]].abs()
    df.to_csv(file, index=False)
    # if params_to_compare is not None:
    #     print(params_to_compare)
    #     for param in params_to_compare:
    #         df = threshold_func(df, param[1], threshold)
    #     return df


def edit_sketch(config):
    file_name = f"{config['ard_dir']}/{config['ard_sketch_name']}.ino"
    file = open(file_name, "r")
    code = file.read()
    file.close()

    # Fill all the holes in the sketch
    for key, value in config["holes"].items():
        code = code.replace(key, value)
    
    # Write the new code to the file
    file = open(file_name, "w")
    file.write(code)
    file.close()

    file_lines = open(file_name, "r").readlines()
    

    # make main setup
    for line_num in range(len(file_lines)):
        if '//[ARD:SETUP_DEF]' in file_lines[line_num]:
           file_lines[line_num] = 'void setup()\n'
    

    for line_num in range(len(file_lines)):
        if '//[ARD:REMOVE_BEGIN]' in file_lines[line_num]:

            while '//[ARD:REMOVE_END]' not in file_lines[line_num]:
                file_lines[line_num] = ''
                line_num += 1
    
    for line_num in range(len(file_lines)):
        if '//[ARD:ADD_HEADER]' in file_lines[line_num]:
            file_lines[line_num] = '#include "distributions_latest.hpp"\n'
    


    # if convergence:
    #     for line_num in range(len(file_lines)):
    #         if '//[ARD:CONVERGENCE_BEGIN]' in file_lines[line_num]:
    #             file_lines[line_num] = ''
    #             line_num += 1
    #             while '//[ARD:CONVERGENCE_END]' not in file_lines[line_num]:
    #                 file_lines[line_num] = file_lines[line_num].replace('//', '')
    #                 line_num += 1
    #             file_lines[line_num] = ''
    #             line_num += 1

    for line_num in range(len(file_lines)):
        if '//[ARD:START_TIME]' in file_lines[line_num]:
            file_lines[line_num] = 'unsigned long start_time = millis();\n'
    
    for line_num in range(len(file_lines)):
        if '//[ARD:END_TIME]' in file_lines[line_num]:
            # file_lines[line_num] = f'unsigned long end_time = millis(); \
            #                         unsigned long curr_total_time = end_time - start_time;\
            #                         Serial.print({config["end_time_comment"]});\
            #                         Serial.println((float)curr_total_time);\n'

            file_lines[line_num] = f'unsigned long end_time = millis(); \
                                    unsigned long curr_total_time = end_time - start_time;\n'                 


    
    for line_num in range(len(file_lines)):
        if '//[ARD:SERIAL_MONITOR]' in file_lines[line_num]:
            file_lines[line_num] = 'Serial.begin(9600); while(!Serial){;}\n'
    
    for line_num in range(len(file_lines)):
        if '//[ARD:UNDEFS]' in file_lines[line_num]:
            file_lines[line_num] = '#undef max\n#undef min\n#undef abs\n#undef round\n'
    
    for line_num in range(len(file_lines)):
        if '//[ARD:ADD_LOOP]' in file_lines[line_num]:
            file_lines[line_num] = 'void loop(){}\n'

    line_num = 0
    while line_num < len(file_lines):

        if '//[ARD:UNCOMMENT_BEGIN]' in file_lines[line_num]:
            file_lines[line_num] = ''
            line_num+=1
            while '//[ARD:UNCOMMENT_END]' not in file_lines[line_num]:
                file_lines[line_num] = file_lines[line_num].replace('//', '')
                line_num += 1
            file_lines[line_num] = ''
        line_num += 1
    # for line in file_lines:
        # print(line)
    with open(file_name, "w") as file:
        file.writelines(file_lines)


def convert_to_arduino(config):
    print(f"S2S: Converting to Arduino {config['sketch_name']}")
    dir = config["dir"]
    sketch_name = config["sketch_name"]
    copy_command = f"cp {dir}/{sketch_name}_sketch.cpp {config['ard_dir']}/{config['ard_sketch_name']}.ino"
    os.system(copy_command)
    edit_sketch(config)


def get_run_file(config):
    dir = config["dir"]
    sketch_name = config["sketch_name"]
    copy_command = f"cp {dir}/{sketch_name}_sketch.cpp {dir}/{sketch_name}_run.cpp"
    os.system(copy_command)
    # run file name
    run_file_name = f"{dir}/{sketch_name}_run.cpp"
    run_file = open(run_file_name, "r")
    runfile_code = run_file.read()
    run_file.close()
    # Fill all the holes in the sketch
    for key, value in config["holes"].items():
        runfile_code = runfile_code.replace(key, value)
    
    # Write the new code
    run_file = open(run_file_name, "w")
    run_file.write(runfile_code)
    run_file.close()


def compile_sketch(config):
    dir = config["dir"]
    sketch_name = config["sketch_name"]
    copy_command = f"cp {dir}/{sketch_name}_sketch.cpp {dir}/{sketch_name}_run.cpp"
    os.system(copy_command)
    # run file name
    run_file_name = f"{dir}/{sketch_name}_run.cpp"
    run_file = open(run_file_name, "r")
    runfile_code = run_file.read()
    run_file.close()
    # Fill all the holes in the sketch
    for key, value in config["holes"].items():
        runfile_code = runfile_code.replace(key, value)
    
    # Write the new code
    run_file = open(run_file_name, "w")
    run_file.write(runfile_code)
    run_file.close()

    #compile the file
    compile_config = {
        "dir": dir,
        "file_name": f"{sketch_name}_run.cpp",
        "output_file": f"{sketch_name}_run",
        'misc_flags' : config["misc_flags"]
    }
    compile(compile_config)
    return compile_config

# df["grad_mean"] = df["grad_mean"].abs()
#     df= df[(df["grad_mean"] < 1)]

def run_experiment(config, noresults=False):
    # compile the sketch
    compile_config = compile_sketch(config)
    # run the sketch
    run(compile_config, [f"{config['dir']}/{config['result_file']}"])

    if noresults:
        return
    else:
        df = pd.read_csv(f"{config['dir']}/{config['result_file']}")
        main_result = see_results(config, params = config["params"])
        return main_result
# def fixed_vs_float(fixed_config, float_config):
#     compile_sketch(fixed_config)
#     compile_sketch(float_config)
#     double_config = float_config.copy()
#     double_config["[PY:TYPE]"] = "double"
#     compile_sketch(double_config)
#     run(fixed_config, fixed_config["sketch_name"] + "_result.csv")
#     run(float_config, float_config["sketch_name"] + "_result.csv")
#     run(double_config, double_config["sketch_name"] + "_result.csv")

def check_stopping_conditions(config):
    # compile the sketch
    compile_config = compile_sketch(config)
    # run the sketch
    run(compile_config, [f"{config['dir']}/{config['result_file']}"])

    df = pd.read_csv(f"{config['dir']}/{config['result_file']}")
    # df = df.sort_values(['elbo'], ascending=False)
    # print(df.head(10))


    if "stopping_params" not in config:
        df["grad"] = df["grad"].abs()
        df= df[(df["grad"] < 0.01/(config["scaling"]*config["learning_rate"]))]
        df = df.sort_values(by=['iter'], ascending=True)
        print(df.head(1))
        return df.head(1)
        # print(df.head(10))
    else:
        for param in config["stopping_params"]:
            df[param] = df[param].abs()
            df= df[(df[param] < 0.01/(config["scaling"]*config["learning_rate"]))]
            df = df.sort_values(by=['iter'], ascending=True)
        print(df.head(1))
        return df.head(1)



def run_exp_10_times(config, params = None):
    df = pd.DataFrame()
    for i in range(10):
        temp = run_experiment(config)
        df = df.append(temp)
    if params is not None:
        for param in params:
            print(f"{param}: {df[param].mean()}")
    else:
        print(df)


def run_exp_10_times_with_stopping_conditions(config, params = None):
    df = pd.DataFrame()
    for i in range(10):
        temp = check_stopping_conditions(config)
        df = df.append(temp)
    if params is not None:
        for param in params:
            print(f"{param}: {df[param].mean()}")
    else:
        print(df)


def error_betabinomial(pred, true = 0.46):
    return abs((pred - true)/true)



if __name__ == "__main__":


    bb_max = 1.21068e+06
    linreg_max = 5.32291e+06
    plankton_max = 6.77228e+07
    # temperature_max = 3.19881e+08
    poly1_max = 2.25575e+08
    poly2_max = 5.62501e+06
    kalman1d_max =  1.3877e+08
    hmm_max = 8.73483e+08


    
    exp_names = ["bb", "linreg", "plankton", "poly1", "poly2", "kalman1d"]
    exp_max = [bb_max, linreg_max, plankton_max, poly1_max, poly2_max, kalman1d_max]


    # BETA BINOMIAL EXPERIMENT
    print("BETA BINOMIAL EXPERIMENT")
    bb_float_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/BetaBinomial",
        "sketch_name": "BetaBinomial_float",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]" : "float",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:N]": "1",
            "[PY:DATA]":"0.45",
            "[PY:NUM_EXPERIMENTS]": "50",
        },
        "result_file": "BetaBinomial_float_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.001,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/BetaBinomial/BetaBinomial_float_ard_run",
        "ard_sketch_name": "BetaBinomial_float_ard_run",
        "end_time_comment": '"Total time BetaBinomial float Point:"',
    }

    bb_double_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/BetaBinomial",
        "sketch_name": "BetaBinomial_double",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]" : "double",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:N]": "1",
            "[PY:DATA]":"0.45",
            "[PY:NUM_EXPERIMENTS]": "50",
        },
        "result_file": "BetaBinomial_double_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.001,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/BetaBinomial/BetaBinomial_double_ard_run",
        "ard_sketch_name": "BetaBinomial_double_ard_run",
        "end_time_comment": '"Total time BetaBinomial Double:"',
    }

    bb_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/BetaBinomial",
        "sketch_name": "BetaBinomial_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:FRAC]": "21",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:N]": "1",
            "[PY:DATA]":"0.45",
            "[PY:SCALING]": "2",
            "[PY:NUM_EXPERIMENTS]": "50"
        },
        "result_file": "BetaBinomial_fixed_result.csv",
        "params":["iter", "mean", "sigma"],
        "learning_rate": 0.01,
        "scaling": 64,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/BetaBinomial/BetaBinomial_fixed_ard_run",
        "ard_sketch_name": "BetaBinomial_fixed_ard_run",
        "end_time_comment": '"Total time BetaBinomial Fixed Point:"',
    }
    # convert_to_arduino(bb_fixed_dict)
    # convert_to_arduino(bb_float_dict)
    # convert_to_arduino(bb_double_dict)
    run_experiment(bb_fixed_dict)
    # run_experiment(bb_float_dict)
    # run_exp_10_times(bb_fixed_dict, ["mean"])
    # run_exp_10_times(bb_float_dict, ["mean"])
    # run_exp_10_times(bb_double_dict, ["mean"])

    # result = run_experiment(bb_float_dict)
    
    # result = run_experiment(bb_double_dict)
    # error = error_betabinomial(result["mean"].values[0], float(bb_double_dict["holes"]["[PY:DATA]"]))
    # print("Geomean error: ", error)

    # result = run_experiment(bb_fixed_dict)
    # check_stopping_conditions(bb_fixed_dict)
    # run_exp_10_times_with_stopping_conditions(bb_fixed_dict, ["mean"])
    # error = error_betabinomial(result["mean"].values[0], float(bb_fixed_dict["holes"]["[PY:DATA]"]))
    # print("Geomean error: ", error)


    # TEMPERATURE
    print("TEMPERATURE EXPERIMENT")

    temperature_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Temperature",
        "sketch_name": "temperature_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:FRAC]": "11",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:SCALING]": "4",
            "[PY:NUM_EXPERIMENTS]": "50",
            "[PY:NUM_EXP]":"50"
        },
        "result_file": "temperature_fixed_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.1,
        "scaling": 64,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Temperature/Temperature_fixed_ard_run",
        "ard_sketch_name": "Temperature_fixed_ard_run",
        "end_time_comment": '"Total time Temperature Fixed Point:"',

    }

    temperature_float_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Temperature",
        "sketch_name": "temperature_float",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]": "float",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:NUM_EXPERIMENTS]": "50",
        }
        ,
        "result_file": "temperature_float_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.1,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Temperature/Temperature_float_ard_run",
        "ard_sketch_name": "Temperature_float_ard_run",
        "end_time_comment": '"Total time Temperature Float Point:"',

    }

    temperature_double_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Temperature",
        "sketch_name": "temperature_double",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]": "double",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:NUM_EXPERIMENTS]": "50",

        }
        ,
        "result_file": "temperature_double_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.1,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Temperature/Temperature_double_ard_run",
        "ard_sketch_name": "Temperature_double_ard_run",
        "end_time_comment": '"Total time Temperature double Point:"',
    }

    # convert_to_arduino(temperature_fixed_dict)
    convert_to_arduino(temperature_float_dict)
    convert_to_arduino(temperature_double_dict)
    # run_experiment(temperature_fixed_dict)
    # run_exp_10_times(temperature_fixed_dict, ["mean"])
    # run_exp_10_times(temperature_float_dict, ["mean"])
    # run_exp_10_times(temperature_double_dict, ["mean"])
    # run_experiment(temperature_float_dict)
    # run_experiment(temperature_double_dict)

    # run_exp_10_times_with_stopping_conditions(temperature_fixed_dict, ["mean"])



    # LIGHTSPEED

    print("LIGHTSPEED EXPERIMENT")
    lightspeed_float_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Lightspeed",
            "sketch_name": "lightspeed_float",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:TYPE]": "float",
                "[PY:NUM_SAMPLES]" : "8",
                "[PY:NUM_ITERS]" : "1000",
                "[PY:LEARNING_RATE]" : "0.001",
            },
            "result_file": "lightspeed_float_result.csv",
            "params":["iter","mean", "sigma"],
            "learning_rate": 0.01,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Lightspeed/Lightspeed_float_ard_run",
            "ard_sketch_name": "Lightspeed_float_ard_run",
            "end_time_comment": '"Total time Lightspeed Float Point:"',
        }

    lightspeed_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Lightspeed",
        "sketch_name": "lightspeed_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:FRAC]": "15",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1000",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:SCALING]": "16"
        },
        "result_file": "lightspeed_fixed_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Lightspeed/Lightspeed_fixed_ard_run",
        "ard_sketch_name": "Lightspeed_fixed_ard_run",
        "end_time_comment": '"Total time Lightspeed fixed Point:"',
    }

    # run_experiment(lightspeed_float_dict)
    # run_experiment(lightspeed_fixed_dict)
    # run_exp_10_times(lightspeed_float_dict, ["mean"])

    # PLANKTON EXPERIMENT
    print("PLANKTON EXPERIMENT")

    plankton_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Plankton",
        "sketch_name": "plankton_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:FRAC]": "12",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:SCALING]": "4",
            "[PY:DATA]":"12.914, 14.0088, 14.339",
            "[PY:NUM_EXPERIMENTS]":"50"
        },
        "result_file": "plankton_fixed_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.1,
        "scaling": 512,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Plankton/Plankton_fixed_ard_run",
        "ard_sketch_name": "Plankton_fixed_ard_run",
        "end_time_comment": '"Total time Plankton Fixed Point:"',
    }

    plankton_float_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Plankton",
        "sketch_name": "plankton_float",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]": "float",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:DATA]":"12.914, 14.0088, 14.339",
            "[PY:NUM_EXPERIMENTS]":"50"
        },
        "result_file": "plankton_float_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Plankton/Plankton_float_ard_run",
        "ard_sketch_name": "Plankton_float_ard_run",
        "end_time_comment": '"Total time Plankton Float Point:"',
    }

    plankton_double_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Plankton",
        "sketch_name": "plankton_double",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]": "double",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:DATA]":"12.914, 14.0088, 14.339",
            "[PY:NUM_EXPERIMENTS]":"50", 

        },
        "result_file": "plankton_double_result.csv",
        "params":["iter","mean", "sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Plankton/Plankton_double_ard_run",
        "ard_sketch_name": "Plankton_double_ard_run",
        "end_time_comment": '"Total time Plankton double Point:"',
    }

    # convert_to_arduino(plankton_float_dict)
    # convert_to_arduino(plankton_fixed_dict)
    # convert_to_arduino(plankton_double_dict)
    # run_experiment(plankton_fixed_dict)
    # run_experiment(plankton_float_dict)
    # run_exp_10_times(plankton_fixed_dict, ["mean"])
    # run_exp_10_times(plankton_float_dict, ["mean"])
    # run_exp_10_times(plankton_double_dict, ["mean"])
    # run_experiment(plankton_float_dict)
    # run_experiment(plankton_double_dict)

    # check_stopping_conditions(plankton_fixed_dict)
    # run_exp_10_times_with_stopping_conditions(plankton_fixed_dict, ["mean"])


    # UNEMPLOYMENT EXPERIMENT
    print("UNEMPLOYMENT EXPERIMENT")

    unemployment_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Unemployment",
        "sketch_name": "Unemployment_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:FRAC]": "16",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "5000",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:BITSHIFT]": "128",
        },
        "result_file": "unemployment_fixed_result.csv",
        "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma", "sigma_mu","sigma_sigma"],
        "learning_rate": 0.01,
        "stopping_params":["grad_a0","grad_a1"],
        "scaling": 512,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Unemployment/Unemployment_fixed_ard_run",
        "ard_sketch_name": "Unemployment_fixed_ard_run",
        "end_time_comment": '"Total time Unemployment Fixed Point:"',
    }

    unemployment_float_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Unemployment",
        "sketch_name": "Unemployment_float",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]": "float",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "5000",
            "[PY:LEARNING_RATE]" : "0.0001",
        },
        "result_file": "unemployment_float_result.csv",
        "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma", "sigma_mu","sigma_sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Unemployment/Unemployment_float_ard_run",
        "ard_sketch_name": "Unemployment_float_ard_run",
        "end_time_comment": '"Total time Unemployment Float Point:"',
    }

    unemployment_double_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Unemployment",
        "sketch_name": "Unemployment_double",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:TYPE]": "double",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "5000",
            "[PY:LEARNING_RATE]" : "0.0001",
        },
        "result_file": "unemployment_double_result.csv",
        "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma", "sigma_mu","sigma_sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Unemployment/Unemployment_double_ard_run",
        "ard_sketch_name": "Unemployment_double_ard_run",
        "end_time_comment": '"Total time Unemployment double Point:"',
    }

    # convert_to_arduino(unemployment_fixed_dict)
    # convert_to_arduino(unemployment_float_dict)
    # convert_to_arduino(unemployment_double_dict)

    # run_experiment(unemployment_fixed_dict)
    # check_stopping_conditions(unemployment_fixed_dict)
    # run_experiment(unemployment_float_dict)
    # run_experiment(unemployment_double_dict)
    # check_stopping_conditions(unemployment_fixed_dict)
    # run_exp_10_times_with_stopping_conditions(unemployment_fixed_dict, ["a0_mu","a1_mu","sigma_mu"])
    # run_exp_10_times(unemployment_fixed_dict, ["a0_mu","a1_mu","sigma_mu"])
    # run_experiment(unemployment_fixed_dict)


    # LinReg
    print("LinReg EXPERIMENT")
    # for 50 data points PY:FRAC = 14, scaling = 7
    # for 5000 data points PY:FRAC = 12, scaling 7
    # for 10000 data points PY:FRAC = 10, scaling 7
    linreg_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg",
        "sketch_name": "LinReg_a0_x_p_b_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:NUM_DATA_SAMPLES]" : "50",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:FRAC]": "8",
            "[PY:BITSHIFT]":"3",
            "[PY:NUM_SAMPLES]":"4",
            "[PY:NUM_EXPERIMENTS]" : "50",
            "[PY:TEST_SET_SIZE]" : "50"
        },
        "result_file": "linreg_fixed_result.csv",
        "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma"],
        "learning_rate": 0.01,
        "scaling": 256,
        "stopping_params":["grad_a0","grad_a1"],
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg/LinReg_a0_x_p_b_fixed_ard_run",
        "ard_sketch_name": "LinReg_a0_x_p_b_fixed_ard_run",
        "end_time_comment": '"Total time LinReg Fixed Point:"',
    }

    linreg_float_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg",
        "sketch_name": "LinReg_a0_x_p_b_float_wo_scaling",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:NUM_DATA_SAMPLES]" : "50",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:TYPE]" : "float",
            "[PY:NUM_SAMPLES]":"4",
            "[PY:NUM_EXPERIMENTS]" : "50",
            "[PY:TEST_SET_SIZE]" : "50"
        },
        "result_file": "linreg_float_result.csv",
        "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma"],
        "learning_rate": 0.001,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg/LinReg_a0_x_p_b_float_wo_scaling_ard_run",
        "ard_sketch_name": "LinReg_a0_x_p_b_float_wo_scaling_ard_run",
        "end_time_comment": '"Total time LinReg Float Point:"',
    }

    linreg_double_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg",
        "sketch_name": "LinReg_a0_x_p_b_double_wo_scaling",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:NUM_DATA_SAMPLES]" : "50",
            "[PY:NUM_ITERS]" : "1500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:TYPE]" : "double",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_EXPERIMENTS]" : "2",
            "[PY:TEST_SET_SIZE]" : "50"
        },
        "result_file": "linreg_double_result.csv",
        "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma"],
        "learning_rate": 0.001,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg/LinReg_a0_x_p_b_double_wo_scaling_ard_run",
        "ard_sketch_name": "LinReg_a0_x_p_b_double_wo_scaling_ard_run",
        "end_time_comment": '"Total time LinReg Double Point:"',
    }

    ############### LinReg Fixed Point Table 2################
    # data_size = [50, 200, 500, 1000, 8000, 16000]

    # data_size = [500, 600, 700, 800, 900, 1000]

    # data_size = [710] * 10

    # frac_linreg_vanilla = [8] * len(data_size)
    # bitshift_linreg_vanilla = [0] * len(data_size)

    # # frac_linreg = [10, 9 , 8,8,8,8]
    # # bitshift_linreg = [2,3,3,4,7,8]

    # for i in range(len(data_size)):
    #     print("Data Size: ", data_size[i])
    #     linreg_fixed_dict["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(data_size[i])
    #     linreg_fixed_dict["holes"]["[PY:FRAC]"] = str(frac_linreg_vanilla[i])
    #     linreg_fixed_dict["holes"]["[PY:BITSHIFT]"] = str(bitshift_linreg_vanilla[i])
    #     # linreg_fixed_dict["result_file"] = "linreg_fixed_result_"+str(data_size[i])+".csv"
    #     run_experiment(linreg_fixed_dict)
    #################

    # convert_to_arduino(linreg_fixed_dict)
    # convert_to_arduino(linreg_float_dict)
    # convert_to_arduino(linreg_double_dict)

    # run_experiment(linreg_fixed_dict)
    # run_experiment(linreg_float_dict)
    # run_experiment(linreg_double_dict)

    # run_exp_10_times(linreg_fixed_dict, ["a0_mu","a0_sigma", "a1_mu","a1_sigma"])
    # run_exp_10_times(linreg_float_dict, ["a0_mu","a0_sigma", "a1_mu","a1_sigma"])
    # run_exp_10_times(linreg_double_dict, ["a0_mu","a0_sigma", "a1_mu","a1_sigma"])

    # check_stopping_conditions(linreg_fixed_dict)
    # run_exp_10_times_with_stopping_conditions(linreg_fixed_dict, ["a0_mu","a1_mu"])
    # run_exp_10_times(linreg_fixed_dict, ["a0_mu","a1_mu"])
    print("Poly1 EXPERIMENT")
    poly1_fixed_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly1_fixed",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "50",
                "[PY:LEARNING_RATE]" : "0.01",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1500",
                "[PY:FRAC]": "8",
                "[PY:BITSHIFT]":"8",
                "[PY:NUM_EXPERIMENTS]" : "50",
                "[PY:TEST_SET_SIZE]" : "50",
            },
            "result_file": "poly1_fixed_result.csv",
            "params":["iter","mean","sigma"],
            "learning_rate": 0.01,
            "scaling": 512,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly1_fixed_ard_run",
            "ard_sketch_name": "poly1_fixed_ard_run",
            "end_time_comment": '"Total time Poly1 Fixed Point:"',
            
        }
    
    poly1_float_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly1_float",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "50",
                "[PY:LEARNING_RATE]" : "0.01",
                "[PY:TYPE]" : "float",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1500",
                "[PY:NUM_EXPERIMENTS]" : "4",
                "[PY:TEST_SET_SIZE]" : "50",
            },
            "result_file": "poly1_float_result.csv",
            "params":["iter","mean","sigma"],
            "learning_rate": 0.01,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly1_float_ard_run",
            "ard_sketch_name": "poly1_float_ard_run",
            "end_time_comment": '"Total time Poly1 Float Point:"',
    }

    poly1_double_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly1_double",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "50",
                "[PY:LEARNING_RATE]" : "0.01",
                "[PY:TYPE]" : "double",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1500",
                "[PY:NUM_EXPERIMENTS]" : "2",
                "[PY:TEST_SET_SIZE]" : "50",
            },
            "result_file": "poly1_double_result.csv",
            "params":["iter","mean","sigma"],
            "learning_rate": 0.01,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly1_double_ard_run",
            "ard_sketch_name": "poly1_double_ard_run",
            "end_time_comment": '"Total time Poly1 Double Point:"',
    }


    ################# POLY1 FIXED POINT EXPERIMENT #################
    # data_size = [50, 200, 500, 1000, 8000, 16000]

    # data_size = [225]*50

    # frac_linreg = [8] * len(data_size)
    # bitshift_linreg = [0] * len(data_size)
    # # frac_linreg = [8, 8 , 8,8,8,8]
    # # bitshift_linreg = [5,7,8,8,8,8]

    # for i in range(len(data_size)):
    #     print("Data Size: ", data_size[i])
    #     poly1_fixed_dict["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(data_size[i])
    #     poly1_fixed_dict["holes"]["[PY:FRAC]"] = str(frac_linreg[i])
    #     poly1_fixed_dict["holes"]["[PY:BITSHIFT]"] = str(bitshift_linreg[i])
    #     run_experiment(poly1_fixed_dict)

        # linreg_fixed_dict["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(data_size[i])
        # linreg_fixed_dict["holes"]["[PY:FRAC]"] = str(frac_linreg[i])
        # linreg_fixed_dict["holes"]["[PY:BITSHIFT]"] = str(bitshift_linreg[i])
        # # linreg_fixed_dict["result_file"] = "linreg_fixed_result_"+str(data_size[i])+".csv"
        # run_experiment(linreg_fixed_dict)






    # convert_to_arduino(poly1_fixed_dict)
    # convert_to_arduino(poly1_float_dict)
    # convert_to_arduino(poly1_double_dict)

    # run_experiment(poly1_fixed_dict)

    # run_experiment(poly1_fixed_dict)
    # run_experiment(poly1_float_dict)
    # run_experiment(poly1_double_dict)
    # check_stopping_conditions(poly1_fixed_dict)

    # run_exp_10_times_with_stopping_conditions(poly1_fixed_dict, ["mean","sigma"])
    # run_exp_10_times(poly1_fixed_dict, ["mean","sigma"])
    # run_exp_10_times(poly1_float_dict, ["mean","sigma"])
    # run_exp_10_times(poly1_double_dict, ["mean","sigma"])


    print("Kalman1D EXPERIMENT")
    kalman1d_double_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Kalman1D",
        "sketch_name": "kalman1d_double",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:NUM_DATA_SAMPLES]" : "500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:TYPE]" : "double",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "200",
            "[PY:NUM_EXPERIMENTS]" : "1",
            "[PY:BURN_IN]" : "0",
        },
        "result_file": "kalman1d_double_result.csv",
        "params":["iter","mean","sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Kalman1D/kalman1d_double_ard_run",
        "ard_sketch_name": "kalman1d_double_ard_run",
        "end_time_comment": '"Total time Temperature double Point:"',
    }


    kalman1d_fixed_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Kalman1D",
        "sketch_name": "kalman1d_fixed",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:NUM_DATA_SAMPLES]" : "500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:TYPE]" : "fpm::fixed<std::int32_t, std::int64_t, 10>",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "200",
            "[PY:NUM_EXPERIMENTS]" : "1",
            "[PY:BURN_IN]" : "0",
            "[PY:SCALING]" : "8"
        },
        "result_file": "kalman1d_fixed_result.csv",
        "params":["iter","mean","sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Kalman1D/kalman1d_fixed_ard_run",
        "ard_sketch_name": "kalman1d_fixed_ard_run",
        "end_time_comment": '"Total time Temperature fixed Point:"',
    }


    kalman1d_float_dict = {
        "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Kalman1D",
        "sketch_name": "kalman1d_float",
        "misc_flags": ["-O3"],
        "holes": {
            "[PY:NUM_DATA_SAMPLES]" : "500",
            "[PY:LEARNING_RATE]" : "0.01",
            "[PY:TYPE]" : "float",
            "[PY:NUM_SAMPLES]" : "4",
            "[PY:NUM_ITERS]" : "200",
            "[PY:NUM_EXPERIMENTS]" : "1",
            "[PY:BURN_IN]" : "50",
        },
        "result_file": "kalman1d_float_result.csv",
        "params":["iter","mean","sigma"],
        "learning_rate": 0.01,
        "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/Kalman1D/kalman1d_float_ard_run",
        "ard_sketch_name": "kalman1d_float_ard_run",
        "end_time_comment": '"Total time Temperature float Point:"',
    }

    # run_experiment(kalman1d_double_dict, noresults=True)
    # run_experiment(kalman1d_float_dict, noresults=True)
    # run_experiment(kalman1d_fixed_dict, noresults=True)
    # convert_to_arduino(kalman1d_fixed_dict)
    # convert_to_arduino(kalman1d_double_dict)
    # run_experiment(kalman1d_float_dict, noresults=True)
    # convert_to_arduino(kalman1d_float_dict)

    # POLY2
    print("POLY2 EXPERIMENT")

    poly2_fixed_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly2_fixed",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "50",
                "[PY:LEARNING_RATE]" : "0.01",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1500",
                "[PY:FRAC]": "8",
                "[PY:BITSHIFT]":"3",
                "[PY:NUM_EXPERIMENTS]" : "10",
                "[PY:TEST_SET_SIZE]" : "50",
            },
            "result_file": "poly2_fixed_result.csv",
            "params":["iter","a0_mean", "a0_sigma", "a1_mean", "a1_sigma"],
            "learning_rate": 0.01,
            "scaling": 512,
            "stopping_params":["grad_a0","grad_a1"],
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly2_fixed_ard_run",
            "ard_sketch_name": "poly2_fixed_ard_run",
            "end_time_comment": '"Total time Poly2 Fixed Point:"',
    }

    poly2_float_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly2_float",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "50",
                "[PY:LEARNING_RATE]" : "0.01",
                "[PY:TYPE]" : "float",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1500",
                "[PY:NUM_EXPERIMENTS]" : "5",
                "[PY:TEST_SET_SIZE]" : "50",
            },
            "result_file": "poly2_float_result.csv",
            "params":["iter","a0_mean", "a0_sigma", "a1_mean", "a1_sigma"],
            "learning_rate": 0.01,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly2_float_ard_run",
            "ard_sketch_name": "poly2_float_ard_run",
            "end_time_comment": '"Total time Poly2 Float Point:"',
    }

    poly2_double_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly2_double",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "50",
                "[PY:LEARNING_RATE]" : "0.01",
                "[PY:TYPE]" : "double",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1500",
                "[PY:NUM_EXPERIMENTS]" : "2",
                "[PY:TEST_SET_SIZE]" : "50",
            },
            "result_file": "poly2_double_result.csv",
            "params":["iter","a0_mean", "a0_sigma", "a1_mean", "a1_sigma"],
            "learning_rate": 0.01,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly2_double_ard_run",
            "ard_sketch_name": "poly2_double_ard_run",
            "end_time_comment": '"Total time Poly2 Double Point:"',
    }

    data_size = [50, 200, 500, 1000, 8000, 16000]
    data_size =[ 1000, 2000, 4000, 6000, 8000]
    data_size = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    data_size = [1000, 1050, 1100]
    data_size = [1000] * 50
    data_size = [500, 750]
    data_size = [500, 625, 750]
    data_size = [563]*10
    frac_linreg = [8] * len(data_size)
    bitshift_linreg = [0] * len(data_size)
    # frac_linreg = [10, 9 , 8,8,8,8]
    # bitshift_linreg = [2,2,3,4,7,8]

    # for i in range(len(data_size)):
    #     print("Data Size: ", data_size[i])
    #     poly2_fixed_dict["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(data_size[i])
    #     poly2_fixed_dict["holes"]["[PY:FRAC]"] = str(frac_linreg[i])
    #     poly2_fixed_dict["holes"]["[PY:BITSHIFT]"] = str(bitshift_linreg[i])
    #     run_experiment(poly2_fixed_dict)

    # convert_to_arduino(poly2_fixed_dict)
    # convert_to_arduino(poly2_float_dict)
    # convert_to_arduino(poly2_double_dict)

    # check_stopping_conditions(poly2_fixed_dict)
    # run_experiment(poly2_fixed_dict)
    # run_experiment(poly2_float_dict)
    # # run_exp_10_times(poly2_double_dict, ["a0_mean", "a0_sigma", "a1_mean", "a1_sigma"])
    # run_experiment(poly2_double_dict)
    # run_experiment(poly2_double_dict)


    # run_exp_10_times(poly2_double_dict, ["a0_mean", "a0_sigma", "a1_mean", "a1_sigma"])
    # run_exp_10_times_with_stopping_conditions(poly2_fixed_dict, ["a0_mean", "a1_mean"])
    # run_exp_10_times(poly2_fixed_dict, ["a0_mean", "a1_mean"])
    # run_exp_10_times(poly2_float_dict, ["a0_mean", "a1_mean"])

    # POLY3 
    print("POLY3 EXPERIMENT")
    poly3_float_dict = {
            "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
            "sketch_name": "poly3_float",
            "misc_flags": ["-O3"],
            "holes": {
                "[PY:NUM_DATA_SAMPLES]" : "1000",
                "[PY:LEARNING_RATE]" : "0.001",
                "[PY:TYPE]" : "float",
                "[PY:NUM_SAMPLES]" : "4",
                "[PY:NUM_ITERS]" : "1000",
            },
            "result_file": "poly3_float_result.csv",
            "params":["iter","a0_mean", "a0_sigma", "a1_mean", "a1_sigma"],
            "learning_rate": 0.01,
            "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly3_float_ard_run",
            "ard_sketch_name": "poly3_float_ard_run",
            "end_time_comment": '"Total time Poly3 Float Point:"',
    }

    # run_experiment(poly3_float_dict)
    # run_exp_10_times(poly3_float_dict, ["a0_mean", "a0_sigma", "a1_mean", "a1_sigma"])

    linreg_data =   [50, 200, 500, 2000, 4000, 6000, 8000, 10000, 16000]
    bitprofiles_linreg =[(7,15,16),(7,15,16),(7,15,16), (7,15,16), (8,16,15), (8,7,14),(8,7,14),(8,7,14),(8,18,13)]
    poly1reg_data = [50, 200, 500,2000, 4000, 6000, 8000, 10000, 16000]
    bitprofiles_poly1reg = [(9,19,12),(9,19,12),(9,19,12),(9,19,12),(9,20,11), (9,20,11),(9,21,10),(9,21,11), (9,21,11)]
    poly2reg_data = [50, 200, 500,2000, 4000, 6000, 8000, 10000, 16000]
    bitprofiles_poly2reg = [(9,19,12),(9,19,12),(9,19,12),(9,19,12),(9,20,11), (9,20,11),(9,21,10),(9,21,11), (9,22,9)]


    # linreg_fixed_data_exp = {
    #     "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg",
    #     "sketch_name": "LinReg_a0_x_p_b_fixed",
    #     "misc_flags": ["-O3"],
    #     "holes": {
    #         "[PY:NUM_DATA_SAMPLES]" : "500",
    #         "[PY:NUM_ITERS]" : "5000",
    #         "[PY:LEARNING_RATE]" : "0.01",
    #         "[PY:FRAC]": "16",
    #         "[PY:BITSHIFT]":"7",
    #         "[PY:NUM_SAMPLES]":"4",
    #     },
    #     "result_file": "linreg_fixed_result.csv",
    #     "params":["iter","a0_mu","a0_sigma", "a1_mu","a1_sigma"],
    #     "learning_rate": 0.01,
    #     "scaling": 256,
    #     "stopping_params":["grad_a0","grad_a1"],
    #     "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/LinReg/LinReg_a0_x_p_b_fixed_ard_run",
    #     "ard_sketch_name": "LinReg_a0_x_p_b_fixed_ard_run",
    #     "end_time_comment": '"Total time LinReg Fixed Point:"',
    # }

    # results_linreg = pd.DataFrame()

    # for i in range(len(linreg_data)):
    #     linreg_fixed_data_exp["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(linreg_data[i])
    #     linreg_fixed_data_exp["holes"]["[PY:BITSHIFT]"] = str(bitprofiles_linreg[i][0])
    #     linreg_fixed_data_exp["holes"]["[PY:FRAC]"] = str(bitprofiles_linreg[i][2])
    #     results_linreg = results_linreg.append(run_experiment(linreg_fixed_data_exp))
    
    # results_linreg.to_csv("./results_linreg_data_experiment.csv")
    # print(results_linreg)




    # poly1_fixed_data_exp = {
    #         "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
    #         "sketch_name": "poly1_fixed",
    #         "misc_flags": ["-O3"],
    #         "holes": {
    #             "[PY:NUM_DATA_SAMPLES]" : "1000",
    #             "[PY:LEARNING_RATE]" : "0.01",
    #             "[PY:NUM_SAMPLES]" : "4",
    #             "[PY:NUM_ITERS]" : "5000",
    #             "[PY:FRAC]": "12",
    #             "[PY:BITSHIFT]":"10",

    #         },
    #         "result_file": "poly1_fixed_result.csv",
    #         "params":["iter","mean","sigma"],
    #         "learning_rate": 0.01,
    #         "scaling": 512,
    #         "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly1_fixed_ard_run",
    #         "ard_sketch_name": "poly1_fixed_ard_run",
    #         "end_time_comment": '"Total time Poly1 Fixed Point:"',
            
    #     }

    # results_poly1 = pd.DataFrame()

    # for i in range(len(poly1reg_data)):
    #     poly1_fixed_data_exp["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(poly1reg_data[i])
    #     poly1_fixed_data_exp["holes"]["[PY:BITSHIFT]"] = str(bitprofiles_poly1reg[i][0])
    #     poly1_fixed_data_exp["holes"]["[PY:FRAC]"] = str(bitprofiles_poly1reg[i][2])
    #     results_poly1 = results_poly1.append(run_experiment(poly1_fixed_data_exp))
    
    # results_poly1.to_csv("./results_poly1_data_experiment.csv")
    # print(results_poly1)








    # poly2_fixed_data_exp = {
    #         "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
    #         "sketch_name": "poly2_fixed",
    #         "misc_flags": ["-O3"],
    #         "holes": {
    #             "[PY:NUM_DATA_SAMPLES]" : "1000",
    #             "[PY:LEARNING_RATE]" : "0.01",
    #             "[PY:NUM_SAMPLES]" : "4",
    #             "[PY:NUM_ITERS]" : "5000",
    #             "[PY:FRAC]": "13",
    #             "[PY:BITSHIFT]":"9",
    #         },
    #         "result_file": "poly2_fixed_result.csv",
    #         "params":["iter","a0_mean", "a0_sigma", "a1_mean", "a1_sigma"],
    #         "learning_rate": 0.01,
    #         "scaling": 512,
    #         "stopping_params":["grad_a0","grad_a1"],
    #         "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly2_fixed_ard_run",
    #         "ard_sketch_name": "poly2_fixed_ard_run",
    #         "end_time_comment": '"Total time Poly2 Fixed Point:"',

    # }

    # results_poly2 = pd.DataFrame()

    # for i in range(len(poly2reg_data)):
    #     poly2_fixed_data_exp["holes"]["[PY:NUM_DATA_SAMPLES]"] = str(poly2reg_data[i])
    #     poly2_fixed_data_exp["holes"]["[PY:BITSHIFT]"] = str(bitprofiles_poly2reg[i][0])
    #     poly2_fixed_data_exp["holes"]["[PY:FRAC]"] = str(bitprofiles_poly2reg[i][2])
    #     results_poly2 = results_poly2.append(run_experiment(poly2_fixed_data_exp))

    # results_poly2.to_csv("./results_poly2_data_experiment.csv")
    # print(results_poly2)







    # poly2_fixed_plot_config = {
    #         "dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg",
    #         "sketch_name": "poly2_fixed",
    #         "misc_flags": ["-O3"],
    #         "holes": {
    #             "[PY:NUM_DATA_SAMPLES]" : "2000",
    #             "[PY:LEARNING_RATE]" : "0.01",
    #             "[PY:NUM_SAMPLES]" : "4",
    #             "[PY:NUM_ITERS]" : "5000",
    #             "[PY:FRAC]": "12",
    #             "[PY:BITSHIFT]":"9",
    #         },
    #         "result_file": "poly2_fixed_result.csv",
    #         "params":["iter","a0_mean", "a0_sigma", "a1_mean", "a1_sigma"],
    #         "learning_rate": 0.01,
    #         "scaling": 512,
    #         "stopping_params":["grad_a0","grad_a1"],
    #         "ard_dir": "/home/ashitabh/Documents/fpm_test_code/Benchmarks/PolyReg/poly2_fixed_ard_run",
    #         "ard_sketch_name": "poly2_fixed_ard_run",
    #         "end_time_comment": '"Total time Poly2 Fixed Point:"',

    # }


    # get_run_file(poly2_fixed_plot_config)
    # alpha_values =[0.9]
    # fixed_exp = [bb_fixed_dict, linreg_fixed_dict, plankton_fixed_dict, poly1_fixed_dict, poly2_fixed_dict, kalman1d_fixed_dict] 

    # for alpha in alpha_values:
    #     print("alpha: ", alpha)
    #     for exp, max, fixed_dict in zip(exp_names, exp_max, fixed_exp):
    #         I, K, F = get_virtual_bits(max, alpha)
    #         if "holes" in fixed_dict:
    #             if "[PY:BITSHIFT]" in fixed_dict["holes"]:
    #                 fixed_dict["holes"]["[PY:BITSHIFT]"] = str(K)
    #             if "[PY:FRAC]" in fixed_dict["holes"]:
    #                 fixed_dict["holes"]["[PY:FRAC]"] = str(F)
    #             if "[PY:SCALING]" in fixed_dict["holes"]:
    #                 fixed_dict["holes"]["[PY:SCALING]"] = str(2**K)

    #         print("*"*50)
    #         print("Experiment: ", exp)
    #         print("*"*50)
    #         print ( exp, "I: ", I, "K: ", K, "F: ", F)
    #         if exp == "kalman1d":
    #             run_experiment(fixed_dict, noresults=True)
    #         else:
    #             run_experiment(fixed_dict)
    #     break



