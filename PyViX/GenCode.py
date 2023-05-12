from distributions import *


def main_body(input):
    headers = ["\"../include/distributions_latest.hpp\"", "\"../include/ios.hpp\"", "<iostream>", "<fstream>", "<random>", "<array>"]
    header_string = [f'#include {header}\n' for header in headers]


    definition = 'int main(int argc, char *argv[]) {\n'
    output_list = header_string + [definition] + input + ['\treturn 0;\n}']

    return output_list


def test_print(input):
    [print(line) for line in input]



def gen_assign_constant(object_input,value, running_type):
    # assert (isinstance(object, Constant))
    return f'Constant<{running_type}> {object_input}({value});\n'


def create_for_loop(body, init = 0, end = 1000, name_counter = 'c_iter'):
    # print("Loop body:")
    # print(body)
    return [f"for(int {name_counter} = {init}; {name_counter} < {end}; {name_counter}++) \n" + '{\n'] + body + ["\n}\n"]


def gen_while_true(body):
    return ["while(true) {\n"] + body + ["\n}\n"]

def gen_assignment(lhs, rhs):
    return f'{lhs} = {rhs};\n'

def write_step_update(priors, running_type):
    step_update_list = []
    for prior in priors:
        prior_name = prior.get_string()
        prior_learnable_params = prior.learnable_params

        if len(prior_learnable_params) >0:
            # f"step_mu_{prior_name} = grad_clipping<{running_type}>(current_del_elbo_mu_{prior_name})/num_samples, LB, UB);\n"
            step_update_list.append(f"step_mu_{prior_name} = grad_clipping<{running_type}>((del_elbo_mu_{prior_name}/num_samples), LB, UB);\n")
            if len(prior_learnable_params) > 1:
                step_update_list.append(f"step_sigma_{prior_name} = grad_clipping<{running_type}>((del_elbo_sigma_{prior_name}/num_samples), LB, UB);\n")
    
    return step_update_list

def new_values(priors):
    new_values_list = []
    for prior in priors:
        prior_learnables = prior.learnable_params
        if len(prior_learnables) > 0:
            prior_name = prior.get_string()
            new_values_list.append(f"new_mu_{prior_name} = {prior.posterior.mu.name}.value() + learning_rate *(step_mu_{prior_name});\n")
            if len(prior_learnables) > 1:
                new_values_list.append(f"new_sigma_{prior_name} = {prior.posterior.sigma.name}.value() + learning_rate *(step_sigma_{prior_name});\n")

    return new_values_list


def final_update_containers(priors, running_type):
    final_update_list = []
    for prior in priors:
        prior_learnables = prior.learnable_params
        if len(prior_learnables) > 0:
            prior_learnables = prior.learnable_params
            prior_name = prior.get_string()

            final_update_list.append(f"{prior.posterior.mu.name}(grad_clipping<{running_type}>(new_mu_{prior_name}, {prior_name}.lb_value(), {prior_name}.ub_value()));\n")
            if len(prior_learnables) > 1:
                final_update_list.append(f"{prior.posterior.sigma.name}(grad_clipping<{running_type}>(new_sigma_{prior_name}, 0.1 , 5));\n")


    return final_update_list


def get_best_assign(priors, running_type):
    get_best_assign_list = []
    for prior in priors:
        prior_learnables = prior.learnable_params

        if len(prior_learnables) > 0:
            prior_name = prior.get_string()
            get_best_assign_list.append(f"{running_type} best_mu_{prior_name} = 0;\n")
            if len(prior_learnables) > 1:
                get_best_assign_list.append(f"{running_type} best_sigma_{prior_name} = 0;\n")

    return get_best_assign_list


def update_best_post_values(priors):
    best_post_values = []
    if_str = "if (total_elbo /num_samples  > max_elbo) {\n max_elbo = total_elbo /num_samples;\n"
    best_post_values.append(if_str)
    for prior in priors:
        prior_learnables = prior.learnable_params
        if len(prior_learnables) > 0:
            prior_name = prior.get_string()
            best_post_values.append(f"best_mu_{prior_name} = {prior.posterior.mu.name}.value();\n")
            if len(prior_learnables) > 1:
                best_post_values.append(f"best_sigma_{prior_name} = {prior.posterior.sigma.name}.value();\n")
    best_post_values.append("}\n")

    return best_post_values


def print_results(priors):
    print_results_list = []
    for prior in priors:
        prior_learnables = prior.learnable_params
        if len(prior_learnables) > 0:
            prior_name = prior.get_string()
            print_results_list.append(f"std::cout << \"mu_{prior_name}: \" << (float) best_mu_{prior_name} << std::endl;\n")
            if len(prior_learnables) > 1:
                print_results_list.append(f"std::cout << \"sigma_{prior_name}: \" << (float) best_sigma_{prior_name} << std::endl;\n")

    return print_results_list

def gen_code(model, file_name= "trial", type = "float"):
    print("Filename: ", file_name)
    assert(isinstance(model, Model))

    # Add header
    #include "../../include/distributions_latest.hpp"
    #include "../../include/ios.hpp"
    #include <iostream>
    #include <fstream>
    #include <random>

    
    # print priors
    priors = model.priors
    prior_names = [prior.get_string() for prior in priors]
    running_type = None
    I, K, F = None, None, None
    if type == "float":
        running_type = "float"
    elif type == "fixed":
        I,K,F = model.get_bit_configuration(0.9)
        running_type = f"fpm::fixed<std::int32_t, std::int64_t, {F}>"
        print("Running type: ", running_type)

    # const int BITSHIFT = K
    if type == "fixed":
        bitshift_string = f'const int NET_SCALING = std::pow(2,{K});\n'
    
    # print input data
    observed_data = model.observed.data.data
    observed_data_object = model.get_observed()
    observed_data_list_str = str(observed_data).replace('[', '{').replace(']', '}').replace(' ', '')

    input_data_object = model.get_data()
    input_data = model.input_data.data
    input_data_container = f'Constant<{running_type}> {input_data_object.get_string()}_container(0);\n'
    assert(len(input_data) == len(observed_data))
    input_data = str(input_data).replace('[', '{').replace(']', '}').replace(' ', '')
    

    max_elb_assign_str = f"{running_type} max_elbo = std::numeric_limits<{running_type}>::lowest();\n"

    best_values_list_assign = get_best_assign(priors, running_type)

    


    
    
    
    # set data
    observed_data_string = f'std::array<{running_type}, num_data_samples> {model.observed.get_array_name()} = {observed_data_list_str};\n'
    # set input data
    input_data_string = f'std::array<{running_type}, num_data_samples> {input_data_object.get_string()}_array = {input_data};\n'
    num_data_samples = f'const int num_data_samples = {len(observed_data)};\n'
    # add data to output
    data_output = None
    if type == "fixed":
        data_output = [observed_data_string, input_data_string, input_data_container, bitshift_string, max_elb_assign_str] + best_values_list_assign
    else:
        data_output = [observed_data_string, input_data_string, input_data_container, max_elb_assign_str] + best_values_list_assign
    # print(data_output)

    # set num_data_samples 
    # num_data_samples = f'const int num_data_samples = {model.num_data};\n'
    #set num_experiments
    num_experiments = f'const int num_experiments = {model.num_experiments};\n'
    #set num_samples
    num_samples = f'const int num_samples = {model.num_samples};\n'

    learning_rate_str = f'const {running_type} learning_rate = ({running_type}) 0.01;\n'

    num_iterations = f'const int num_iterations = {model.iterations};\n'

    lb_ub_str = f'{running_type} UB = ({running_type}) 128 ;\n{running_type} LB = ({running_type}) -128 ;\n'

    hyper_params = [ num_experiments, num_samples, learning_rate_str, lb_ub_str, num_iterations]

    main_body_content = [num_data_samples] + data_output + hyper_params

       # print(prior.posterior.complete_definition(running_type = running_type))
    # printing priors
    for prior in priors:
        prior_name = prior.get_string()
        prior_string = []
        print("Prior name: ", prior_name)
        # print("Type: ", type(prior))
        if isinstance(prior, Normal):
            prior_mean = prior.mu
            prior_sigma = prior.sigma
            prior_mean_name = prior_mean.get_string()
            prior_sigma_name = prior_sigma.get_string()
            print(f"Prior mean: {prior_mean_name}")
            print(f"Prior sigma: {prior_sigma_name}")
        
        if isinstance(prior, Uniform):
            prior_min = prior.lb
            prior_max = prior.ub
            prior_string = [f"Constant<{running_type}> {prior_name}_min ({prior_min});\n", f"Constant<{running_type}> {prior_name}_max ({prior_max});\n"]
            prior_sample = [f"Constant<{running_type}> {prior_name}_sample (0);\n"]
            prior_definition = [f"auto {prior_name} = uniform_dist<{running_type}>({prior_name}_min, {prior_name}_max, {prior_name}_sample);\n"]
            
            main_body_content += prior_string + prior_sample + prior_definition
            
    for prior in priors:
        print("Prior name: ", prior.get_string())
        main_body_content += prior.posterior.complete_definition(running_type = running_type)
          
    # printing observed variable

    observed_variable = model.observed.dist
    observed_variable_name = observed_variable.get_string()
    
    print("Observed variable name: ", observed_variable_name)
    observed_mean = observed_variable.mu
    observed_mean_name = None

    observed_variable_final = model.observed.dist

    if isinstance(observed_variable_final, Normal):

        observed_variable_sigma_value = observed_variable.sigma
        print("Sigma assign: ", observed_variable_sigma_value.get_string())

        main_body_content += [observed_variable_sigma_value.get_assignment(running_type)]

        if isinstance(observed_mean, Node):
            observed_mean_name = observed_mean.get_string()
        
            if input_data_object.get_string() in observed_mean_name:
            # repalce observed variable name with observed data name + _container
                observed_mean_name = observed_mean_name.replace(input_data_object.get_string(), input_data_object.get_string() + "_container")
        
            
            print("Observed mean name: ", observed_mean_name)
            # print("Object data: ", input_data_object.get_string())
            observed_mean_string = [f"auto mean_{observed_variable_name} = {observed_mean_name};\n"]
            main_body_content += observed_mean_string
            main_body_content += [model.observed.data.get_constant_container(running_type)]
            obs_assignment = model.observed.get_assignment(running_type)
            main_body_content += [obs_assignment]

    second_term_assign_str = f"{running_type} second_term = ({running_type}) 0;\n"
    main_body_content += [second_term_assign_str]
    for prior in priors:
        prior_name = prior.get_string()
        learnable_params = prior.get_learnable_params()

        # curr_sample of prior

        
        curr_sample_str = f"{running_type} {prior_name}_sample_value = ({running_type}) 0;\n"
        
        main_body_content += [curr_sample_str]

        if len(learnable_params) > 0:
            curr_mean = learnable_params[0]
            if isinstance(curr_mean, Variable):
                # define del_elbo of the prior
                del_elbo_str = f"{running_type} del_elbo_mu_{prior_name} = ({running_type}) 0;\n"
                step_str = f"{running_type} step_mu_{prior_name} = ({running_type}) 0;\n"
                new_value_str = f"{running_type} new_mu_{prior_name} = ({running_type}) 0;\n"
                main_body_content += [del_elbo_str]
                main_body_content += [step_str]
                main_body_content += [new_value_str]
            if len(learnable_params) > 1:
                curr_sigma = learnable_params[1]
                if isinstance(curr_sigma, Variable):
                    del_elbo_str = f"{running_type} del_elbo_sigma_{prior_name} = ({running_type}) 0;\n"
                    step_str = f"{running_type} step_sigma_{prior_name} = ({running_type}) 0;\n"
                    new_value_str = f"{running_type} new_sigma_{prior_name} = ({running_type}) 0;\n"
                    main_body_content += [del_elbo_str]
                    main_body_content += [step_str]
                    main_body_content += [new_value_str]
                    # print("Del elbo string: ", del_elbo_str)
                # print("Del elbo string: ", del_elbo_str)
        # print("Learnable params: ", learnable_params)
    # Some standard variables

    # current_score
    current_score = f"{running_type} current_score = ({running_type}) 0;\n"
    main_body_content += [current_score]
    # total_elbo
    total_elbo = f"{running_type} total_elbo = ({running_type}) 0;\n"
    main_body_content += [total_elbo]
    # zero 
    zero = f"{running_type} zero = ({running_type}) 0;\n"
    main_body_content += [zero]

    

    # creating main loop body
    # current_del_elbo_*
    iteration_for_loop = []
    reinit_to_zero_curr_del_elbos = []
    for prior in priors:
        prior_name = prior.get_string()
        learnable_params = prior.get_learnable_params()

        if len(learnable_params) > 0:
            curr_mean = learnable_params[0]
            if isinstance(curr_mean, Variable):
                reinit_to_zero_curr_del_elbos += [f"del_elbo_mu_{prior_name} = zero;\n"]
            if len(learnable_params) > 1:
                curr_sigma = learnable_params[1]
                if isinstance(curr_sigma, Variable):
                    reinit_to_zero_curr_del_elbos += [f"del_elbo_sigma_{prior_name} = zero;\n"]


    reinit_to_zero_curr_del_elbos += ["total_elbo = zero;\n"]
    # reinit_to_zero_curr_del_elbos += [f"current_score = ({running_type}) 0;\n"]

    # scoring total data in for loop
    data_score_for_loop_body = []
    for_counter = "data_k"
    data_score_upper_limit = model.observed.data.get_size()

    print("Data score upper limit: ", data_score_upper_limit)

    data_score_for_loop_body += [f"{model.get_data().get_container_name()}({model.get_data().get_array_name()}[{for_counter}]);\n"]
    data_score_for_loop_body += [f"{model.observed.data.string}({model.observed.get_array_name()}[{for_counter}]);\n"]

    if type == "fixed":
        data_score_for_loop_body += [f"current_score += {model.observed.get_string()}.log_pdf().value()/NET_SCALING;\n"]
    else:
        data_score_for_loop_body += [f"current_score += {model.observed.get_string()}.log_pdf().value();\n"]

    forloop = create_for_loop(data_score_for_loop_body, end = data_score_upper_limit, name_counter = for_counter )

    # print seccond term
    



    # main_body_content += forloop
    # print("For loop: ", forloop)



    # print(data_score_for_loop_body)






    # for each sample
    forloop_body = [] +[f"current_score = ({running_type}) 0;\n"]
    for prior in priors:

        curr_while_body = []
        # build while loop body
        sample_value_string = prior.sample_value_name
        # curr_while_body += [sample_value_string]

        assign_string = gen_assignment(sample_value_string, f"{prior.posterior.get_string()}.sample()")
        curr_while_body += [assign_string]
        # if 
        if_state = f"if ({sample_value_string} >= {prior.get_string()}.lb_value() && {sample_value_string} <= {prior.get_string()}.ub_value())\n"
        if_state += "{\n"
        if_state += f"{prior.sample_container_name}({sample_value_string});\n"
        if_state += "break;\n" 
        if_state += "}\n"
        
        forloop_body += gen_while_true(curr_while_body+[if_state])
        # main_body_content += curr_while_body 

        # print("Assign string: ", assign_string)


    # create num_samples for loop

    second_term_str = "second_term = current_score + ("
    for prior in priors:
        second_term_str += f"{prior.get_string()}.log_pdf().value() - {prior.posterior.string}.log_pdf().value() "
        if prior != priors[-1]:
            second_term_str += "+ "
    if running_type == "fixed":
        second_term_str += ")/NET_SCALING;\n"
    else:
        second_term_str += ");\n"

    
    update_curr_del_elbos = []
    for prior in priors:
        prior_name = prior.get_string()
        learnable_params = prior.get_learnable_params()
        if len(learnable_params) > 0:
            curr_mean = learnable_params[0]
            if isinstance(curr_mean, Variable):
                update_curr_del_elbos += [f"del_elbo_mu_{prior_name} += {prior.posterior.get_string()}.diff_log({prior.posterior.mu.name}) *(second_term);\n"]
            if len(learnable_params) > 1:
                curr_sigma = learnable_params[1]
                if isinstance(curr_sigma, Variable):
                    update_curr_del_elbos += [f"del_elbo_sigma_{prior_name} += {prior.posterior.get_string()}.diff_log({prior.posterior.sigma.name}) *(second_term);\n"]


    forloop_num_samples = create_for_loop(forloop_body + forloop + [second_term_str] + [f"total_elbo += second_term;\n"] + update_curr_del_elbos, end = "num_samples", name_counter = "sample_k")
    reinit_plus_forloop_num_samples = reinit_to_zero_curr_del_elbos + forloop_num_samples
    # print("For loop body: ", forloop)
    # loop_body = create_for_loop(reinit_to_zero_curr_del_elbos + forloop_num_samples + forloop)
    
    # loop_body += ["total_elbo = zero;\n"]
    # print("Loop body: ", loop_body)
    iteration_for_loop += reinit_plus_forloop_num_samples

    # second term of elbo
    

    # print("Second term: ", second_term_str)
    # iteration_for_loop += [second_term_str]

    # iteration_for_loop += [f"total_elbo += second_term;\n"]


    
    
    # print(update_curr_del_elbos)
    # iteration_for_loop += update_curr_del_elbos

    iteration_for_loop += update_best_post_values(priors)


    step_update_lists = write_step_update(priors=priors, running_type= running_type)
    iteration_for_loop += step_update_lists

    iteration_for_loop += new_values(priors)
    print(step_update_lists)

    final_update_lists = final_update_containers(priors, running_type)

    iteration_for_loop += final_update_lists

    iteration_loop_body = create_for_loop(iteration_for_loop, end = "num_iterations", name_counter = "iteration_k")

    main_body_content += iteration_loop_body
    # print("Normal detected")
    

    # generate algorithm now
    # const_int_num_samples = f"const int num_samples = {model.num_samples};\n"
    main_body_content += print_results(priors)

    main_output = main_body(main_body_content)

    #print to file
    with open(f"./{file_name}.cpp", "w") as f:
        print("writing to file")
        f.writelines(main_output)
    


    # print(main_output)
    # test_print(main_output)
    # print(test_output)
    # for prior in priors:
        
        # print(prior.get_string())

    


