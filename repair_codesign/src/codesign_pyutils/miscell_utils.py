
import numpy as np

import os

from codesign_pyutils.misc_definitions import get_design_map, get_coll_joint_map

def str2bool(v: str):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def wait_for_confirmation(do_something = "proceed",\
                          or_do_something_else = "stop",\
                          on_confirmation = "Confirmation received!",\
                          on_denial = "Stop received!"):

  usr_input = input("\n \nPress Enter to " + do_something + \
                    " or type \"N/n\" to " + or_do_something_else + ". \n -> ")

  if usr_input == "":

      print("\n" + on_confirmation + "\n")
      go_on = True
  
  else:

      if (usr_input == "no" or usr_input == "No" or usr_input == "N" or usr_input == "n"):
        
        print("\n" +  on_denial + "\n")
        go_on = False
        
      else: # user typed something else

        print("\nOps.. you did not type any of the options! \n")

        go_on = wait_for_confirmation()

  return go_on

def check_str_list(comp_list = ["x", "y", "z"], input = []):
  
  presence_array = [False] * len(comp_list)

  for i in range(len(comp_list)):

    for j in range(len(input)):

      if (input[j] == comp_list[i] or input[j] == comp_list[i].upper()) and presence_array[i] != True:

        presence_array[i] = True

  return np.where(presence_array)[0]
  
def rot_error_axis_sel_not_supp(axis_selector: np.ndarray, rot_error_approach: str):

  if len(axis_selector) != 3:
  
    raise Exception("\nSelecting the constrained axis when using \"" + rot_error_approach + "\" orientation error is not supported yet.\n")

def get_min_cost_index(costs: list, solve_failed_array: list):

    # negative numbers are assignes to not solved probelms

    best_index = - 1
    n_glb_tests = len(costs)

    if not solve_failed_array[0]:

        best_index = 0

    for i in range(1, n_glb_tests):

        if (not solve_failed_array[i]) and (best_index != -1): # if solution is valid and there exist a previous valid solution

          if (costs[i] - costs[best_index]) < 0:
            
            best_index = i
        
        if (not solve_failed_array[i]) and (best_index == -1): # no previous valid sol exists, but current solution is valid

          best_index = i        

    if best_index < 0:

        best_index = np.argmin(costs)

    return best_index

def get_full_abs_paths_sol(init_opt_results_path: str, init_failed_results_path: str):

  file_list_opt = os.listdir(init_opt_results_path)
  file_list_failed = os.listdir(init_failed_results_path)
  is_file_opt = [True] * len(file_list_opt) + [False] * len(file_list_failed)
  full_file_list = file_list_opt + file_list_failed
  full_file_paths = [""] * len(full_file_list)

  for i in range(len(full_file_list)):

    if is_file_opt[i]:

        full_file_paths[i] = init_opt_results_path + "/" + full_file_list[i]
    
    else:

        full_file_paths[i] = init_failed_results_path + "/" + full_file_list[i]

  return full_file_paths

def extract_q_design(input_data: list):

    design_var_map = get_design_map()

    design_indeces = [design_var_map["mount_h"],\
        design_var_map["should_w_l"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"]]

    # design_indeces = [design_var_map["mount_h"],\
    #     design_var_map["should_w_l"],\
    #     design_var_map["should_roll_l"]]

    n_samples = len(input_data)

    design_data = np.zeros((len(design_indeces), n_samples))

    for i in range(n_samples):

        design_data[:, i] = input_data[i][design_indeces, 0] # design variables are constant over the nodes (index 0 is sufficient)

    return design_data

def extract_q_joint(input_data: list):

  design_var_map = get_design_map()

  coll_aux_jnt_map = get_coll_joint_map()

  n_tot_vars = 24
  full_indxs_array = np.linspace(0, n_tot_vars - 1, n_tot_vars, dtype=int)
  indxs_to_be_removed = [design_var_map["mount_h"],\
  design_var_map["should_w_l"],\
  design_var_map["should_roll_l"],\
  design_var_map["wrist_off_l"], \
  design_var_map["should_w_r"],\
  design_var_map["should_roll_r"],\
  design_var_map["wrist_off_r"], \
  coll_aux_jnt_map["link5_coll_joint_l"], \
  coll_aux_jnt_map["link5_coll_joint_r"]]

  joint_q_indxs = np.delete(full_indxs_array, indxs_to_be_removed)

  n_samples = len(input_data)
  
  joint_q_dim = n_tot_vars - n_samples

  joint_data = [None] * n_samples
  
  for i in range(n_samples):

      joint_data[i] = input_data[i][joint_q_indxs, :]

  return joint_data

def gen_y_sampling(n_y_samples: int, y_sampl_ub: np.double):

    y_sampl_lb = - y_sampl_ub

    if n_y_samples == 1:

        dy = 0.0

    else:

        dy = (y_sampl_ub - y_sampl_lb) / (n_y_samples - 1)

    y_sampling = np.array( [0.0] * n_y_samples)
    
    for i in range(n_y_samples):
        
        y_sampling[i] = y_sampl_lb + dy * i

    return y_sampling

def compute_solution_divs(n_multistrt: int, n_prcss: int):
    
    n_sol_tries = n_multistrt
    n_p = n_prcss

    n_divs = int(np.floor(n_sol_tries / n_p)) 

    n_remaining_sols = n_sol_tries - n_divs * n_p

    opt_divs = []

    if n_multistrt <= n_prcss: #assign a multistart to each process

      for i in range(n_multistrt):

        opt_divs.append(list(range(n_divs * i, n_divs * i + n_divs)))
        
    else: # distribute them appropiately

      for i in range(n_p):

          opt_divs.append(list(range(n_divs * i, n_divs * i + n_divs)))

    if n_remaining_sols != 0:
        
        for i in range(n_remaining_sols):

            opt_divs[i].append(n_divs * (n_p - 1) + n_divs + i)

        # opt_divs = [[]] * (n_p + 1)

        # for i in range(n_p + 1):
            
        #     if i == n_p:

        #         opt_divs[i] = list(range(n_divs * i, n_divs * i + n_remaining_sols))

        #     else:

        #         opt_divs[i] = list(range(n_divs * i, n_divs * i + n_divs))

    return opt_divs

def select_best_sols(perc: float, opt_costs: list, opt_q_design: np.ndarray):

  perc = abs(float(perc))
  if perc > 100.0:
      perc = 100.0

  n_opt_sol = len(opt_costs)
  n_selection = round(n_opt_sol * perc/100)

  sorted_costs_indeces = np.argsort(np.array(opt_costs)) # indeces from best cost(lowest) to worst (biggest)
  selection_indeces = sorted_costs_indeces[:n_selection] # first n_selection indeces

  opt_q_design_selections = opt_q_design[:, selection_indeces]
  opt_costs_sorted = [opt_costs[i] for i in selection_indeces]

  return opt_q_design_selections, opt_costs_sorted

def correct_list(input_node_list: np.ndarray):

    input_node_list = input_node_list.tolist()

    output_node_list = []

    # applying ugly corrections
    if type(input_node_list[0][0]) == np.ndarray: # probably here we have read a multitask solution
        # which will have different number of nodes.
        reduced_list = input_node_list[0]
        for i in range(len(reduced_list)):

            output_node_list.append(reduced_list[i].tolist()[0])

    else:

        output_node_list = input_node_list

    return output_node_list  

    