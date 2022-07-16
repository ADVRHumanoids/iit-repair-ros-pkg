
import numpy as np

import time 

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def wait_for_confirmation(do_something = "proceed",\
                          or_do_something_else = "stop",\
                          on_confirmation = "Confirmation received!",\
                          on_denial = "Stop received!"):

  usr_input = input("\n \n Press Enter to " + do_something + \
                    " or type \"N/n\" to " + or_do_something_else + ". \n -> ")

  if usr_input == "":

      print("\n" + on_confirmation + "\n")
      go_on = True
  
  else:

      if (usr_input == "no" or usr_input == "No" or usr_input == "N" or usr_input == "n"):
        
        print("\n" +  on_denial + "\n")
        go_on = False
        
      else: # user typed something else

        print("\n Ops.. you did not type any of the options! \n")

        go_on = wait_for_confirmation()

  return go_on

def check_str_list(comp_list = ["x", "y", "z"], input = []):
  
  presence_array = [False] * len(comp_list)

  for i in range(len(comp_list)):

    for j in range(len(input)):

      if (input[j] == comp_list[i] or input[j] == comp_list[i].upper()) and presence_array[i] != True:

        presence_array[i] = True

  return np.where(presence_array)[0]
  
def rot_error_axis_sel_not_supp(axis_selector, rot_error_approach):

  if len(axis_selector) != 3:
  
    raise Exception("\nSelecting the constrained axis when using \"" + rot_error_approach + "\" orientation error is not supported yet.\n")

def get_min_cost_index(costs):

    # to not solved probelms negative numbers are assignes

    best_index = - 1
    n_glb_tests = len(costs)

    if n_glb_tests == 1:

        best_index = 0

    for i in range(n_glb_tests):

        if i != 0 and costs[i] >= 0 and (costs[i] - costs[i - 1]) < 0:
            
            best_index = i

    if best_index < 0:

        raise Exception("Not able to solve any of the problems!! I will not be able to select the best cost.")

    return best_index



