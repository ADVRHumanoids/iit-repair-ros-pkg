from asyncio import tasks
from horizon.utils import mat_storer
from horizon.solvers import Solver

from datetime import datetime
from datetime import date

import numpy as np

import time 

from codesign_pyutils.tasks import FlippingTaskGen

from codesign_pyutils.miscell_utils import wait_for_confirmation

import warnings

import argparse

def solve_prb_standalone(task,\
                        slvr: Solver,\
                        q_init=None, q_dot_init=None,\
                        prbl_name = "Problem",
                        on_failure = "\n Failed to solve problem!! \n'"):

    # standard routine for solving the problem

    solve_failed = False

    try:
        
        task.set_ig(q_init, q_dot_init)

        t = time.time()

        slvr.solve()  # solving

        solution_time = time.time() - t

        print(f'\n {prbl_name} solved in {solution_time} s \n')
        
        solve_failed = False

    except:
        
        print(on_failure)

        solve_failed = True

    
    return solve_failed

def solve_main_prb_soft_init(task: FlippingTaskGen, slvr_init: Solver, slvr: Solver,\
                    q_ig_main=None, q_dot_ig_main=None,\
                    prbl_name = "Problem",\
                    on_failure = "\n Failed to solve problem with soft initialization!! \n'"):

    # Routine for solving the "hard" problem employing the solution 
    # from another solver.
    # If solution fails, the user can choose to try to solve the "hard"
    # problem without any initialization from the other solution

    task.set_ig(slvr_init.getSolutionDict()["q"], slvr_init.getSolutionDict()["q_dot"])

    solve_failed = False

    try: # try to solve the main problem with the results of the soft initialization
    
        t = time.time()

        slvr.solve()  # solving soft problem

        solution_time = time.time() - t 

        print(f'\n {prbl_name} solved in {solution_time} s \n')

        solve_failed = False
    
    except:
        
        print(on_failure)

        proceed_to_hard_anyway = wait_for_confirmation(\
                    do_something = "try to solve the main problem without soft initialization",\
                    or_do_something_else = "stop here", \
                    on_confirmation = "Trying to solve the main problem  without soft initialization...",\
                    on_denial = "Stopping here!")
        
        if proceed_to_hard_anyway:
            
            solve_failed = solve_prb_standalone(task, slvr, q_ig_main, q_dot_ig_main)

        else:

            solve_failed = True

    return solve_failed

def try_init_solve_or_go_on(arguments: argparse.Namespace,\
                            init_task: FlippingTaskGen, init_slvr: Solver,\
                            task: FlippingTaskGen, slvr: Solver,\
                            q_ig_main=None, q_dot_ig_main=None, \
                            q_ig_init=None, q_dot_ig_init=None):
        
        # Routine for solving the initialization problem.
        # If solution fails, the user can choose to solve
        # the main problem anyway, without employing the 
        # solution the initialization problem.

        soft_sol_failed =  False
        sol_failed = False

        if arguments.use_init_guess:

            init_task.set_ig(q_ig_init, q_dot_ig_init)

        try: # try solving the soft initialization problem
            
            t = time.time()

            init_slvr.solve()  # solving soft problem

            solution_time = time.time() - t 

            print(f'\n Soft initialization problem solved in {solution_time} s \n')

            soft_sol_failed = False
        
        except: # failed to solve the soft problem 

            soft_sol_failed = True

            print('\n Failed to solve the soft initialization problem!! \n')

            proceed_to_hard_anyway = wait_for_confirmation(\
                                     do_something = "try to solve the main problem anyway",\
                                     or_do_something_else = "stop here", \
                                     on_confirmation = "Trying to solve the main problem  ...",\
                                     on_denial = "Stopping here!")

            if proceed_to_hard_anyway:
                
                sol_failed = solve_prb_standalone(task, slvr, q_ig_main, q_dot_ig_main)
            
            else:

                sol_failed = True
            
        return soft_sol_failed, sol_failed

# def build_multiple_flipping_tasks(arguments: argparse.Namespace,\
#                     task: FlippingTaskGen, n_filling_nodes, y_sampling, right_arm_picks,\
#                     urdf_full_path,\
#                     ig_abs_path, \
                    
#                     is_soft_pose_cnstr = False,\
#                     t_exec = 6.0, epsi = 0.001, verbose = False, \
#                     object_length = 0.05):

#     # Routine for automatically adding multiple flipping tasks,
#     # based on the provided y-axis sampling. 

#     # initialize main problem task
#     task = FlippingTaskGen(cocktail_size = object_length, filling_n_nodes = n_filling_nodes)

#     # add tasks to the task holder object
#     next_node = 0 # used to place the next task on the right problem nodes
#     for i in range(len(y_sampling)):

#         next_node = task.add_in_place_flip_task(init_node = next_node,\
#                                     object_pos_wrt_ws = np.array([0.0, y_sampling[i], 0.0]), \
#                                     object_q_wrt_ws = np.array([0, 1, 0, 0]), \
#                                     #  pick_q_wrt_ws = np.array([np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0]),\
#                                     right_arm_picks = right_arm_picks)

#     if verbose:
        
#         print("Flipping task node list: ", task.nodes_list)
#         print("Total employed nodes: ", task.total_nnodes)
#         print("Number of added subtasks:", task.n_of_tasks, "\n")

#     # initialize problem
#     task.init_prb(urdf_full_path, arguments.base_weight_pos, arguments.base_weight_rot,\
#                 arguments.weight_global_manip,\
#                 is_soft_pose_cnstr = arguments.soft_pose_cnstrnt,\
#                 tf_single_task = t_exec)

#     # generate an initial guess, based one the script arguments
#     q_ig_main, q_dot_ig_main =  generate_ig(arguments, ig_abs_path,\
#                             task,\
#                             n_glob_tests, ig_seed,\
#                             True)
#     # set constraints and costs
#     task.setup_prb(rot_error_epsi, q_ig_main, q_dot_ig_main)

#     # build the main problem
#     build_multiple_flipping_tasks(args, task, y_sampling,\
#                                 right_arm_picks, urdf_full_path,\
#                                 epsi = rot_error_epsi)

#     # build the problem 
#     task.init_prb(urdf_full_path, weight_glob_man = arguments.weight_global_manip,\
#                             is_soft_pose_cnstr = is_soft_pose_cnstr, epsi = epsi)

#     return task

def do_one_solve_pass(arguments: argparse.Namespace,\
                    task: FlippingTaskGen, slvr: Solver,\
                    q_ig_main=None, q_dot_ig_main=None,\
                    task_init=None, slvr_init=None,\
                    q_ig_init=None, q_dot_ig_init=None):

    init_sol_failed = False
    solve_failed = False

    if not arguments.warmstart:

        solve_failed = solve_prb_standalone(task, slvr, q_ig_main, q_dot_ig_main)

    else:

        init_sol_failed, solve_failed = try_init_solve_or_go_on(arguments,\
                                        task_init, slvr_init,\
                                        task, slvr,\
                                        q_ig_main = q_ig_main, q_dot_ig_main = q_dot_ig_main, \
                                        q_ig_init = q_ig_init, q_dot_ig_init = q_dot_ig_init)

        if not init_sol_failed: # initialization solution succedeed --> try to solve the main problem

            solve_failed = solve_main_prb_soft_init(task,\
                                                    slvr_init, slvr, \
                                                    q_ig_main, q_dot_ig_main)

    return init_sol_failed, solve_failed

def generate_ig(arguments: argparse.Namespace,\
                abs_paths, task,\
                n_sol_tries, seed,\
                verbose = False):

    q_ig = [None] * n_sol_tries
    q_dot_ig = [None] * n_sol_tries


    if arguments.use_init_guess:

        np.random.seed(seed)
        
        for i in range(n_sol_tries):

            if arguments.load_initial_guess:
                
                # check compatibility between load paths and number of solution attempts
                if len(abs_paths) != n_sol_tries:

                    raise Exception("You set " + str(n_sol_tries) + \
                                    "solution attempts, but only provided " +\
                                    str(len(abs_paths)) + " solutions to load." )
                                
                try:
                                    
                    ms_ig_load = mat_storer.matStorer(abs_paths[i])

                    ig_sol = ms_ig_load.load()

                    q_ig[i] = ig_sol["q"]
                    q_dot_ig[i] = ig_sol["q_dot"]

                except:
                    
                    warnings.warn("Failed to load initial guess from file! I will use random intialization.")

                    q_ig[i] = np.tile(np.random.uniform(task.lbs, task.ubs,\
                                            (1, task.nq)).T, (1, task.total_nnodes))
                    
                    q_dot_ig[i] = np.zeros((task.nv, task.total_nnodes - 1))

                if (np.shape(q_ig[i])[1] != task.total_nnodes) or \
                    (np.shape(q_ig[i])[0] != task.nq):

                    raise Exception("\nThe loaded initial guess has shape: [" + \
                                    str(np.shape(q_ig[i])[0]) +  ", " + str(np.shape(q_ig[i])[1]) +  "]" + \
                                    " while the problem has shape: [" + \
                                    str(task.nq) +  ", " + str(task.total_nnodes) + \
                                    "].\n")

            else:
                
                
                q_ig[i] = np.tile(np.random.uniform(task.lbs, task.ubs,\
                                            (1, task.nq)).T, (1, task.total_nnodes))

                q_dot_ig[i] = np.zeros((task.nv, task.total_nnodes - 1))

                
            if verbose:
                
                print("q initial guess:", q_ig)
                print("q_dot initial guess:", q_dot_ig)
            
                                            
    return q_ig, q_dot_ig
    
