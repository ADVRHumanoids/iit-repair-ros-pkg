
from horizon.solvers import Solver

import numpy as np

import time 

from codesign_pyutils.tasks import TaskGen

from codesign_pyutils.miscell_utils import wait_for_confirmation

import argparse

from termcolor import colored

def solve_prb_standalone(task: TaskGen,\
                        slvr: Solver,\
                        q_init=None, q_dot_init=None,\
                        prbl_name = "Problem",
                        on_failure = "\n Failed to solve problem!! \n", 
                        on_success = "\n Converged to an optimal solution!! \n",
                        is_second_step_opt = False, 
                        q_codes_s1 = None):

    # standard routine for solving the problem
        
    task.set_ig(q_init, q_dot_init)

    if not is_second_step_opt:  # first level optimization 

        task.wrist_off_ref.assign(task.sliding_wrist_offset) # has no effect if the task was build with is_sliding_wrist set to False
    
    else: # second level optimization
        
        if q_codes_s1 is None:

            raise Exception("solve_prb_standalone: if running the second level optimization you must provide \
                            an initialization for the codesign vars.")

        task.q_codes_ref.assign(q_codes_s1) # assign design variables from first level optimization

    t = time.time()

    is_optimal = slvr.solve()  # solving

    solution_time = time.time() - t
    
    if not is_optimal:

        print(colored(on_failure, 'red'))
        
        print(colored(f'\nTime elapsed: {solution_time} s \n', "red"))

    else:

        print(colored(on_success, 'green'))

        print(colored(f'\nTime elapsed: {solution_time} s \n', "green"))

    solve_failed = not is_optimal
    
    return solve_failed, solution_time

def solve_main_prb_soft_init(task: TaskGen, slvr_init: Solver, slvr: Solver,\
                    q_ig_main=None, q_dot_ig_main=None,\
                    prbl_name = "Problem",\
                    on_failure = "\n Failed to converge using soft initialization!! \n'"):

    # Routine for solving the "hard" problem employing the solution 
    # from another solver.
    # If solution fails, the user can choose to try to solve the "hard"
    # problem without any initialization from the other solution

    task.set_ig(slvr_init.getSolutionDict()["q"], slvr_init.getSolutionDict()["q_dot"])

    solve_failed = False

    t = time.time()

    is_optimal = slvr.solve()  # solving soft problem

    solution_time = time.time() - t 

    print(f'\n {prbl_name} converged to solution in {solution_time} s \n')
    
    solve_failed = not is_optimal

    if solve_failed:
        
        print(on_failure)

        proceed_to_hard_anyway = wait_for_confirmation(\
                    do_something = "try to solve the main problem without soft initialization",\
                    or_do_something_else = "stop here", \
                    on_confirmation = "Trying to solve the main problem  without soft initialization...",\
                    on_denial = "Stopping here!")
        
        if proceed_to_hard_anyway:
            
            solve_failed, _ = solve_prb_standalone(task, slvr, q_ig_main, q_dot_ig_main)


    return solve_failed, solution_time

def try_init_solve_or_go_on(arguments: argparse.Namespace,\
                            init_task: TaskGen, init_slvr: Solver,\
                            task: TaskGen, slvr: Solver,\
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

        t = time.time()

        init_is_optimal = init_slvr.solve()  # solving soft problem

        solution_time = time.time() - t 

        print(f'\n Soft initialization problem solved in {solution_time} s \n')

        soft_sol_failed = init_is_optimal
        
        if not init_is_optimal:# failed to solve the soft problem 

            print('\n Failed to converge (soft initialization problem) \n')

            proceed_to_hard_anyway = wait_for_confirmation(\
                                     do_something = "try to solve the main problem anyway",\
                                     or_do_something_else = "stop here", \
                                     on_confirmation = "Trying to solve the main problem  ...",\
                                     on_denial = "Stopping here!")

            if proceed_to_hard_anyway:
                
                sol_failed = solve_prb_standalone(task, slvr, q_ig_main, q_dot_ig_main)
            

            
        return soft_sol_failed, sol_failed

def do_one_solve_pass(arguments: argparse.Namespace,\
                    task: TaskGen, slvr: Solver,\
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
                abs_paths: str, task: TaskGen,\
                n_sol_tries: int, seed: int,\
                verbose = False):

    q_ig = [None] * n_sol_tries
    q_dot_ig = [None] * n_sol_tries

    # solution_index_name = "solution_index"

    np.random.seed(seed)
    
    for i in range(n_sol_tries):

        # if arguments.load_initial_guess:
            
        #     # check compatibility between load paths and number of solution attempts
        #     if len(abs_paths) != n_sol_tries:

        #         raise Exception("Solution load mismatch: You set " + str(n_sol_tries) + \
        #                         " solution attempts, but provided " +\
        #                         str(len(abs_paths)) + " solutions to load." )
                            
        #     try:
            
        #         ms_ig_load = mat_storer.matStorer(abs_paths[i])
                
        #         ig_sol = ms_ig_load.load()

        #         solution_index = ig_sol[solution_index_name][0][0]

        #         q_ig[solution_index] = ig_sol["q"]
        #         q_dot_ig[solution_index] = ig_sol["q_dot"]

        #         if (np.shape(q_ig[solution_index])[1] != task.total_nnodes) or \
        #         (np.shape(q_ig[solution_index])[0] != task.nq):

        #             raise Exception("\nThe loaded initial guess has shape: [" + \
        #                             str(np.shape(q_ig[i])[0]) +  ", " + str(np.shape(q_ig[i])[1]) +  "]" + \
        #                             " while the problem has shape: [" + \
        #                             str(task.nq) +  ", " + str(task.total_nnodes) + \
        #                             "].\n")

        #     except:
                
        #         raise Exception("Failed to load initial guess from file! I will use random intialization.")

        # else:
            
            
        q_ig[i] = np.tile(np.random.uniform(task.lbs, task.ubs,\
                                    (1, task.nq)).T, (1, task.total_nnodes))

        q_dot_ig[i] = np.zeros((task.nv, task.total_nnodes - 1))

            
        if verbose:
            
            print("q initial guess:", q_ig)
            print("q_dot initial guess:", q_dot_ig)
            
                                            
    return q_ig, q_dot_ig
