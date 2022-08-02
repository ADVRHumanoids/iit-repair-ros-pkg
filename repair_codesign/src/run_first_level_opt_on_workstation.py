#!/usr/bin/env python3

from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver
import os, argparse
from os.path import exists

import numpy as np

import subprocess

import rospkg

from codesign_pyutils.miscell_utils import str2bool,\
                                        get_min_cost_index
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.task_utils import solve_prb_standalone, \
                                        generate_ig              
from codesign_pyutils.tasks import TaskGen

import multiprocessing as mp_classic

from datetime import datetime
from datetime import date

def compute_solution_divs(n_multistrt: int, n_prcss: int):
    
    n_sol_tries = n_multistrt
    n_p = n_prcss

    n_divs = int(np.round(n_sol_tries / n_p)) 

    n_remaining_sols = n_sol_tries - n_divs * n_p

    opt_divs = [[]] * n_p


    for i in range(n_p):

        if i == (n_p - 1) and n_remaining_sols != 0:
            
            opt_divs[i] = list(range(n_divs * i, n_divs * i + n_divs + n_remaining_sols)) 

        else:

            opt_divs[i] = list(range(n_divs * i, n_divs * i + n_divs)) 


        # opt_divs = [[]] * (n_p + 1)

        # for i in range(n_p + 1):
            
        #     if i == n_p:

        #         opt_divs[i] = list(range(n_divs * i, n_divs * i + n_remaining_sols))

        #     else:

        #         opt_divs[i] = list(range(n_divs * i, n_divs * i + n_divs))

    return opt_divs

def solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array):

    sol_index = 0 # different index
    for node in multistart_nodes:

        print("\n SOLVING PROBLEM N.: ", node + 1)
        print("\n")
                    
        solve_failed = solve_prb_standalone(task, slvr, q_ig[node], q_dot_ig[node])
        solutions[sol_index] = slvr.getSolutionDict()

        print("Solution cost " + str(node) + ": ", solutions[sol_index]["opt_cost"])
        sol_costs[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()

        solve_failed_array[sol_index] = solve_failed

        sol_index = sol_index + 1
    
    return True

def gen_y_sampling(n_y_samples, y_sampl_ub):

    y_sampl_lb = - y_sampl_ub
    if n_y_samples == 1:
        dy = 0.0
    else:
        dy = (y_sampl_ub - y_sampl_lb) / (n_y_samples - 1)

    y_sampling = np.array( [0.0] * n_y_samples)
    for i in range(n_y_samples):
        
        y_sampling[i] = y_sampl_lb + dy * i

    return y_sampling

def gen_task_copies(filling_n_nodes, sliding_wrist_offset, 
                    n_y_samples, y_sampl_ub):

    
    y_sampling = gen_y_sampling(n_y_samples, y_sampl_ub)

    # initialize problem task
    task = TaskGen(filling_n_nodes = filling_n_nodes, \
                                    sliding_wrist_offset = sliding_wrist_offset)
    object_q = np.array([1, 0, 0, 0])

    # add tasks to the task holder object
    next_node = 0 # used to place the next task on the right problem nodes
    # in place flip task
    for i in range(len(y_sampling)):

        next_node = task.add_in_place_flip_task(init_node = next_node,\
                        object_pos_wrt_ws = np.array([0.0, y_sampling[i], 0.0]), \
                        object_q_wrt_ws = object_q, \
                        pick_q_wrt_ws = object_q,\
                        right_arm_picks = right_arm_picks)
    # # bimanual task
    # for j in range(len(y_sampling)):

    #     next_node = task.add_bimanual_task(init_node = next_node,\
    #                     object_pos_wrt_ws = np.array([0.0, y_sampling[j], 0.0]), \
    #                     object_q_wrt_ws = object_q, \
    #                     pick_q_wrt_ws = object_q,\
    #                     right_arm_picks = right_arm_picks)

    # initialize problem
    task.init_prb(urdf_full_path,
                    weight_glob_man = args.weight_global_manip, weight_class_man = args.weight_class_manip,\
                    tf_single_task = t_exec_task)

    print("Task node list: ", task.nodes_list)
    print("Task list: ", task.task_list)
    print("Task names: ", task.task_names)
    print("Task dict: ", task.task_dict)
    print("Total employed nodes: ", task.total_nnodes)
    print("Number of added subtasks:", task.n_of_tasks, "\n")

    # set constraints and costs
    task.setup_prb(rot_error_epsi, is_classical_man = args.use_classical_man)

    if solver_type != "ilqr":

        Transcriptor.make_method(transcription_method,\
                                task.prb,\
                                transcription_opts)
    
    ## Creating the solver
    slvr = solver.Solver.make_solver(solver_type, task.prb, slvr_opt)

    if solver_type == "ilqr":

        slvr.set_iteration_callback()


    return task, slvr

def sol_main(args, multistart_nodes, q_ig, q_dot_ig, task, slvr, result_path, opt_path, fail_path,\
        id_unique,\
        process_id):
    
    n_multistarts_main = len(multistart_nodes)

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_costs = [1e10] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main

    solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array)

    # solutions packaging for postprocessing
    
    sol_dumper = SolDumper()

    # n_opt_sol = len(np.where(np.array(solve_failed_array)== False)[0])

    # best_index = get_min_cost_index(sol_costs, solve_failed_array)

    # other_stuff = {"solve_failed": solve_failed_array, 
    #                 "n_opt_sol": n_opt_sol, "n_unfeas_sol": n_multistarts_main - n_opt_sol,
    #                 "sol_costs": sol_costs, "best_sol_index": best_index}
    
    # sol_dumper.add_storer(other_stuff, result_path,\
    #                         "additional_info_p" + str(process_id) + "_t" + id_unique,\
    #                         False)

    sol_index = 0
    for node in multistart_nodes:
            
        full_solution = {**(solutions[sol_index]),
                        **(cnstr_opt[sol_index]),
                        **{"q_ig": q_ig[sol_index], "q_dot_ig": q_dot_ig[sol_index]}, \
                        **{"solution_index": node}}

        if not solve_failed_array[sol_index]:

            sol_dumper.add_storer(full_solution, opt_path,\
                            solution_base_name + "_p" + str(process_id) + "_n" + str(node) + "_t" + id_unique, False)
        else:

            sol_dumper.add_storer(full_solution, fail_path,\
                            solution_base_name + "_p" + str(process_id) + "_n" + str(node) + "_t" + id_unique, False)

        sol_index = sol_index + 1

    sol_dumper.dump() 

    print("\n Solutions of process " + str(process_id) + " dumped. \n")
                    
    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')

    parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
                        help = 'whether to load ig from files', default = False)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 0.01)
    parser.add_argument('--use_classical_man', '-ucm', type=str2bool,\
                        help = 'whether to use the classical manipulability index', default = False)
    parser.add_argument('--n_multistarts', '-msn', type=int,\
                        help = 'number of multistarts to use', default = 4)
    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)                      
    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = False)

    args = parser.parse_args()
    
    # unique id used for generation of results
    unique_id = date.today().strftime("%d-%m-%Y") + "-" +\
                        datetime.now().strftime("%H_%M_%S")

    # number of parallel processes on which to run optimization
    # set to number of cpu counts to saturate
    processes_n = mp_classic.cpu_count()

    # useful paths
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
    urdf_name = "repair_full"
    urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
    xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
    codesign_path = rospackage.get_path("repair_codesign")
    results_path = codesign_path + "/test_results/test_results_" + unique_id
    opt_results_path = results_path + "/opt" 
    failed_results_path = results_path + "/failed"

    solution_base_name = "repair_codesign_opt"

    sliding_wrist_command = "is_sliding_wrist:=" + "true"
    show_softhand_command = "show_softhand:=" + "true"

    # generate update urdf every time the script runs
    try:

        # print(sliding_wrist_command)
        xacro_gen = subprocess.check_call(["xacro",\
                                        xacro_full_path, \
                                        sliding_wrist_command, \
                                        show_softhand_command, \
                                        "-o", 
                                        urdf_full_path])

    except:

        print('Failed to generate URDF.')

    # task-specific options
    right_arm_picks = True
    filling_n_nodes = 0
    rot_error_epsi = 0.0000001

    # samples
    n_y_samples = 5
    y_sampl_ub = 0.4

    # number of solution tries with different (random) initializations
    n_multistarts = args.n_multistarts

    # solver options
    solver_type = 'ipopt'
    if args.use_ma57:

        slvr_opt = {
            "ipopt.tol": 0.0000001, 
            "ipopt.max_iter": 1000,
            "ipopt.constr_viol_tol": 0.000001,
            "ilqr.verbose": True, 
            "ipopt.linear_solver": "ma57"}

    else:

        slvr_opt = {
            "ipopt.tol": 0.0000001, 
            "ipopt.max_iter": 1000,
            "ipopt.constr_viol_tol": 0.000001,
            "ilqr.verbose": True}


    full_file_paths = None # not used

    # seed used for random number generation
    ig_seed = args.ig_seed

    # single task execution time
    t_exec_task = 6

    # transcription options (if used)
    transcription_method = 'multiple_shooting'
    intgrtr = 'RK4'
    transcription_opts = dict(integrator = intgrtr)

    sliding_wrist_offset = 0.0

    proc_sol_divs = compute_solution_divs(n_multistarts, processes_n)

    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)
        os.makedirs(opt_results_path)
        os.makedirs(failed_results_path)

    task_copies = [None] * len(proc_sol_divs)
    slvr_copies = [None] * len(proc_sol_divs)
    
    for p in range(len(proc_sol_divs)):
        
        task_copies[p], slvr_copies[p] = gen_task_copies(filling_n_nodes, sliding_wrist_offset, 
                    n_y_samples, y_sampl_ub)

    # some initializations
    q_ig = [None] * n_multistarts
    q_dot_ig = [None] * n_multistarts

    # generating initial guesses, based on the script arguments
    q_ig, q_dot_ig =  generate_ig(args, full_file_paths,\
                                    task_copies[0],\
                                    n_multistarts, ig_seed,\
                                    False)
    
    # dumping info on the task 

    # inizialize a dumper object for post-processing

    task_info_dumper = SolDumper()

    other_stuff = {"dt": task_copies[0].dt, "filling_nodes": task_copies[0].filling_n_nodes,
                    "task_base_nnodes": task_copies[0].task_base_n_nodes_dict,
                    "right_arm_picks": task_copies[0].rght_arm_picks, 
                    "w_man_base": args.weight_global_manip, 
                    "w_clman_base": args.weight_global_manip,
                    "wman_actual": args.weight_class_manip, 
                    "nodes_list": task_copies[0].nodes_list, 
                    "tasks_list": task_copies[0].task_list,
                    "tasks_dict": task_copies[0].task_dict, 
                    "y_sampl_ub": y_sampl_ub, "n_y_samples": n_y_samples, 
                    "ig_seed": ig_seed, 
                    "solver_type": solver_type, "slvr_opts": slvr_opt, 
                    "transcription_method": transcription_method, 
                    "integrator": intgrtr, 
                    "sliding_wrist_offset": sliding_wrist_offset, 
                    "n_multistarts": n_multistarts, 
                    "proc_sol_divs": proc_sol_divs, 
                    "unique_id": unique_id}
    
    task_info_dumper.add_storer(other_stuff, results_path,\
                            "employed_task_info_t" + unique_id,\
                            False)

    task_info_dumper.dump()

    print("\n Task info solution dumped. \n")

    proc_list = [None] * len(proc_sol_divs)
    # launch solvers and solution dumpers on separate processes
    for p in range(len(proc_sol_divs)):

        proc_list[p] = mp_classic.Process(target=sol_main, args=(args, proc_sol_divs[p],\
                                                            q_ig, q_dot_ig, task_copies[p], slvr_copies[p],\
                                                            results_path, opt_results_path, failed_results_path,\
                                                            unique_id,\
                                                            p, ))
        proc_list[p].start()
        
    
    # for p in range(len(proc_sol_divs)): # wait until all processes are finished

    #     proc_list[p].join() # wait until all processes are terminated

