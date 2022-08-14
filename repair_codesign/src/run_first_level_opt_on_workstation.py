#!/usr/bin/env python3

from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver
import os, argparse

import numpy as np

import subprocess

import rospkg

from codesign_pyutils.miscell_utils import str2bool, compute_solution_divs,\
                                            gen_y_sampling
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.task_utils import solve_prb_standalone, \
                                        generate_ig              
from codesign_pyutils.tasks import TaskGen

import multiprocessing as mp

from datetime import datetime
from datetime import date

from termcolor import colored

def gen_task_copies(filling_n_nodes, sliding_wrist_offset, 
                    n_y_samples, y_sampl_ub, 
                    coll_yaml_path = ""):

    
    y_sampling = gen_y_sampling(n_y_samples, y_sampl_ub)

    right_arm_picks = np.array([True] * len(y_sampling))
    for i in range(len(y_sampling)):
        
        if y_sampling[i] <= 0 : # on the right
            
            right_arm_picks[i] = True
        
        else:

            right_arm_picks[i] = False

    # initialize problem task
    task = TaskGen(filling_n_nodes = filling_n_nodes, \
                                    sliding_wrist_offset = sliding_wrist_offset,\
                                    coll_yaml_path = coll_yaml_path)

    task.add_tasks(y_sampling, right_arm_picks)

    # initialize problem
    task.init_prb(urdf_full_path,
                    weight_glob_man = args.weight_global_manip, weight_class_man = args.weight_class_manip,\
                    tf_single_task = t_exec_task)

    print(colored("Task node list: " + str(task.nodes_list), "magenta"))
    print(colored("Task list: " + str(task.task_list), "magenta"))
    print(colored("Task names: " + str(task.task_names), "magenta"))
    print(colored("Task dict: " + str( task.task_dict), "magenta"))
    print(colored("Total employed nodes: " + str(task.total_nnodes), "magenta"))
    print(colored("Number of added subtasks:" + str(task.n_of_tasks) + "\n", "magenta"))

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

def solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array,
            trial_idxs, 
            n_multistarts, 
            max_retry_n, 
            process_id):

    solution_time = -1.0

    for sol_index in range(len(multistart_nodes)):
        
        trial_index = 0
        solve_failed = True

        print(colored("\nSOLVING PROBLEM OF MULTISTART NODE: " + str(multistart_nodes[sol_index]) +\
                        ".\nProcess n." + str(process_id) + \
                        ".\nIn-process index: " + str(sol_index + 1) + \
                        "/" + str(len(multistart_nodes)), "magenta"))

        print("\n")

        while True:

            solve_failed, solution_time = solve_prb_standalone(task,\
                slvr, q_ig[multistart_nodes[sol_index] + n_multistarts * trial_index],\
                q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_index])

            if trial_index < max_retry_n: # not reached maximum retry number

                if solve_failed:
                    
                    trial_index = trial_index + 1

                    print(colored("Solution of node " + str(multistart_nodes[sol_index]) + \
                        "- in-process index: "  + str(sol_index + 1) + "/" + str(len(multistart_nodes)) + \
                        " failed --> starting trial n." + str(trial_index), "yellow"))
                
                else:

                    break # exit while
            
            else:

                break # exit loop and read solution (even if it failed
        
        trial_idxs[sol_index] = trial_index # assign trial index (== 0 if solution is optimal on first attempt)

        solutions[sol_index] = slvr.getSolutionDict()

        print_color = "magenta" if not solve_failed else "yellow"
        print(colored("COMLETED SOLUTION PROCEDURE OF MULTISTART NODE:" + str(multistart_nodes[sol_index]) + \
            ".\nProcess n." + str(process_id) + \
            ".\nIn-process index: " + str(sol_index + 1) + \
            "/" + str(len(multistart_nodes)) + \
            ".\nOpt. cost: " + str(solutions[sol_index]["opt_cost"]), print_color))

        sol_costs[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()

        solve_failed_array[sol_index] = solve_failed

    return solution_time

def sol_main(args, multistart_nodes, q_ig, q_dot_ig, task, slvr, result_path, opt_path, fail_path,\
        id_unique,\
        process_id, \
        n_multistarts, 
        max_retry_n):
    
    n_multistarts_main = len(multistart_nodes) # number of multistarts assigned to this main instance

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_costs = [1e10] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main
    trial_idxs = [-1] * n_multistarts_main

    solution_time = solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array, 
            trial_idxs, 
            n_multistarts, 
            max_retry_n, 
            process_id)

    # solutions packaging for postprocessing
    
    sol_dumper = SolDumper()

    for sol_index in range(len(multistart_nodes)):
            
        full_solution = {**(solutions[sol_index]),
                        **(cnstr_opt[sol_index]),
                        "q_ig": q_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]],
                        "q_dot_ig": q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]], \
                        "solution_index": multistart_nodes[sol_index], 
                        "trial_index": trial_idxs[sol_index], 
                        "solution_time": solution_time, 
                        "solve_failed": solve_failed_array[sol_index]}

        if not solve_failed_array[sol_index]:

            sol_dumper.add_storer(full_solution, opt_path,\
                            solution_base_name +\
                            "_p" + str(process_id) + \
                            "_r" + str(trial_idxs[sol_index]) + \
                            "_n" + str(multistart_nodes[sol_index]) + \
                            "_t" + \
                            id_unique, False)
        else:

            sol_dumper.add_storer(full_solution, fail_path,\
                            solution_base_name +\
                            "_p" + str(process_id) + \
                            "_r" + str(trial_idxs[sol_index]) + \
                            "_n" + str(multistart_nodes[sol_index]) + \
                            "_t" + \
                            id_unique, False)

    sol_dumper.dump() 

    print(colored("\nSolutions of process " + str(process_id) + \
        " dumped. \n", "magenta"))

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='First level optimization script for the co-design of RePAIR project')

    # parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
    #                     help = 'whether to load ig from files', default = False)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 1)
    parser.add_argument('--use_classical_man', '-ucm', type=str2bool,\
                        help = 'whether to use the classical manipulability index', default = False)

    parser.add_argument('--n_msrt_trgt', '-mst', type=int,\
                        help = 'number of  target optimal solutions the script will try to find',
                        default = 200)
    parser.add_argument('--max_trials_factor', '-mtf', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 15)

    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)       

    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = True)

    parser.add_argument('--sliding_wrist_offset', '-wo', type = np.double,\
                        help = 'sliding_wrist_offset', default = 0.0)

    parser.add_argument('--filling_nnodes', '-fnn', type = int,\
                        help = 'filling nodes between base task nodes', default = 5)
    parser.add_argument('--n_y_samples', '-nys', type = int,\
                        help = 'number of y-axis samples on which tasks are placed', default = 4)
    parser.add_argument('--y_sampl_ub', '-yub', type = np.double,\
                        help = 'upper bound of the y sampling', default = 0.4)
    parser.add_argument('--rot_error_epsi', '-rot_ep', type = np.double,\
                        help = 'rotation error tolerance', default = 0.0000001)
    parser.add_argument('--t_exec_task', '-t_exec', type = np.double,\
                        help = 'execution time for a single task', default = 6.0)
    
    parser.add_argument('--slvr_type', '-slvr', type = str,\
                        help = 'solver type', default = "ipopt")
    parser.add_argument('--ipopt_tol', '-ipopt_tol', type = np.double,\
                        help = 'IPOPT tolerance', default = 0.0000001)
    parser.add_argument('--ipopt_max_iter', '-max_iter', type = int,\
                        help = 'IPOPT max iterations', default = 1000)
    parser.add_argument('--ipopt_cnstr_tol', '-ipopt_cnstr', type = np.double,\
                        help = 'IPOPT constraint violation tolerance', default = 0.000001)
    parser.add_argument('--ipopt_verbose', '-ipopt_v', type = int,\
                        help = 'IPOPT verbose flag', default = 4)

    parser.add_argument('--run_externally', '-run_ext', type=str2bool,\
                        help = 'whether this script is run from the higher level pipeline script', default = False)
    parser.add_argument('--unique_id', '-id', type=str,\
                        help = 'unique id passed from higherl level script (only used if run_externally == True)',
                        default = "")
    parser.add_argument('--dump_folder_name', '-dfn', type=str,\
                    help = 'dump directory name',
                    default = "first_level")
    parser.add_argument('--res_dir_basename', '-rdbs', type=str,\
                    help = '',
                    default = "test_results")
    parser.add_argument('--solution_base_name', '-sbn', type=str,\
                    help = '',
                    default = "repair_codesign_opt_l1")

    args = parser.parse_args()
    
    unique_id = ""

    if args.run_externally:

        unique_id = args.unique_id
    
    else:

        # unique id used for generation of results
        unique_id = date.today().strftime("%d-%m-%Y") + "-" +\
                            datetime.now().strftime("%H_%M_%S")

    # number of parallel processes on which to run optimization
    # set to number of cpu counts to saturate
    processes_n = mp.cpu_count()

    # useful paths
    dump_folder_name = args.dump_folder_name
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
    urdf_name = "repair_full"
    urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
    xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
    codesign_path = rospackage.get_path("repair_codesign")
    results_path = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dir_basename + "_" +\
                unique_id + "/" + dump_folder_name
    opt_results_path = results_path + "/opt" 
    failed_results_path = results_path + "/failed"

    coll_yaml_name = "arm_coll.yaml"
    coll_yaml_path = rospackage.get_path("repair_urdf") + "/config/" + coll_yaml_name
    
    solution_base_name = args.solution_base_name

    sliding_wrist_command = "is_sliding_wrist:=" + "true"
    show_softhand_command = "show_softhand:=" + "true"
    show_coll_command = "show_coll:=" + "true"
    # preliminary ops

    try:

        
        # print(sliding_wrist_command)
        xacro_gen = subprocess.check_call(["xacro",\
                                        xacro_full_path, \
                                        sliding_wrist_command, \
                                        show_softhand_command, \
                                        show_coll_command, \
                                        "-o", 
                                        urdf_full_path])

    except:

        print(colored('Failed to generate URDF.', "red"))

    # task-specific options
    filling_n_nodes = args.filling_nnodes
    rot_error_epsi = args.rot_error_epsi

    # samples
    n_y_samples = args.n_y_samples
    y_sampl_ub = args.y_sampl_ub

    # number of solution tries with different (random) initializations
    n_msrt_trgt = args.n_msrt_trgt

    # solver options
    solver_type = args.slvr_type
    if args.use_ma57:

        slvr_opt = {
            "ipopt.tol": args.ipopt_tol, 
            "ipopt.max_iter": args.ipopt_max_iter,
            "ipopt.constr_viol_tol": args.ipopt_cnstr_tol,
            "ipopt.print_level": args.ipopt_verbose, 
            "ilqr.verbose": True, 
            "ipopt.linear_solver": "ma57"}

    else:

        slvr_opt = {
            "ipopt.tol": args.ipopt_tol, 
            "ipopt.max_iter": args.ipopt_max_iter,
            "ipopt.constr_viol_tol": args.ipopt_cnstr_tol,
            "ipopt.print_level": args.ipopt_verbose, 
            "ilqr.verbose": True, 
            "ipopt.linear_solver": "mumps"}


    full_file_paths = None # not used

    # seed used for random number generation
    ig_seed = args.ig_seed

    # single task execution time
    t_exec_task = args.t_exec_task

    # transcription options (if used)
    transcription_method = 'multiple_shooting'
    intgrtr = 'RK4'
    transcription_opts = dict(integrator = intgrtr)

    sliding_wrist_offset = args.sliding_wrist_offset

    max_retry_n = args.max_trials_factor - 1
    max_ig_trials = n_msrt_trgt * args.max_trials_factor
    proc_sol_divs = compute_solution_divs(n_msrt_trgt, processes_n)

    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)
        os.makedirs(opt_results_path)
        os.makedirs(failed_results_path)

    task_copies = [None] * len(proc_sol_divs)
    slvr_copies = [None] * len(proc_sol_divs)

    for p in range(len(proc_sol_divs)):
        
        task_copies[p], slvr_copies[p] = gen_task_copies(filling_n_nodes,
                                                sliding_wrist_offset, 
                                                n_y_samples, y_sampl_ub, 
                                                coll_yaml_path)

    # generating initial guesses, based on the script arguments
    q_ig, q_dot_ig =  generate_ig(args, full_file_paths,\
                                    task_copies[0],\
                                    max_ig_trials, ig_seed,\
                                    False)
    
    # dumping info on the task 

    # inizialize a dumper object for post-processing

    task_info_dumper = SolDumper()

    other_stuff = {"dt": task_copies[0].dt, "filling_nodes": task_copies[0].filling_n_nodes,
                    "task_base_nnodes": task_copies[0].task_base_n_nodes_dict,
                    "right_arm_picks": task_copies[0].rght_arm_picks, 
                    "use_classical_man": args.use_classical_man,
                    "w_man_base": args.weight_global_manip, 
                    "w_clman_base": args.weight_class_manip,
                    "w_man_actual": task_copies[0].weight_glob_man, 
                    "w_clman_actual": task_copies[0].weight_classical_man, 
                    "nodes_list": task_copies[0].nodes_list, 
                    "tasks_list": task_copies[0].task_list,
                    "tasks_dict": task_copies[0].task_dict, 
                    "y_sampl_ub": y_sampl_ub, "n_y_samples": n_y_samples, 
                    "ig_seed": ig_seed, 
                    "solver_type": solver_type, "slvr_opts": slvr_opt, 
                    "transcription_method": transcription_method, 
                    "integrator": intgrtr, 
                    "sliding_wrist_offset": sliding_wrist_offset, 
                    "n_msrt_trgt": n_msrt_trgt,
                    "max_retry_n": max_retry_n,  
                    "proc_sol_divs": np.array(proc_sol_divs, dtype=object), 
                    "unique_id": unique_id, 
                    "rot_error_epsi": rot_error_epsi, 
                    "t_exec_task": t_exec_task, 
                    "sliding_wrist_offset": sliding_wrist_offset
                    }
    
    task_info_dumper.add_storer(other_stuff, results_path,\
                            "employed_task_info_t" + unique_id,\
                            False)

    task_info_dumper.dump()

    print(colored("\nTask info solution dumped. \n", "magenta"))

    progress_bar_index = 0
    proc_list = [None] * len(proc_sol_divs)
    # launch solvers and solution dumpers on separate processes
    
    for p in range(len(proc_sol_divs)):

        proc_list[p] = mp.Process(target=sol_main, args=(args, proc_sol_divs[p],\
                                                            q_ig, q_dot_ig, task_copies[p], slvr_copies[p],\
                                                            results_path, opt_results_path, failed_results_path,\
                                                            unique_id,\
                                                            p,
                                                            n_msrt_trgt, 
                                                            max_retry_n,))
        proc_list[p].start()

    for p in range(len(proc_sol_divs)):

            while proc_list[p].is_alive():

                continue
                    
    for p in range(len(proc_sol_divs)):

        print(colored("Joining process" + str(p), "magenta"))

        proc_list[p].join()


