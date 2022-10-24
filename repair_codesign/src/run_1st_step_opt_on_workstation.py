#!/usr/bin/env python3

from horizon.ros.replay_trajectory import *

import os, argparse

import numpy as np

import subprocess

import rospkg

from codesign_pyutils.miscell_utils import str2bool, compute_solution_divs

from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.solution_utils import solve_prb_standalone, \
                                        generate_ig              
from codesign_pyutils.tasks import CodesTaskGen

import multiprocessing as mp

from datetime import datetime
from datetime import date

from termcolor import colored

from codesign_pyutils.misc_definitions import get_design_map

from codesign_pyutils.task_utils import gen_task_copies,\
                                        gen_slvr_copies,\
                                        sol_main_s1_s1,
                                        solve_s1,\
                                        enforce_codes_cnstr_on_ig

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='First level optimization script for the co-design of RePAIR project')

    # parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
    #                     help = 'whether to load ig from files', default = False)

    # ANNOTATION ON WEIGHTS; the average max. joint velocity resulting from the optimization 
    # is of the order of 0.1 rad/s the max. joint torque necessary to execute the motion is of the order
    # of 10 Nm since the costs are squared, this gives a ratio of the costs, if  using unit weights,
    # of 0.01/100 = 1e-4 so, to balance static torque and non-local man weights, the first should be approximately
    # 1e-4 smaller than the second. 

    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 0.1)
    parser.add_argument('--weight_static_tau', '-wstau', type = np.double,\
                        help = 'weight static torque minimization term', default = 0.000005)
    
    parser.add_argument('--use_classical_man', '-ucm', type=str2bool,\
                        help = 'whether to use the classical manipulability index', default = False)

    parser.add_argument('--use_static_tau', '-ustau', type=str2bool,\
                        help = 'whether to use the static tau minimization cost', default = False)

    parser.add_argument('--urdf_full_path', '-urdf', type=str,\
                        help = 'full path to URDF', default = "")
    parser.add_argument('--coll_yaml_path', '-coll', type=str,\
                        help = 'full path to collision YAML', default = "")
    parser.add_argument('--cost_weights_yaml_path', '-weights', type=str,\
                        help = 'full path to weights YAML', default = "")

    parser.add_argument('--is_in_place_flip', '-iplf', type=str2bool,\
                        help = 'whether to use in place flip task', default = True)
    parser.add_argument('--is_biman_pick', '-ibp', type=str2bool,\
                        help = 'whether to use bimanual pick task', default = False)

    parser.add_argument('--n_msrt_trgt', '-mst', type=int,\
                        help = 'number of  target optimal solutions the script will try to find',
                        default = 10)
    parser.add_argument('--max_trials_factor', '-mtf', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 15)

    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)       

    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = True)

    parser.add_argument('--is_sliding_wrist', '-isw', type = str2bool,\
                        help = 'if wrist off. is to be used as an additional codes variable', default = False)
    parser.add_argument('--sliding_wrist_offset', '-wo', type = np.double,\
                        help = 'sliding_wrist_offset', default = 0.0)

    parser.add_argument('--filling_nnodes', '-fnn', type = int,\
                        help = 'filling nodes between base task nodes', default = 0)

    parser.add_argument('--n_y_samples_flip', '-nysf', type = int,\
                        help = 'number of y-axis samples on which tasks(flipping task) are placed', default = 4)
    parser.add_argument('--y_sampl_ub_flip', '-yubf', type = np.double,\
                        help = 'upper bound of the y sampling (flipping task)', default = 0.4)
    parser.add_argument('--n_y_samples_biman', '-nysb', type = int,\
                        help = 'number of y-axis samples on which tasks(bimanual task) are placed', default = 2)
    parser.add_argument('--y_sampl_ub_biman', '-yubb', type = np.double,\
                        help = 'upper bound of the y sampling (bimanual task)', default = 0.2)

    parser.add_argument('--rot_error_epsi', '-rot_ep', type = np.double,\
                        help = 'rotation error tolerance', default = 0.0000001)
    parser.add_argument('--t_exec_task', '-t_exec', type = np.double,\
                        help = 'execution time for a single task', default = 6.0)
    
    parser.add_argument('--slvr_type', '-slvr', type = str,\
                        help = 'solver type', default = "ipopt")
    parser.add_argument('--ipopt_tol', '-ipopt_tol', type = np.double,\
                        help = 'IPOPT tolerance', default = 0.00001)
    parser.add_argument('--ipopt_max_iter', '-max_iter', type = int,\
                        help = 'IPOPT max iterations', default = 450)
    parser.add_argument('--ipopt_cnstr_tol', '-ipopt_cnstr', type = np.double,\
                        help = 'IPOPT constraint violation tolerance', default = 0.000001)
    parser.add_argument('--ipopt_verbose', '-ipopt_v', type = int,\
                        help = 'IPOPT verbose flag', default = 2)

    parser.add_argument('--run_externally', '-run_ext', type=str2bool,\
                        help = 'whether this script is run from the higher level pipeline script', default = False)
    parser.add_argument('--unique_id', '-id', type=str,\
                        help = 'unique id passed from higher level script (only used if run_externally == True)',
                        default = "")
    parser.add_argument('--dump_folder_name', '-dfn', type=str,\
                    help = 'dump directory name',
                    default = "first_step")
    parser.add_argument('--res_dir_basename', '-rdbs', type=str,\
                    help = '',
                    default = "test_results")
    parser.add_argument('--solution_base_name', '-sbn', type=str,\
                    help = '',
                    default = "repair_codesign_opt_s1")

    args = parser.parse_args()
    
    is_second_lev_opt = False

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
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    urdf_full_path = args.urdf_full_path
    codesign_path = rospackage.get_path("repair_codesign")

    dump_folder_name = args.dump_folder_name
    results_path = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dir_basename + "_" +\
                unique_id + "/" + dump_folder_name

    opt_results_path = results_path + "/opt" 
    failed_results_path = results_path + "/failed"
    coll_yaml_path = args.coll_yaml_path
    cost_weights_yaml_path = args.cost_weights_yaml_path

    solution_base_name = args.solution_base_name

    # task-specific options
    filling_n_nodes = args.filling_nnodes
    rot_error_epsi = args.rot_error_epsi

    # samples
    n_y_samples = [args.n_y_samples_flip, args.n_y_samples_biman]
    y_sampl_ub = [args.y_sampl_ub_flip, args.y_sampl_ub_biman]

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
    is_sliding_wrist = args.is_sliding_wrist

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
        
        print(colored("Generating task copy for process n." + str(p), "magenta"))

        task_copies[p] = gen_task_copies(args.weight_global_manip,
                                        args.weight_class_manip,
                                        args.weight_static_tau,
                                        filling_n_nodes,
                                        sliding_wrist_offset, 
                                        n_y_samples, y_sampl_ub,
                                        urdf_full_path,
                                        t_exec_task,
                                        rot_error_epsi,
                                        args.use_classical_man,
                                        args.use_static_tau,
                                        is_sliding_wrist,
                                        coll_yaml_path,
                                        cost_weights_yaml_path,
                                        is_second_lev_opt, 
                                        args.is_in_place_flip, 
                                        args.is_biman_pick)
        
        slvr_copies[p] = gen_slvr_copies(task_copies[p],
                            solver_type,
                            transcription_method, 
                            transcription_opts, 
                            slvr_opt)

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
                    "use_static_tau": args.use_static_tau,
                    "w_man_base": args.weight_global_manip, 
                    "w_clman_base": args.weight_class_manip,
                    "w_stau_base": args.weight_static_tau,
                    "w_man_actual": task_copies[0].weight_glob_man, 
                    "w_clman_actual": task_copies[0].weight_classical_man, 
                    "w_stau_actual": task_copies[0].weight_static_tau,
                    "w_rel_mat_man": task_copies[0].vel_weights, 
                    "w_rel_clman_rot": task_copies[0].weight_clman_rot, 
                    "w_rel_clman_trasl": task_copies[0].weight_clman_trasl, 
                    "w_rel_mat_stau": task_copies[0].weight_static_tau,
                    "nodes_list": task_copies[0].nodes_list, 
                    "tasks_list": task_copies[0].task_list,
                    "tasks_dict": task_copies[0].task_dict, 
                    "y_sampl_ub": y_sampl_ub, "n_y_samples": n_y_samples, 
                    "ig_seed": ig_seed, 
                    "solver_type": solver_type, "slvr_opts": slvr_opt, 
                    "transcription_method": transcription_method, 
                    "integrator": intgrtr, 
                    "sliding_wrist_offset": sliding_wrist_offset, 
                    "is_sliding_wrist": is_sliding_wrist,
                    "n_msrt_trgt": n_msrt_trgt,
                    "max_retry_n": max_retry_n,  
                    "proc_sol_divs": np.array(proc_sol_divs, dtype=object), 
                    "unique_id": unique_id, 
                    "rot_error_epsi": rot_error_epsi, 
                    "t_exec_task": t_exec_task, 
                    "is_in_place_flip": args.is_in_place_flip, 
                    "is_biman_pick": args.is_biman_pick, 
                    "n_int": task_copies[0].n_int}
    
    task_info_dumper.add_storer(other_stuff, results_path,\
                            "first_step_info_t" + unique_id,\
                            False)

    task_info_dumper.dump()

    print(colored("\nTask info solution dumped. \n", "magenta"))

    progress_bar_index = 0
    proc_list = [None] * len(proc_sol_divs)
    # launch solvers and solution dumpers on separate processes
    
    for p in range(len(proc_sol_divs)):

        proc_list[p] = mp.Process(target=sol_main_s1, args=(args, proc_sol_divs[p],\
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


