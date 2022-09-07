#!/usr/bin/env python3

import os, argparse

import numpy as np

import rospkg
  
import multiprocessing as mp

from codesign_pyutils.miscell_utils import str2bool, \
                                            compute_solution_divs
from codesign_pyutils.misc_definitions import get_design_map
from codesign_pyutils.task_utils import gen_cl_man_gen_copies, gen_slvr_copies
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.solution_utils import solve_prb_standalone, \
                                        generate_ig            
from codesign_pyutils.post_proc_utils import PostProcS1

from termcolor import colored

def enforce_codes_cnstr_on_ig(q_ig):

    # adding q_codes to the initial guess
    design_var_map = get_design_map()

    design_indeces = [design_var_map["mount_h"],\
        design_var_map["should_w_l"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"],\
        design_var_map["should_w_r"],\
        design_var_map["should_roll_r"],\
        design_var_map["wrist_off_r"]]

    design_indeces_aux = [design_var_map["mount_h"],\
        design_var_map["should_w_l"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"]]

    for i in range(len(q_ig)):

        q_codes_ig = q_ig[i][design_indeces_aux, :]

        q_codes_extended = np.concatenate((q_codes_ig, q_codes_ig[1:]), axis=0)

        q_ig[i][design_indeces, :] = np.transpose(np.tile(q_codes_extended, (len(q_ig[0][0, :]), 1)))

def solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt, cnstr_lmbd,\
            solve_failed_array,
            trial_idxs, 
            n_multistarts, 
            max_retry_n, 
            process_id, 
            q_codes):

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

            solve_failed, solution_time = solve_prb_standalone(task, slvr,\
                            q_ig[multistart_nodes[sol_index] + n_multistarts * trial_index],\
                            q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_index], \
                            is_second_level_opt= True, \
                            q_codes_l1=q_codes)

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

        print_color = "green" if not solve_failed else "yellow"
        print(colored("COMLETED SOLUTION PROCEDURE OF MULTISTART NODE:" + str(multistart_nodes[sol_index]) + \
            ".\nProcess n." + str(process_id) + \
            ".\nIn-process index: " + str(sol_index + 1) + \
            "/" + str(len(multistart_nodes)) + \
            ".\nOpt. cost: " + str(solutions[sol_index]["opt_cost"]), print_color))

        sol_costs[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()
        cnstr_lmbd[sol_index] = slvr.getCnstrLmbdSolDict()

        solve_failed_array[sol_index] = solve_failed

    return solution_time

def sol_main(args, multistart_nodes, q_ig, q_dot_ig, task, slvr, opt_path, fail_path,\
        id_unique,\
        process_id, \
        n_multistarts, 
        max_retry_n, 
        q_codes):
        
    n_multistarts_main = len(multistart_nodes) # number of multistarts assigned to this main instance

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_costs = [1e10] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main
    cnstr_lmbd = [None] * n_multistarts_main
    trial_idxs = [-1] * n_multistarts_main

    solution_time = solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt, cnstr_lmbd,\
            solve_failed_array, 
            trial_idxs, 
            n_multistarts, 
            max_retry_n, 
            process_id, 
            q_codes)

    # solutions packaging for postprocessing
    
    sol_dumper = SolDumper()

    for sol_index in range(len(multistart_nodes)):
        
        full_solution = {**(solutions[sol_index]),
                        **(cnstr_opt[sol_index]),
                        **(cnstr_lmbd[sol_index]),
                        "q_ig": q_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]],
                        "q_dot_ig": q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]], \
                        "multistart_index": multistart_nodes[sol_index], 
                        "trial_index": trial_idxs[sol_index], 
                        "solution_time": solution_time, 
                        "solve_failed": solve_failed_array[sol_index], 
                        "run_id": id_unique}

        if not solve_failed_array[sol_index]:

            sol_dumper.add_storer(full_solution, opt_path,\
                            args.solution_base_name +\
                            "_p" + str(process_id) + \
                            "_r" + str(trial_idxs[sol_index]) + \
                            "_n" + str(multistart_nodes[sol_index]) + \
                            "_t" + \
                            id_unique, False)
        else:

            sol_dumper.add_storer(full_solution, fail_path,\
                            args.solution_base_name +\
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
        description='Second level optimization script for the co-design of RePAIR project')
    parser.add_argument('--n_msrt_trgt', '-mst', type=int,\
                        help = 'number of multistarts (per cluster) to use', default = 100)
    parser.add_argument('--max_trials_factor', '-mtf', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 5)
    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)                      
    parser.add_argument('--res_dirname', '-d', type=str,\
                        help = 'directory name from where results are to be loaded', default = "load_dir")
    
    parser.add_argument('--urdf_full_path', '-urdf', type=str,\
                        help = 'full path to URDF', 
                        default = "/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_urdf/urdf/repair_full.urdf")
    parser.add_argument('--coll_yaml_path', '-coll', type=str,\
                        help = 'full path to collision YAML', 
                        default = "/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_urdf/config/arm_coll.yaml")

    parser.add_argument('--dump_dir_name', '-dfn', type=str,\
                    help = 'dump directory name',
                    default = "second_level")

    parser.add_argument('--load_dir_name', '-ldn', type=str,\
                    help = 'load directory name',
                    default = "first_level")

    parser.add_argument('--res_dir_basename', '-rdbs', type=str,\
                    help = '',
                    default = "test_results")
    parser.add_argument('--solution_base_name', '-sbn', type=str,\
                    help = '',
                    default = "repair_codesign_cl_man_ref_gen")
    
    parser.add_argument('--slvr_type', '-slvr', type = str,\
                        help = 'solver type', default = "ipopt")
    parser.add_argument('--ipopt_tol', '-ipopt_tol', type = np.double,\
                        help = 'IPOPT tolerance', default = 0.0000001)
    parser.add_argument('--ipopt_max_iter', '-max_iter', type = int,\
                        help = 'IPOPT max iterations', default = 350)
    parser.add_argument('--ipopt_cnstr_tol', '-ipopt_cnstr', type = np.double,\
                        help = 'IPOPT constraint violation tolerance', default = 0.000001)
    parser.add_argument('--ipopt_verbose', '-ipopt_v', type = int,\
                        help = 'IPOPT verbose flag', default = 2)

    parser.add_argument('--cl_man_gen_path', '-cl_man_p', type=str,\
                    default = '/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/load_dir')
    parser.add_argument('--cl_man_gen_dirname', '-cl_man_d', type=str,\
                    default = 'cl_man_ref_gen')
    parser.add_argument('--unique_id', '-id', type=str,\
                    help = '', 
                    default = 'prova')
    
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 1)

    parser.add_argument('--sliding_wrist_offset', '-wo', type = np.double,\
                        help = 'sliding_wrist_offset', default = 0.0)
    
    args = parser.parse_args()

    # number of parallel processes on which to run optimization
    # set to number of cpu counts to saturate
    processes_n = mp.cpu_count()

    # useful paths
    dump_folder_name = args.dump_dir_name
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package

    urdf_full_path = args.urdf_full_path
    codesign_path = rospackage.get_path("repair_codesign")

    dump_basepath = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dirname + "/" + dump_folder_name

    coll_yaml_path = args.coll_yaml_path
    solution_base_name = args.solution_base_name

    # load best solution from pipeline results

    # number of solution tries with different (random) initializations
    n_msrt_trgt = args.n_msrt_trgt

    solver_type = args.slvr_type

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
    t_exec_task = 2

    # transcription options (if used)
    transcription_method = 'multiple_shooting'
    intgrtr = 'RK4'
    transcription_opts = dict(integrator = intgrtr)

    sliding_wrist_offset = args.sliding_wrist_offset

    max_retry_n = args.max_trials_factor - 1
    max_ig_trials = n_msrt_trgt * args.max_trials_factor
    proc_sol_divs = compute_solution_divs(n_msrt_trgt, processes_n)

    if  (not os.path.isdir(dump_basepath)):

        os.makedirs(dump_basepath)

    opt_path= args.cl_man_gen_path + "/" + args.cl_man_gen_dirname + "/opt"
    fail_path = args.cl_man_gen_path + "/" + args.cl_man_gen_dirname + "/failed"

    if not os.path.isdir(opt_path):
        os.makedirs(opt_path)
    else:
        os.rmdir(opt_path)
        os.makedirs(opt_path)
    if not os.path.isdir(fail_path):
        os.makedirs(fail_path)
    else:
        os.rmdir(fail_path)
        os.makedirs(fail_path)

    task_copies = [None] * len(proc_sol_divs)
    slvr_copies = [None] * len(proc_sol_divs)

    for p in range(len(proc_sol_divs)):
        
        print(colored("Generating task copy for process n." + str(p), "magenta"))

        task_copies[p] = gen_cl_man_gen_copies(
                                        sliding_wrist_offset, 
                                        urdf_full_path,
                                        t_exec_task,
                                        coll_yaml_path, 
                                        args.weight_class_manip)
        
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

    # inizialize a dumper object for post-processing

    task_info_dumper = SolDumper()

    other_stuff = {"dt": task_copies[0].dt, 
                    "t_exec_task": t_exec_task,
                    "ig_seed": ig_seed, 
                    "solver_type": solver_type, "slvr_opts": slvr_opt, 
                    "transcription_method": transcription_method, 
                    "integrator": intgrtr, 
                    "sliding_wrist_offset": sliding_wrist_offset,
                    "n_msrt_trgt": n_msrt_trgt,
                    "max_retry_n": max_retry_n, 
                    "proc_sol_divs": np.array(proc_sol_divs, dtype=object),
                    "n_int": task_copies[0].n_int
                    }

    task_info_dumper.add_storer(other_stuff, dump_basepath,\
                            "cl_man_ref_gen",\
                            False)

    task_info_dumper.dump()

    proc_list = [None] * len(proc_sol_divs)
    # launch solvers and solution dumpers on separate processes

    q_codes = np.array([0.4, 0.3, 0.57, 0.0])
    for p in range(len(proc_sol_divs)): # for each process (for each cluster, multistarts are solved parallelizing on processes)

        proc_list[p] = mp.Process(target=sol_main, args=(args, proc_sol_divs[p],\
                                                            q_ig, q_dot_ig, task_copies[p], slvr_copies[p],\
                                                            opt_path, fail_path,\
                                                            args.unique_id,\
                                                            p,
                                                            n_msrt_trgt, 
                                                            max_retry_n,
                                                            q_codes,))
        proc_list[p].start()
    
    for p in range(len(proc_sol_divs)):

        while proc_list[p].is_alive():

            continue
                
    for p in range(len(proc_sol_divs)):
        
        print(colored("Joining process " + str(p), "magenta"))

        proc_list[p].join() # wait until all processes of cluster cl are terminated


