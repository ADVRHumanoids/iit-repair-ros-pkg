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

import multiprocessing as mp_classic

from datetime import datetime
from datetime import date

def solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array):

    sol_index = 0 # different index

    solution_time = -1.0

    for node in multistart_nodes:

        print("\n SOLVING PROBLEM N.: ", node + 1)
        print("\n")
                    
        solve_failed, solution_time = solve_prb_standalone(task, slvr, q_ig[node], q_dot_ig[node])
        solutions[sol_index] = slvr.getSolutionDict()

        print("Solution cost " + str(node) + ": ", solutions[sol_index]["opt_cost"])
        sol_costs[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()

        solve_failed_array[sol_index] = solve_failed

        sol_index = sol_index + 1
    
    return solution_time

def gen_task_copies(filling_n_nodes, sliding_wrist_offset, 
                    n_y_samples, y_sampl_ub):

    
    y_sampling = gen_y_sampling(n_y_samples, y_sampl_ub)

    right_arm_picks = np.array([True] * len(y_sampling))
    for i in range(len(y_sampling)):
        
        if y_sampling[i] <= 0 : # on the right
            
            right_arm_picks[i] = True
        
        else:

            right_arm_picks[i] = False

    # initialize problem task
    task = TaskGen(filling_n_nodes = filling_n_nodes, \
                                    sliding_wrist_offset = sliding_wrist_offset)

    task.add_tasks(y_sampling, right_arm_picks)

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

    solution_time = solve(multistart_nodes,\
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
                        **{"solution_index": node}, 
                        "solution_time": solution_time, 
                        "solve_failed": solve_failed_array[sol_index]}

        if not solve_failed_array[sol_index]:

            sol_dumper.add_storer(full_solution, opt_path,\
                            solution_base_name +\
                            "_p" +\
                            str(process_id) + \
                            "_n" + str(node) +\
                            "_t" + \
                            id_unique, False)
        else:

            sol_dumper.add_storer(full_solution, fail_path,\
                            solution_base_name +\
                            "_p" +\
                            str(process_id) + \
                            "_n" + str(node) +\
                            "_t" + \
                            id_unique, False)

        sol_index = sol_index + 1

    sol_dumper.dump() 

    print("\n Solutions of process " + str(process_id) + " dumped. \n")
                    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='First level optimization script for the co-design of RePAIR project')

    # parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
    #                     help = 'whether to load ig from files', default = False)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 0.01)
    parser.add_argument('--use_classical_man', '-ucm', type=str2bool,\
                        help = 'whether to use the classical manipulability index', default = False)

    parser.add_argument('--n_multistarts', '-msn', type=int,\
                        help = 'number of multistarts to use', default = 200)
    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)       

    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = False)

    parser.add_argument('--sliding_wrist_offset', '-wo', type = np.double,\
                        help = 'sliding_wrist_offset', default = 0.0)

    parser.add_argument('--filling_nnodes', '-fnn', type = int,\
                        help = 'filling nodes between base task nodes', default = 0)
    parser.add_argument('--n_y_samples', '-nys', type = int,\
                        help = 'number of y-axis samples on which tasks are placed', default = 5)
    parser.add_argument('--y_sampl_ub', '-yub', type = np.double,\
                        help = 'upper bound of the y sampling', default = 0.4)
    parser.add_argument('--rot_error_epsi', '-rot_ep', type = np.double,\
                        help = 'rotation error tolerance', default = 0.0000001)
    parser.add_argument('--t_exec_task', '-t_exec', type = np.double,\
                        help = 'execution time for a single task', default = 6.0)
                       
    parser.add_argument('--ipopt_tol', '-ipopt_tol', type = np.double,\
                        help = 'IPOPT tolerance', default = 0.0000001)
    parser.add_argument('--ipopt_max_iter', '-max_iter', type = int,\
                        help = 'IPOPT max iterations', default = 1000)
    parser.add_argument('--ipopt_cnstr_tol', '-ipopt_cnstr', type = np.double,\
                        help = 'IPOPT constraint violation tolerance', default = 0.000001)
    parser.add_argument('--ipopt_verbose', '-ipopt_v', type = int,\
                        help = 'IPOPT verbose flag', default = 4)

    args = parser.parse_args()
    
    # unique id used for generation of results
    unique_id = date.today().strftime("%d-%m-%Y") + "-" +\
                        datetime.now().strftime("%H_%M_%S")

    # number of parallel processes on which to run optimization
    # set to number of cpu counts to saturate
    processes_n = mp_classic.cpu_count()

    # useful paths
    dump_folder_name = "first_level"
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
    urdf_name = "repair_full"
    urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
    xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
    codesign_path = rospackage.get_path("repair_codesign")
    results_path = codesign_path + "/test_results/test_results_" +\
                unique_id + "/" + dump_folder_name
    opt_results_path = results_path + "/opt" 
    failed_results_path = results_path + "/failed"

    solution_base_name = "repair_codesign_opt_l1"

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

        print('Failed to generate URDF.')

    # task-specific options
    filling_n_nodes = args.filling_nnodes
    rot_error_epsi = args.rot_error_epsi

    # samples
    n_y_samples = args.n_y_samples
    y_sampl_ub = args.y_sampl_ub

    # number of solution tries with different (random) initializations
    n_multistarts = args.n_multistarts

    # solver options
    solver_type = 'ipopt'
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
                    "n_multistarts": n_multistarts, 
                    # "proc_sol_divs": proc_sol_divs, 
                    "unique_id": unique_id, 
                    "rot_error_epsi": rot_error_epsi, 
                    "t_exec_task": t_exec_task, 
                    "sliding_wrist_offset": sliding_wrist_offset
                    }
    
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
        
    for p in range(len(proc_sol_divs)):

            while proc_list[p].is_alive():

                continue
                    
    for p in range(len(proc_sol_divs)):

        print("Joining process" + str(p))

        proc_list[p].join()


