#!/usr/bin/env python3

import os, argparse

import numpy as np

import subprocess

import rospkg

from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.solution_utils import solve_prb_standalone, \
                                        generate_ig              

import multiprocessing as mp

from codesign_pyutils.miscell_utils import str2bool, \
                                            extract_q_design,\
                                            compute_solution_divs

from codesign_pyutils.clustering_utils import Clusterer
from codesign_pyutils.misc_definitions import get_design_map
from codesign_pyutils.load_utils import LoadSols

from codesign_pyutils.task_utils import gen_task_copies, gen_slvr_copies

from termcolor import colored

def add_l1_codes2ig(q_codes_l1, q_ig):

    # adding q_codes to the initial guess
    design_var_map = get_design_map()

    design_indeces = [design_var_map["mount_h"],\
        design_var_map["should_w_l"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"],\
        design_var_map["should_w_r"],\
        design_var_map["should_roll_r"],\
        design_var_map["wrist_off_r"]]

    q_codes_l1_extended = np.concatenate((q_codes_l1, q_codes_l1[1:]), axis=0)

    for i in range(len(q_ig)):

        q_ig[i][design_indeces, :] = np.transpose(np.tile(q_codes_l1_extended, (len(q_ig[0][0, :]), 1)))

def solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array, 
            q_codes_l1, 
            cluster_id, 
            trial_idxs, 
            n_multistarts, 
            max_retry_n, 
            process_id):

    solution_time = -1.0

    for sol_index in range(len(multistart_nodes)):

        trial_index = 0
        solve_failed = True

        print(colored("\nSOLVING PROBLEM OF MULTISTART NODE: " + str(multistart_nodes[sol_index]) +\
                        ".\nCluster n." + str(cluster_id) + \
                        ".\nProcess n." + str(process_id) + \
                        ".\nIn-process index: " + str(sol_index + 1) + \
                        "/" + str(len(multistart_nodes)), "magenta"))
        print("\n")

        while True:
            
            solve_failed, solution_time = solve_prb_standalone(task, slvr,\
                                q_ig[multistart_nodes[sol_index] + n_multistarts * trial_index],\
                                q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_index], \
                                is_second_level_opt= True, \
                                q_codes_l1=q_codes_l1)
            
            if trial_index < max_retry_n: # not reached maximum retry number

                if solve_failed:
                    
                    trial_index = trial_index + 1

                    print(colored("Solution of node " + str(multistart_nodes[sol_index]) + \
                        "- in-process index: "  + str(sol_index + 1) + "/" + str(len(multistart_nodes)) + \
                        "\n (cluster n." + str(cluster_id) + ")\n" + \
                        " failed --> starting trial n." + str(trial_index), "yellow"))
                
                else:

                    break # exit while
            
            else:

                break # exit loop and read solution (even if it failed)

        
        trial_idxs[sol_index] = trial_index # assign trial index (== 0 if solution is optimal on first attempt)

        solutions[sol_index] = slvr.getSolutionDict() # get the first available optimal solution

        print_color = "green" if not solve_failed else "yellow"
        print(colored("COMLETED SOLUTION PROCEDURE OF MULTISTART NODE:" + str(multistart_nodes[sol_index]) + \
            ".\nCluster n." + str(cluster_id) + \
            ".\nProcess n." + str(process_id) + \
            ".\nIn-process index: " + str(sol_index + 1) + \
            "/" + str(len(multistart_nodes)) + \
            ".\nOpt. cost: " + str(solutions[sol_index]["opt_cost"]), print_color))

        sol_costs[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()

        solve_failed_array[sol_index] = solve_failed # for now, solve_failed will always be true
    
    return solution_time
    
def sol_main(multistart_nodes, q_ig, q_dot_ig, task, slvr, opt_path, fail_path,\
        id_unique,\
        process_id,
        cluster_id,
        first_lev_sol_id, 
        q_codes_l1, 
        n_multistarts, 
        max_retry_n):
    
    n_multistarts_main = len(multistart_nodes) # number of multistarts assigned to this main instance

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_costs = [1e10] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main
    trial_idxs = [-1] * n_multistarts_main

    # adding q_codes to the initial guess
    add_l1_codes2ig(q_codes_l1, q_ig)

    solution_time = solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array, 
            q_codes_l1, 
            cluster_id, 
            trial_idxs, 
            n_multistarts, 
            max_retry_n, 
            process_id)

    # solutions packaging for postprocessing
    
    sol_dumper = SolDumper()

    for sol_index in range(len(multistart_nodes)):
            
        full_solution = {**(solutions[sol_index]),
                        **(cnstr_opt[sol_index]),
                        "q_ig": q_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]],\
                        "q_dot_ig": q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]], \
                        "multistart_index": multistart_nodes[sol_index], 
                        "trial_index": trial_idxs[sol_index], 
                        "solution_time": solution_time, 
                        "cluster_id": cluster_id, 
                        "first_lev_sol_id": first_lev_sol_id, 
                        "solve_failed": solve_failed_array[sol_index] ,
                        "run_id": id_unique}

        if not solve_failed_array[sol_index]: # for now, the script will only save optimal solutions (does not make much sense to save also failed, at this point)

            sol_dumper.add_storer(full_solution, opt_path,\
                            solution_base_name + \
                            "_id" + str(first_lev_sol_id) + \
                            "_cl" + str(cluster_id) + \
                            "_p" + str(process_id) + \
                            "_r" + str(trial_idxs[sol_index]) + \
                            "_n" + str(multistart_nodes[sol_index]) + \
                            "_t" + \
                             id_unique, False)
        else:

            sol_dumper.add_storer(full_solution, fail_path,\
                            solution_base_name + \
                            "_id" + str(first_lev_sol_id) + \
                            "_cl" + str(cluster_id) + \
                            "_p" + str(process_id) + \
                            "_r" + str(trial_idxs[sol_index]) + \
                            "_n" + str(multistart_nodes[sol_index]) + \
                            "_t" + \
                             id_unique, False)

    sol_dumper.dump() 

    print(colored("\nSolutions of process " + str(process_id) + \
            " ; cluster n. " + str(cluster_id) + \
            " dumped. \n", "magenta"))
                    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='Second level optimization script for the co-design of RePAIR project')
    parser.add_argument('--n_msrt_trgt', '-mst', type=int,\
                        help = 'number of multistarts (per cluster) to use', default = 4)
    parser.add_argument('--max_trials_factor', '-mtf', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 15)
    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)                      
    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = False)
    parser.add_argument('--res_dirname', '-d', type=str,\
                        help = 'directory name from where results are to be loaded', default = "load_dir")
    parser.add_argument('--n_clust', '-nc', type=int,\
                        help = 'number of clusters to be selected', default = 5)
    parser.add_argument('--ipopt_verbose', '-ipopt_v', type = int,\
                        help = 'IPOPT verbose flag', default = 2)
    
    parser.add_argument('--urdf_full_path', '-urdf', type=str,\
                        help = 'full path to URDF', default = "")
    parser.add_argument('--coll_yaml_path', '-coll', type=str,\
                        help = 'full path to collision YAML', default = "")

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
                    default = "repair_codesign_opt_l2")

    args = parser.parse_args()

    is_second_lev_opt = True

    # number of parallel processes on which to run optimization
    # set to number of cpu counts to saturate
    processes_n = mp.cpu_count()

    # useful paths
    dump_folder_name = args.dump_dir_name
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package

    urdf_full_path = args.urdf_full_path
    codesign_path = rospackage.get_path("repair_codesign")

    load_path = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dirname + "/" + args.load_dir_name
    dump_basepath = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dirname + "/" + dump_folder_name

    coll_yaml_path = args.coll_yaml_path
    solution_base_name = args.solution_base_name

    # loading solution and extracting data
    sol_loader = LoadSols(load_path)
    n_opt_sol = len(sol_loader.opt_data)

    opt_costs = [1e6] * n_opt_sol
    opt_full_q = [None] * n_opt_sol
    opt_full_q_dot = [None] * n_opt_sol
    real_sol_index = [None] * n_opt_sol

    for i in range(n_opt_sol):

        opt_full_q[i] = sol_loader.opt_data[i]["q"]
        opt_full_q_dot[i] = sol_loader.opt_data[i]["q_dot"]
        opt_costs[i] = sol_loader.opt_data[i]["opt_cost"][0][0] # [0][0] because MatStorer loads matrices by default
        real_sol_index[i] = sol_loader.opt_data[i]["multistart_index"][0][0] # unique solution index (LoadSols loads solutions with random order)
        # so a unique identifier is necessary --> using the index saved when solutions are dumped

    opt_q_design = extract_q_design(opt_full_q)

    n_d_variables = np.shape(opt_q_design)[0]

    design_var_map = get_design_map()
    design_var_names = list(design_var_map.keys())
    
    opt_index = np.argwhere(np.array(opt_costs) == min(np.array(opt_costs)))[0][0]
    opt_sol_index = sol_loader.opt_data[opt_index]["multistart_index"][0][0] # [0][0] because MatStorer loads matrices by default

    n_int = len(opt_full_q_dot[0][0, :]) # getting number of intervals of a single optimization task

    clusterer = Clusterer(opt_q_design.T, opt_costs, n_int, n_clusters = args.n_clust)

    first_lev_cand_inds = clusterer.compute_first_level_candidates()
    fist_lev_cand_man_measure = clusterer.get_fist_lev_candidate_man_measure()
    fist_lev_cand_opt_costs = clusterer.get_fist_lev_candidate_opt_cost()

    n_clust = clusterer.get_n_clust()

    # unique id used for generation of results
    unique_id = sol_loader.task_info_data["unique_id"][0]

    # task-specific options
    right_arm_picks = sol_loader.task_info_data["right_arm_picks"][0][0]
    filling_n_nodes = sol_loader.task_info_data["filling_nodes"][0][0]
    rot_error_epsi = sol_loader.task_info_data["rot_error_epsi"][0][0]
    # samples
    n_y_samples = sol_loader.task_info_data["n_y_samples"][0]
    y_sampl_ub = sol_loader.task_info_data["y_sampl_ub"][0]
    
    # chosen task
    is_in_place_flip = bool(sol_loader.task_info_data["is_in_place_flip"][0][0])
    is_biman_pick = bool(sol_loader.task_info_data["is_biman_pick"][0][0])

    # number of solution tries with different (random) initializations
    n_msrt_trgt = args.n_msrt_trgt

    # solver options
    solver_type = sol_loader.task_info_data["solver_type"][0]

    slvr_opt = {
            "ipopt.tol": sol_loader.task_info_data["slvr_opts"]["ipopt.tol"][0][0][0][0], 
            "ipopt.max_iter": sol_loader.task_info_data["slvr_opts"]["ipopt.max_iter"][0][0][0][0],
            "ipopt.constr_viol_tol": sol_loader.task_info_data["slvr_opts"]["ipopt.constr_viol_tol"][0][0][0][0],
            "ipopt.print_level": args.ipopt_verbose,\
            "ilqr.verbose": bool(sol_loader.task_info_data["slvr_opts"]["ilqr.verbose"][0][0][0][0]), 
            "ipopt.linear_solver": sol_loader.task_info_data["slvr_opts"]["ipopt.linear_solver"][0][0][0]}

    full_file_paths = None # not used

    # seed used for random number generation
    ig_seed = args.ig_seed

    # single task execution time
    t_exec_task = sol_loader.task_info_data["t_exec_task"][0][0]

    # transcription options (if used)
    transcription_method = sol_loader.task_info_data["transcription_method"][0]
    intgrtr = sol_loader.task_info_data["integrator"][0]
    transcription_opts = dict(integrator = intgrtr)

    sliding_wrist_offset = sol_loader.task_info_data["sliding_wrist_offset"][0][0]
    is_sliding_wrist = sol_loader.task_info_data["is_sliding_wrist"][0][0]

    max_retry_n = args.max_trials_factor - 1
    max_ig_trials = n_msrt_trgt * args.max_trials_factor
    proc_sol_divs = compute_solution_divs(n_msrt_trgt, processes_n)

    print(colored("Distribution of multistarts between processes: \n", "magenta"))
    print(proc_sol_divs)
    print("\n")

    if  (not os.path.isdir(dump_basepath)):

        os.makedirs(dump_basepath)

    clust_path = [""] * n_clust
    opt_path = [""] * n_clust
    fail_path = [""] * n_clust
    for i in range(n_clust):
        
        clust_path[i] = dump_basepath + "/clust" + str(i)
        opt_path[i] = clust_path[i] + "/opt"
        fail_path[i] = clust_path[i] + "/failed"

        if not os.path.isdir(opt_path[i]):
            os.makedirs(opt_path[i])
        else:
            os.rmdir(opt_path[i])
            os.makedirs(opt_path[i])
        if not os.path.isdir(fail_path[i]):
            os.makedirs(fail_path[i])
        else:
            os.rmdir(fail_path[i])
            os.makedirs(fail_path[i])


    use_classical_man = bool(sol_loader.task_info_data["use_classical_man"][0][0])
    weight_global_manip = sol_loader.task_info_data["w_man_base"][0][0]
    weight_class_manip = sol_loader.task_info_data["w_clman_base"][0][0]

    task_copies = [None] * len(proc_sol_divs)
    slvr_copies = [None] * len(proc_sol_divs)

    for p in range(len(proc_sol_divs)):
        
        print(colored("Generating task copy for process n." + str(p), "magenta"))

        task_copies[p] = gen_task_copies(weight_global_manip,
                                        weight_class_manip,
                                        filling_n_nodes,
                                        sliding_wrist_offset, 
                                        n_y_samples, y_sampl_ub,
                                        urdf_full_path,
                                        t_exec_task,
                                        rot_error_epsi,
                                        use_classical_man,
                                        is_sliding_wrist,
                                        coll_yaml_path,
                                        is_second_lev_opt, 
                                        is_in_place_flip, 
                                        is_biman_pick)
        
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
    
    real_first_level_cand_inds = [-1] * n_clust
    first_level_q_design_opt = np.zeros((len(opt_q_design[:, 0]), n_clust))

    for cl in range(n_clust): # for each cluster

        first_level_q_design_opt[:, cl] = opt_q_design[:, first_lev_cand_inds[cl]]
        real_first_level_cand_inds[cl] = real_sol_index[first_lev_cand_inds[cl]]

    # inizialize a dumper object for post-processing

    task_info_dumper = SolDumper()

    other_stuff = {"dt": task_copies[0].dt, "filling_nodes": task_copies[0].filling_n_nodes, 
                    "task_base_nnodes": task_copies[0].task_base_n_nodes_dict,
                    "right_arm_picks": task_copies[0].rght_arm_picks, 
                    "rot_error_epsi": rot_error_epsi,
                    "t_exec_task": t_exec_task,
                    "w_man_base": weight_global_manip, 
                    "w_clman_base": weight_class_manip,
                    "w_man_actual": task_copies[0].weight_glob_man, 
                    "w_clman_actual": task_copies[0].weight_classical_man,
                    "use_classical_man": use_classical_man,
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
                    "n_clust": n_clust,
                    "is_in_place_flip": is_in_place_flip, 
                    "is_biman_pick": is_biman_pick,
                    "l2_cl_cand_inds": np.array(real_first_level_cand_inds), 
                    "l2_cl_best_candidates": first_level_q_design_opt,
                    "l2_cl_opt_costs": fist_lev_cand_opt_costs, 
                    "l2_cl_cand_man_measure": fist_lev_cand_man_measure
                    }

    task_info_dumper.add_storer(other_stuff, dump_basepath,\
                            "second_level_info_t" + unique_id,\
                            False)

    task_info_dumper.dump()

    proc_list = [None] * len(proc_sol_divs)
    # launch solvers and solution dumpers on separate processes

    for cl in range(n_clust): # for each cluster (clusters are solved sequenctially)

        for p in range(len(proc_sol_divs)): # for each process (for each cluster, multistarts are solved parallelizing on processes)

            proc_list[p] = mp.Process(target=sol_main, args=(proc_sol_divs[p],\
                                                                q_ig, q_dot_ig, task_copies[p], slvr_copies[p],\
                                                                opt_path[cl], fail_path[cl],\
                                                                unique_id,\
                                                                p,\
                                                                cl,\
                                                                real_first_level_cand_inds[cl],\
                                                                first_level_q_design_opt[:, cl], \
                                                                n_msrt_trgt, 
                                                                max_retry_n,))
            proc_list[p].start()
        
        for p in range(len(proc_sol_divs)):

            while proc_list[p].is_alive():

                continue
                    
        for p in range(len(proc_sol_divs)):
            
            print(colored("Joining process " + str(p), "magenta"))

            proc_list[p].join() # wait until all processes of cluster cl are terminated


