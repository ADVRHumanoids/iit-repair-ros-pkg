#!/usr/bin/env python3

import os, argparse

import numpy as np

import subprocess

import rospkg
  
import multiprocessing as mp

from codesign_pyutils.miscell_utils import str2bool, \
                                            extract_q_design,\
                                            compute_solution_divs
from codesign_pyutils.clustering_utils import Clusterer
from codesign_pyutils.misc_definitions import get_design_map
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.task_utils import gen_task_copies, gen_slvr_copies
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.solution_utils import solve_prb_standalone, \
                                        generate_ig            
from codesign_pyutils.post_proc_utils import PostProcS1

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
            sol_costs, cnstr_opt, cnstr_lmbd,\
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
        cnstr_lmbd[sol_index] = slvr.getCnstrLmbdSolDict()

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
    cnstr_lmbd = [None] * n_multistarts_main
    trial_idxs = [-1] * n_multistarts_main

    # adding q_codes to the initial guess
    add_l1_codes2ig(q_codes_l1, q_ig)

    solution_time = solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt, cnstr_lmbd,\
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
                        **(cnstr_lmbd[sol_index]),
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

    load_path = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dirname 
    dump_basepath = codesign_path + "/" + args.res_dir_basename + "/" + args.res_dirname + "/" + dump_folder_name

    coll_yaml_path = args.coll_yaml_path
    solution_base_name = args.solution_base_name

    # loading solution and extracting data
    postprc_s1 = PostProcS1(load_path, l1_dirname=args.load_dir_name)
    postprc_s1.clusterize(args.n_clust)

    n_opt_sol = postprc_s1._n_opt_sols
    n_int = postprc_s1._n_int 
    first_lev_cand_inds = postprc_s1._clusterer.get_l1_cl_cands_idx()
    fist_lev_cand_man_measure = postprc_s1._clusterer.get_l1_cl_cands_man_measure()
    fist_lev_cand_opt_costs = postprc_s1._clusterer.get_l1_cl_cands_opt_cost()
    n_clust = postprc_s1._clusterer.get_n_clust()
    # unique id used for generation of results
    unique_id = postprc_s1._unique_id
    # task-specific options
    right_arm_picks = postprc_s1._right_arm_picks
    filling_n_nodes = postprc_s1._filling_nnodes
    rot_error_epsi = postprc_s1._rot_error_epsi
    # samples
    n_y_samples = postprc_s1._ny_sampl
    y_sampl_ub = postprc_s1._y_sampl_ub
    # chosen task
    is_in_place_flip = postprc_s1._is_in_place_flip
    is_biman_pick = postprc_s1._is_biman_pick

    # number of solution tries with different (random) initializations
    n_msrt_trgt = args.n_msrt_trgt

    # solver options
    solver_type = postprc_s1._solver_type

    slvr_opt = {"ipopt.tol": postprc_s1._slvr_opts_tol, 
            "ipopt.max_iter": postprc_s1._slvr_opts_maxiter,
            "ipopt.constr_viol_tol": postprc_s1._slvr_opts_cnstr_viol,
            "ipopt.print_level": postprc_s1._slvr_opts_print_l,\
            "ilqr.verbose": True, 
            "ipopt.linear_solver": postprc_s1._slvr_opts_lin_solv}

    full_file_paths = None # not used

    # seed used for random number generation
    ig_seed = args.ig_seed

    # single task execution time
    t_exec_task = postprc_s1._t_exec_task

    # transcription options (if used)
    transcription_method = postprc_s1._transcription_method
    intgrtr = postprc_s1._integrator
    transcription_opts = dict(integrator = intgrtr)

    sliding_wrist_offset = postprc_s1._wrist_off
    is_sliding_wrist = postprc_s1._is_sliding_wrist

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


    use_classical_man = postprc_s1._is_class_man
    use_static_tau = postprc_s1._use_static_tau
    weight_global_manip = postprc_s1._man_w_base
    weight_class_manip = postprc_s1._class_man_w_base
    weight_static_tau = postprc_s1._static_tau_w_base

    task_copies = [None] * len(proc_sol_divs)
    slvr_copies = [None] * len(proc_sol_divs)

    for p in range(len(proc_sol_divs)):
        
        print(colored("Generating task copy for process n." + str(p), "magenta"))

        task_copies[p] = gen_task_copies(weight_global_manip,
                                        weight_class_manip,
                                        weight_static_tau,
                                        filling_n_nodes,
                                        sliding_wrist_offset, 
                                        n_y_samples, y_sampl_ub,
                                        urdf_full_path,
                                        t_exec_task,
                                        rot_error_epsi,
                                        use_classical_man,
                                        use_static_tau,
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
    first_level_q_design_opt = np.zeros((len(postprc_s1._q_design[:, 0]), n_clust))

    for cl in range(n_clust): # for each cluster

        first_level_q_design_opt[:, cl] = postprc_s1._q_design[:, first_lev_cand_inds[cl]]
        real_first_level_cand_inds[cl] = postprc_s1._ms_indxs[first_lev_cand_inds[cl]]

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
                    "s2_cl_cand_inds": np.array(real_first_level_cand_inds), 
                    "s2_cl_best_candidates": first_level_q_design_opt,
                    "s2_cl_opt_costs": fist_lev_cand_opt_costs, 
                    "s2_cl_man_measure": fist_lev_cand_man_measure, 
                    "n_int": task_copies[0].n_int
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


