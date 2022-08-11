#!/usr/bin/env python3

from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver
import os, argparse

import numpy as np

import subprocess

import rospkg

from codesign_pyutils.miscell_utils import str2bool
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.task_utils import solve_prb_standalone, \
                                        generate_ig              
from codesign_pyutils.tasks import TaskGen

import multiprocessing as mp_classic

from codesign_pyutils.miscell_utils import extract_q_design, compute_man_measure,\
                                             compute_solution_divs, gen_y_sampling
from codesign_pyutils.miscell_utils import Clusterer
from codesign_pyutils.misc_definitions import get_design_map
from codesign_pyutils.load_utils import LoadSols

def solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array, 
            q_codes_first_level, 
            cluster_id):

    sol_index = 0 # different index

    solution_time = -1.0

    for node in multistart_nodes:

        print("\n SOLVING PROBLEM N.: " + str(node + 1) + ", cluster n." + str(cluster_id))
        print("\n")

        solve_failed, solution_time = solve_prb_standalone(task, slvr, q_ig[node], q_dot_ig[node], 
                                                            is_second_level_opt= True, 
                                                            q_codes_first_level=q_codes_first_level)
        solutions[sol_index] = slvr.getSolutionDict()

        print("Solution cost " + str(node) + ": ", solutions[sol_index]["opt_cost"])
        sol_costs[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()

        solve_failed_array[sol_index] = solve_failed

        sol_index = sol_index + 1
    
    return solution_time

def gen_task_copies(weight_global_manip, weight_class_manip, 
                    filling_n_nodes, 
                    n_y_samples, y_sampl_ub, 
                    use_classical_man = False, 
                    coll_yaml_path = ""):

    
    y_sampling = gen_y_sampling(n_y_samples, y_sampl_ub)

    right_arm_picks = np.array([True] * len(y_sampling))
    for i in range(len(y_sampling)):
        
        if y_sampling[i] <= 0 : # on the right
            
            right_arm_picks[i] = True
        
        else:

            right_arm_picks[i] = False

    # initialize problem task
    task = TaskGen(filling_n_nodes = filling_n_nodes, 
                    coll_yaml_path = coll_yaml_path)

    task.add_tasks(y_sampling, right_arm_picks)

    # initialize problem
    task.init_prb(urdf_full_path,
                    weight_glob_man = weight_global_manip, weight_class_man = weight_class_manip,\
                    tf_single_task = t_exec_task)

    print("Task node list: ", task.nodes_list)
    print("Task list: ", task.task_list)
    print("Task names: ", task.task_names)
    print("Task dict: ", task.task_dict)
    print("Total employed nodes: ", task.total_nnodes)
    print("Number of added subtasks:", task.n_of_tasks, "\n")

    # set constraints and costs
    task.setup_prb(rot_error_epsi, is_classical_man = use_classical_man, 
                    is_second_lev_opt = True)

    if solver_type != "ilqr":

        Transcriptor.make_method(transcription_method,\
                                task.prb,\
                                transcription_opts)
    
    ## Creating the solver
    slvr = solver.Solver.make_solver(solver_type, task.prb, slvr_opt)

    if solver_type == "ilqr":

        slvr.set_iteration_callback()


    return task, slvr

def sol_main(multistart_nodes, q_ig, q_dot_ig, task, slvr, opt_path, fail_path,\
        id_unique,\
        process_id,
        cluster_id,
        first_lev_sol_id, 
        q_codes_first_level):
    
    n_multistarts_main = len(multistart_nodes)

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_costs = [1e10] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main

    # adding q_codes to the initial guess
    design_var_map = get_design_map()

    design_indeces = [design_var_map["mount_h"],\
        design_var_map["should_w_l"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"],\
        design_var_map["should_w_r"],\
        design_var_map["should_roll_r"],\
        design_var_map["wrist_off_r"]]

    q_codes_first_level_extended = np.concatenate((q_codes_first_level, q_codes_first_level[1:]), axis=0)

    # print(np.tile(q_codes_first_level_extended, len(q_ig[0][0, :])))
    for i in range(len(q_ig)):

        q_ig[i][design_indeces, :] = np.transpose(np.tile(q_codes_first_level_extended, (len(q_ig[0][0, :]), 1)))

    solution_time = solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_costs, cnstr_opt,\
            solve_failed_array, 
            q_codes_first_level, 
            cluster_id)

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
                        "cluster_id": cluster_id, 
                        "first_lev_sol_id": first_lev_sol_id, 
                        "solve_failed": solve_failed_array[sol_index]}

        if not solve_failed_array[sol_index]:

            sol_dumper.add_storer(full_solution, opt_path,\
                            solution_base_name + \
                            "_id" + str(first_lev_sol_id) + \
                            "_cl" + str(cluster_id) + \
                            "_p" + str(process_id) + \
                            "_n" + str(node) + \
                            "_t" + \
                             id_unique, False)
        else:

            sol_dumper.add_storer(full_solution, fail_path,\
                            solution_base_name + \
                            "_id" + str(first_lev_sol_id) + \
                            "_cl" + str(cluster_id) + \
                            "_p" + str(process_id) + \
                            "_n" + str(node) + \
                            "_t" + \
                             id_unique, False)

        sol_index = sol_index + 1

    sol_dumper.dump() 

    print("\n Solutions of process " + str(process_id) + " ; cluster n. " + str(cluster_id) + " dumped. \n")
                    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='Second level optimization script for the co-design of RePAIR project')
    parser.add_argument('--n_multistarts', '-msn', type=int,\
                        help = 'number of multistarts (per cluster) to use', default = 2)
    parser.add_argument('--ig_seed', '-igs', type=int,\
                        help = 'seed for random initialization generation', default = 1)                      
    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = False)
    parser.add_argument('--res_dirname', '-d', type=str,\
                        help = 'directory name from where results are to be loaded', default = "load_dir")
    parser.add_argument('--n_clust', '-nc', type=int,\
                        help = 'number of clusters to be selected', default = 40)
    parser.add_argument('--ipopt_verbose', '-ipopt_v', type = int,\
                        help = 'IPOPT verbose flag', default = 4)

    args = parser.parse_args()


    # number of parallel processes on which to run optimization
    # set to number of cpu counts to saturate
    processes_n = mp_classic.cpu_count()

    # useful paths
    dump_folder_name = "second_level"
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
    urdf_name = "repair_full"
    urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
    xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
    codesign_path = rospackage.get_path("repair_codesign")

    load_path = codesign_path + "/test_results/" + args.res_dirname + "/first_level"
    dump_basepath = codesign_path + "/test_results/" + args.res_dirname + "/" + dump_folder_name

    coll_yaml_name = "arm_coll.yaml"
    coll_yaml_path = rospackage.get_path("repair_urdf") + "/config/" + coll_yaml_name

    solution_base_name = "repair_codesign_opt_l2"

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
        real_sol_index[i] = sol_loader.opt_data[i]["solution_index"][0][0] # unique solution index (LoadSols loads solutions with random order)
        # so a unique identifier is necessary --> using the index saved when solutions are dumped

    opt_q_design = extract_q_design(opt_full_q)

    n_d_variables = np.shape(opt_q_design)[0]

    design_var_map = get_design_map()
    design_var_names = list(design_var_map.keys())
    
    opt_index = np.argwhere(np.array(opt_costs) == min(np.array(opt_costs)))[0][0]
    opt_sol_index = sol_loader.opt_data[opt_index]["solution_index"][0][0] # [0][0] because MatStorer loads matrices by default

    n_int = len(opt_full_q_dot[0][0, :]) # getting number of intervals of a single optimization task
    man_measure = compute_man_measure(opt_costs, n_int) # scaling opt costs to make them more interpretable

    clusterer = Clusterer(opt_q_design.T, opt_costs, n_int, n_clusters = args.n_clust)

    clusterer.clusterize()

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
    n_y_samples = sol_loader.task_info_data["n_y_samples"][0][0]
    y_sampl_ub = sol_loader.task_info_data["y_sampl_ub"][0][0]

    # number of solution tries with different (random) initializations
    n_multistarts = args.n_multistarts

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

    proc_sol_divs = compute_solution_divs(n_multistarts, processes_n)

    print("Distribution of multistarts between processes: \n")
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
        
        task_copies[p], slvr_copies[p] = gen_task_copies(weight_global_manip, 
                                                        weight_class_manip, 
                                                        filling_n_nodes, 
                                                        n_y_samples, y_sampl_ub, 
                                                        use_classical_man, 
                                                        coll_yaml_path)
    
    # some initializations
    q_ig = [None] * n_multistarts
    q_dot_ig = [None] * n_multistarts

    # generating initial guesses, based on the script arguments
    q_ig, q_dot_ig =  generate_ig(args, full_file_paths,\
                                    task_copies[0],\
                                    n_multistarts, ig_seed,\
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
                    "w_man_base": weight_global_manip, 
                    "w_clman_base": weight_class_manip,
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
                    "n_multistarts_per_cl": n_multistarts,
                    # "proc_sol_divs": proc_sol_divs,
                    "unique_id": unique_id,
                    "n_clust": n_clust,
                    "first_lev_cand_inds": np.array(real_first_level_cand_inds), 
                    "first_lev_best_candidates": first_level_q_design_opt,
                    "first_lev_opt_costs": fist_lev_cand_opt_costs, 
                    "fist_lev_cand_man_measure": fist_lev_cand_man_measure}

    task_info_dumper.add_storer(other_stuff, dump_basepath,\
                            "second_level_info_t" + unique_id,\
                            False)
    # dumping copies in each cluster folder (this is so that the same SolLoader class can be used for post-proc. of second lev. solutions)
    
    for i in range(n_clust):

        task_info_dumper.add_storer(other_stuff, clust_path[i],\
                            "second_level_info_t" + unique_id,\
                            False)

    task_info_dumper.dump()

    proc_list = [None] * len(proc_sol_divs)
    # launch solvers and solution dumpers on separate processes

    for cl in range(n_clust): # for each cluster

        for p in range(len(proc_sol_divs)): # for each process

            proc_list[p] = mp_classic.Process(target=sol_main, args=(proc_sol_divs[p],\
                                                                q_ig, q_dot_ig, task_copies[p], slvr_copies[p],\
                                                                opt_path[cl], fail_path[cl],\
                                                                unique_id,\
                                                                p,\
                                                                cl,\
                                                                real_first_level_cand_inds[cl],\
                                                                first_level_q_design_opt[:, cl]))
            proc_list[p].start()
        
        for p in range(len(proc_sol_divs)):

            while proc_list[p].is_alive():

                continue
                    
        for p in range(len(proc_sol_divs)):
            
            print("Joining process " + str(p))

            proc_list[p].join() # wait until all processes of cluster cl are terminated


