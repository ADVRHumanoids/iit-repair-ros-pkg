import numpy as np

from codesign_pyutils.tasks import CodesTaskGen, HighClManGen

from codesign_pyutils.miscell_utils import gen_y_sampling

from codesign_pyutils.solution_utils import solve_prb_standalone

from termcolor import colored

from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver

def gen_task_copies(weight_global_manip: np.double, weight_class_manip: np.double, 
                    weight_static_tau: np.double,
                    filling_nodes: int,
                    wrist_offset: np.double, 
                    y_samples: list, y_ub: list, 
                    urdf_path: str,
                    t_exec: np.double,
                    rot_err_epsi: np.double,
                    use_classical_man = False,
                    use_static_tau = False, 
                    sliding_wrist = False, 
                    coll_path = "", 
                    cost_weight_path = "",
                    is_second_lev_opt = False, 
                    is_in_place_flip = True, 
                    is_bimanual_pick = False):

    y_sampling = [None] * 2

    y_sampling[0] = gen_y_sampling(y_samples[0], y_ub[0])
    y_sampling[1] = gen_y_sampling(y_samples[1], y_ub[1])

    right_arm_picks = [True] * len(y_sampling[0])

    for i in range(len(y_sampling[0])): # only necessary for the pick and place task
        
        if y_sampling[0][i] <= 0 : # on the right
            
            right_arm_picks[i] = True
        
        else:

            right_arm_picks[i] = False

    # initialize problem task
    task = CodesTaskGen(filling_n_nodes = filling_nodes, \
                    is_sliding_wrist = sliding_wrist,\
                    sliding_wrist_offset = wrist_offset,\
                    coll_yaml_path = coll_path, 
                    cost_weights_yaml_path = cost_weight_path)

    task.add_tasks(y_sampling, right_arm_picks, 
                    is_in_place_flip = is_in_place_flip, 
                    is_bimanual_pick = is_bimanual_pick)

    # initialize problem
    task.init_prb(urdf_path,
                weight_glob_man = weight_global_manip, weight_class_man = weight_class_manip,\
                weight_static_tau = weight_static_tau,
                tf_single_task = t_exec)

    print(colored("Task node list: " + str(task.nodes_list), "magenta"))
    print(colored("Task list: " + str(task.task_list), "magenta"))
    print(colored("Task names: " + str(task.task_names), "magenta"))
    print(colored("Task dict: " + str( task.task_dict), "magenta"))
    print(colored("Total employed nodes: " + str(task.total_nnodes), "magenta"))
    print(colored("Number of added subtasks:" + str(task.n_of_tasks) + "\n", "magenta"))

    # set constraints and costs
    task.setup_prb(rot_err_epsi, is_classical_man = use_classical_man,
                    is_second_lev_opt=is_second_lev_opt, 
                    is_static_tau = use_static_tau)

    return task

def gen_cl_man_gen_copies(
                    wrist_offset: np.double, 
                    urdf_path: str,
                    t_exec: np.double,
                    coll_path = "", 
                    weight_class_manip = 1):

    # initialize problem task
    task = HighClManGen(
                    sliding_wrist_offset = wrist_offset,\
                    coll_yaml_path = coll_path)

    # initialize problem
    task.init_prb(urdf_path,
                weight_class_manip,
                tf_single_task = t_exec)

    # set constraints and costs
    task.setup_prb()

    return task

def gen_slvr_copies(task: CodesTaskGen,
                    solver_type: str,
                    transcription_method: str, 
                    transcription_opts: dict, 
                    slvr_opt: dict):

    if solver_type != "ilqr":

        Transcriptor.make_method(transcription_method,\
                                task.prb,\
                                transcription_opts)
    
    ## Creating the solver
    slvr = solver.Solver.make_solver(solver_type, task.prb, slvr_opt)

    if solver_type == "ilqr":

        slvr.set_iteration_callback()

    return slvr

def compute_node_cl_man(task: CodesTaskGen, q: np.ndarray):

    Jl = task.jac_arm_l( q = q )["J"]
    Jr = task.jac_arm_r( q = q )["J"]
    cl_man_ltrasl, cl_man_lrot, cl_man_ltot = task.compute_cl_man(Jl)
    cl_man_rtrasl, cl_man_rrot, cl_man_rtot = task.compute_cl_man(Jr)

    return cl_man_ltrasl, cl_man_lrot, cl_man_ltot, \
            cl_man_rtrasl, cl_man_rrot, cl_man_rtot   

def compute_ms_cl_man(q: np.ndarray, nodes_list: list, task: CodesTaskGen):
    
    ms_cl_man_lft = np.zeros((2, nodes_list[-1][-1] + 1))
    ms_cl_man_rght = np.zeros((2, nodes_list[-1][-1] + 1))

    for task_idx in range(len(nodes_list)):
    
        for node_idx in range(len(nodes_list[task_idx])):
            
            cl_man_ltrasl, cl_man_lrot, _1, \
                cl_man_rtrasl, cl_man_rrot, _2 = compute_node_cl_man(task, q[:, node_idx])

            ms_cl_man_lft[0, nodes_list[task_idx][node_idx]] = cl_man_ltrasl
            ms_cl_man_lft[1, nodes_list[task_idx][node_idx]] = cl_man_lrot
            ms_cl_man_rght[0, nodes_list[task_idx][node_idx]] = cl_man_rtrasl
            ms_cl_man_rght[1, nodes_list[task_idx][node_idx]] = cl_man_rrot
    
    return ms_cl_man_lft, ms_cl_man_rght


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

def solve_s1(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_tot_cost, sol_costs, cnstr_opt, cnstr_lmbd,\
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

        print_color = "green" if not solve_failed else "yellow"
        print(colored("COMLETED SOLUTION PROCEDURE OF MULTISTART NODE:" + str(multistart_nodes[sol_index]) + \
            ".\nProcess n." + str(process_id) + \
            ".\nIn-process index: " + str(sol_index + 1) + \
            "/" + str(len(multistart_nodes)) + \
            ".\nOpt. cost: " + str(solutions[sol_index]["opt_cost"]), print_color))

        sol_tot_cost[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()
        cnstr_lmbd[sol_index] = slvr.getCnstrLmbdSolDict()

        # also add separate cost values to the dumped data
        cost_dict = task.prb.getCosts()
        sol_cost_dict = {}

        for cost_fnct_key, cost_fnct_val in cost_dict.items():

            sol_cost_dict[cost_fnct_key] = task.prb.evalFun(cost_fnct_val, solutions[sol_index])

        sol_costs[sol_index] = sol_cost_dict

        solve_failed_array[sol_index] = solve_failed

    return solution_time

def sol_main_s1(args, multistart_nodes, q_ig, q_dot_ig, task, slvr, result_path, opt_path, fail_path,\
        id_unique,\
        process_id, \
        n_multistarts, 
        max_retry_n):
    
    n_multistarts_main = len(multistart_nodes) # number of multistarts assigned to this main instance

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_tot_cost = [1e10] * n_multistarts_main
    sol_costs = [None] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main
    cnstr_lmbd = [None] * n_multistarts_main
    trial_idxs = [-1] * n_multistarts_main

    solution_time = solve_s1(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_tot_cost, sol_costs, cnstr_opt, cnstr_lmbd,\
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
                        **(cnstr_lmbd[sol_index]),
                        **(sol_costs[sol_index]),
                        "q_ig": q_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]],
                        "q_dot_ig": q_dot_ig[multistart_nodes[sol_index] + n_multistarts * trial_idxs[sol_index]], \
                        "multistart_index": multistart_nodes[sol_index], 
                        "trial_index": trial_idxs[sol_index], 
                        "solution_time": solution_time, 
                        "solve_failed": solve_failed_array[sol_index], 
                        "run_id": id_unique}

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

def solve_s3(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_tot_cost, sol_costs, cnstr_opt, cnstr_lmbd,\
            solve_failed_array, 
            q_codes_s1, 
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
                                is_second_step_opt= True, \
                                q_codes_s1=q_codes_s1)
            
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

        sol_tot_cost[sol_index] = solutions[sol_index]["opt_cost"]
        cnstr_opt[sol_index] = slvr.getConstraintSolutionDict()
        cnstr_lmbd[sol_index] = slvr.getCnstrLmbdSolDict()

        # also add separate cost values to the dumped data
        cost_dict = task.prb.getCosts()
        sol_cost_dict = {}

        for cost_fnct_key, cost_fnct_val in cost_dict.items():

            sol_cost_dict[cost_fnct_key] = task.prb.evalFun(cost_fnct_val, solutions[sol_index])

        sol_costs[sol_index] = sol_cost_dict
        
        solve_failed_array[sol_index] = solve_failed # for now, solve_failed will always be true
    
    return solution_time
    
def sol_main_s3(multistart_nodes, q_ig, q_dot_ig, task, slvr, opt_path, fail_path,\
        id_unique,\
        process_id,
        cluster_id,
        first_lev_sol_id, 
        q_codes_s1, 
        n_multistarts, 
        max_retry_n):
    
    n_multistarts_main = len(multistart_nodes) # number of multistarts assigned to this main instance

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts_main
    sol_tot_cost = [1e10] * n_multistarts_main
    sol_costs = [None] * n_multistarts_main
    solutions = [None] * n_multistarts_main
    cnstr_opt = [None] * n_multistarts_main
    cnstr_lmbd = [None] * n_multistarts_main
    trial_idxs = [-1] * n_multistarts_main

    # adding q_codes to the initial guess
    add_s1_codes2ig(q_codes_s1, q_ig)

    solution_time = solve(multistart_nodes,\
            task, slvr,\
            q_ig, q_dot_ig,\
            solutions,\
            sol_tot_cost, sol_costs, cnstr_opt, cnstr_lmbd,\
            solve_failed_array, 
            q_codes_s1, 
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
                        **(sol_costs[sol_index]),
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
  