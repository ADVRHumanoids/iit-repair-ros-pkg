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

from multiprocessing import Pool

def solve():


# number of parallel processes on which to run optimization
processes_n = 4

# useful paths
rospackage = rospkg.RosPack() # Only for taking the path to the leg package
urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
codesign_path = rospackage.get_path("repair_codesign")
results_path = codesign_path + "/test_results"
opt_results_path = results_path + "/opt" 
failed_results_path = results_path + "/failed"
init_folder_name = "init0"
init_opt_results_path = results_path + "/initializations/" + init_folder_name + "/opt" 
init_failed_results_path = results_path + "/initializations/" + init_folder_name + "/failed" 

file_name = os.path.splitext(os.path.basename(__file__))[0]

# task-specific options
right_arm_picks = True
filling_n_nodes = 10
rot_error_epsi = 0.0000001

# generating samples along working surface y direction
n_y_samples = 1
y_sampl_ub = 0.0
y_sampl_lb = - y_sampl_ub

if n_y_samples == 1:
    dy = 0.0
else:
    dy = (y_sampl_ub - y_sampl_lb) / (n_y_samples - 1)

y_sampling = np.array( [0.0] * n_y_samples)
for i in range(n_y_samples):
    
    y_sampling[i] = y_sampl_lb + dy * i

# number of solution tries with different (random) initializations
n_multistarts = 20

# resampler option (if used)
refinement_scale = 10

# solver options
solver_type = 'ipopt'
slvr_opt = {
    "ipopt.tol": 0.0000001, 
    "ipopt.max_iter": 1000,
    "ipopt.constr_viol_tol": 0.001,
    "ilqr.verbose": True}

full_file_paths = None # not used

# seed used for random number generation
ig_seed = 1

# single task execution time
t_exec_task = 6

# transcription options (if used)
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

sliding_wrist_offset = 0.0

solution_index_name = "solution_index"

def main(args):

    sliding_wrist_command = "is_sliding_wrist:=" + "true"
    show_softhand_command = "show_softhand:=" + "true"

    # preliminary ops
    if args.gen_urdf:

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
    
    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)
    
    # some initializations
    q_ig_main = [None] * n_multistarts
    q_dot_ig_main = [None] * n_multistarts

    q_ig_init = [None] * n_multistarts # not used
    q_dot_ig_init = [None] * n_multistarts # not used

    slvr = None
    init_slvr = None # not used

    task = None
    task_init = None # not used

    ## Main problem ## 

    # initialize main problem task
    task = TaskGen(filling_n_nodes = filling_n_nodes, \
                                    sliding_wrist_offset = sliding_wrist_offset)
    object_q = np.array([1, 0, 0, 0])

    # add tasks to the task holder object
    next_node = 0 # used to place the next task on the right problem nodes
    # for i in range(len(y_sampling)):

    #     next_node = task.add_in_place_flip_task(init_node = next_node,\
    #                     object_pos_wrt_ws = np.array([0.0, y_sampling[i], 0.0]), \
    #                     object_q_wrt_ws = object_q, \
    #                     pick_q_wrt_ws = object_q,\
    #                     right_arm_picks = right_arm_picks)

    for j in range(len(y_sampling)):

        next_node = task.add_bimanual_task(init_node = next_node,\
                        object_pos_wrt_ws = np.array([0.0, y_sampling[j], 0.0]), \
                        object_q_wrt_ws = object_q, \
                        pick_q_wrt_ws = object_q,\
                        right_arm_picks = right_arm_picks)

    # initialize problem
    task.init_prb(urdf_full_path, args.base_weight_pos, args.base_weight_rot,\
                            args.weight_global_manip, args.weight_class_manip,\
                            is_soft_pose_cnstr = args.soft_pose_cnstrnt,\
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

    # clear generated urdf file
    if exists(urdf_full_path): 

        os.remove(urdf_full_path)
    
    # inizialize a dumper object for post-processing
    if args.dump_sol: 

        sol_dumper = SolDumper()

    # generating initial guesses, based on the script arguments
    q_ig_main, q_dot_ig_main =  generate_ig(args, full_file_paths,\
                                            task,\
                                            n_multistarts, ig_seed,\
                                            False)

    # some initializations before entering the solution loop
    solve_failed_array = [True] * n_multistarts
    sol_costs = [1e10] * n_multistarts
    solutions = [None] * n_multistarts
    cnstr_opt = [None] * n_multistarts

    # solving multiple problems on separate processes

    for i in range(n_multistarts):

        print("\n SOLVING PROBLEM N.: ", i + 1)
        print("\n")
                    
        solve_failed = solve_prb_standalone(task, slvr, q_ig_main[i], q_dot_ig_main[i])
        solutions[i] = slvr.getSolutionDict()

        print("Solution cost " + str(i) + ": ", solutions[i]["opt_cost"])
        sol_costs[i] = solutions[i]["opt_cost"]
        cnstr_opt[i] = slvr.getConstraintSolutionDict()

        solve_failed_array[i] = solve_failed

    # solutions packaging for postprocessing
    n_opt_sol = len(np.where(np.array(solve_failed_array)== False)[0])

    best_index = get_min_cost_index(sol_costs, solve_failed_array)

    other_stuff = {"dt": task.dt, "filling_nodes": task.filling_n_nodes,
                    "task_base_nnodes": task.task_base_n_nodes_dict,
                    "right_arm_picks": task.rght_arm_picks, \
                    "wman_base": args.weight_global_manip, \
                    "wpo_bases": args.base_weight_pos, "wrot_base": args.base_weight_rot, \
                    "wman_actual": task.weight_glob_man, \
                    "wpos_actual": task.weight_pos, "wrot_actual": task.weight_rot, 
                    "solve_failed": solve_failed_array, 
                    "n_opt_sol": n_opt_sol, "n_unfeas_sol": n_multistarts - n_opt_sol,
                    "sol_costs": sol_costs, "best_sol_index": best_index,
                    "nodes_list": task.nodes_list, 
                    "tasks_list": task.task_list,
                    "tasks_dict": task.task_dict}
    
    sol_dumper.add_storer(other_stuff, results_path,\
                                            "additional_info_main", True)

    for i in range(n_multistarts):
            
        full_solution = {**(solutions[i]),
                        **(cnstr_opt[i]),
                        **{"q_ig": q_ig_main[i], "q_dot_ig": q_dot_ig_main[i]}, \
                        **{"solution_index": i}}

        if not solve_failed_array[i]:

            sol_dumper.add_storer(full_solution, opt_results_path,\
                            "flipping_repair" + str(i), True)
        else:

            sol_dumper.add_storer(full_solution, failed_results_path,\
                            "flipping_repair" + str(i), True)

        if i == (n_multistarts - 1): # dump solution after last pass

                sol_dumper.dump() 

                print("\nSolutions dumped. \n")
                    
    exit()
    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--gen_urdf', '-g', type=str2bool,\
                        help = 'whether to generate urdf from xacro', default = True)
    parser.add_argument('--launch_rviz', '-rvz', type=str2bool,\
                        help = 'whether to launch rviz or not', default = True)
    parser.add_argument('--rviz_replay', '-rpl', type=str2bool,\
                        help = 'whether to replay the solution on RViz', default = True)
    parser.add_argument('--dump_sol', '-ds', type=str2bool,\
                        help = 'whether to dump results to file', default = True)
    parser.add_argument('--use_init_guess', '-ig', type=str2bool,\
                        help = 'whether to use initial guesses between solution loops', default = True)
    parser.add_argument('--soft_pose_cnstrnt', '-spc', type=str2bool,\
                        help = 'whether to use soft pose constraints or not', default = False)
    parser.add_argument('--base_weight_pos', '-wp', type = np.double,\
                        help = 'base weight for position tracking (if using soft constraints)', default = 0.001)
    parser.add_argument('--base_weight_rot', '-wr', type = np.double,\
                        help = 'base weight for orientation tracking (if using soft constraints)', default = 0.001)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 0.01)
    parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
                        help = 'whether to load the initial guess from file', default = False)
    parser.add_argument('--use_classical_man', '-ucm', type=str2bool,\
                        help = 'whether to use the classical manipulability index', default = False)

    args = parser.parse_args()

    main(args)