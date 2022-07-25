#!/usr/bin/env python3

from horizon.utils.resampler_trajectory import resampler
from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver
import os, argparse
from os.path import exists

import numpy as np

import subprocess

import rospkg

from codesign_pyutils.ros_utils import FramePub, ReplaySol
from codesign_pyutils.miscell_utils import str2bool,\
                                        wait_for_confirmation,\
                                        get_min_cost_index
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.task_utils import do_one_solve_pass, \
                                        generate_ig              
from codesign_pyutils.tasks import TaskGen

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
filling_n_nodes = 0
rot_error_epsi = 0.0000001

# generating samples along working surface y direction
n_y_samples = 5
y_sampl_ub = 0.4
y_sampl_lb = - y_sampl_ub

if n_y_samples == 1:
    dy = 0.0
else:
    dy = (y_sampl_ub - y_sampl_lb) / (n_y_samples - 1)

y_sampling = np.array( [0.0] * n_y_samples)
for i in range(n_y_samples):
    
    y_sampling[i] = y_sampl_lb + dy * i

# number of solution tries
n_glob_tests = 40

# resampler option (if used)
refinement_scale = 10

# solver options
solver_type = 'ipopt'
slvr_opt = {
    "ipopt.tol": 0.0000001, 
    "ipopt.max_iter": 1000,
    "ipopt.constr_viol_tol": 0.001,
    "ilqr.verbose": True}

# loaded initial guess stuff

file_list_opt = os.listdir(init_opt_results_path)
file_list_failed = os.listdir(init_failed_results_path)
is_file_opt = [True] * len(file_list_opt) + [False] * len(file_list_failed)
full_file_list = file_list_opt + file_list_failed
full_file_paths = [""] * len(full_file_list)

for i in range(len(full_file_list)):

    if is_file_opt[i]:

        full_file_paths[i] = init_opt_results_path + "/" + full_file_list[i]
    
    else:

        full_file_paths[i] = init_failed_results_path + "/" + full_file_list[i]

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

    if args.launch_rviz:

        try:

            rviz_window = subprocess.Popen(["roslaunch",\
                                            "repair_urdf",\
                                            "repair_full_markers.launch", \
                                            sliding_wrist_command,\
                                            show_softhand_command])

        except:

            print('Failed to launch RViz.')
    
    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)
    
    # some initializations
    q_ig_main = [None] * n_glob_tests
    q_dot_ig_main = [None] * n_glob_tests

    q_ig_init = [None] * n_glob_tests
    q_dot_ig_init = [None] * n_glob_tests

    slvr = None
    init_slvr = None

    flipping_task = None
    flipping_task_init = None

    ## Main problem ## 

    # initialize main problem task
    flipping_task = TaskGen(filling_n_nodes = filling_n_nodes, \
                                    sliding_wrist_offset = sliding_wrist_offset)
    object_q = np.array([1, 0, 0, 0])

    # add tasks to the task holder object
    next_node = 0 # used to place the next task on the right problem nodes
    for i in range(len(y_sampling)):

        next_node = flipping_task.add_in_place_flip_task(init_node = next_node,\
                        object_pos_wrt_ws = np.array([0.0, y_sampling[i], 0.0]), \
                        object_q_wrt_ws = object_q, \
                        pick_q_wrt_ws = object_q,\
                        right_arm_picks = right_arm_picks)

    # initialize problem
    flipping_task.init_prb(urdf_full_path, args.base_weight_pos, args.base_weight_rot,\
                            args.weight_global_manip,\
                            is_soft_pose_cnstr = args.soft_pose_cnstrnt,\
                            tf_single_task = t_exec_task)

    print("Flipping task node list: ", flipping_task.nodes_list)
    print("Total employed nodes: ", flipping_task.total_nnodes)
    print("Number of added subtasks:", flipping_task.n_of_tasks, "\n")

    # set constraints and costs
    flipping_task.setup_prb(rot_error_epsi)

    if solver_type != "ilqr":

        Transcriptor.make_method(transcription_method,\
                                flipping_task.prb,\
                                transcription_opts)
    
    ## Creating the solver
    slvr = solver.Solver.make_solver(solver_type, flipping_task.prb, slvr_opt)

    if solver_type == "ilqr":

        slvr.set_iteration_callback()


    if args.warmstart:

        ## Initialization problem ##
        
        # initialize the initialization problem task
        flipping_task_init = TaskGen(filling_n_nodes = filling_n_nodes)
    
        # add tasks to the task holder object
        next_node = 0 # used to place the next task on the right problem nodes
        for i in range(len(y_sampling)):

            next_node = flipping_task_init.add_in_place_flip_task(init_node = next_node,\
                            object_pos_wrt_ws = np.array([0.0, y_sampling[i], 0.0]), \
                            object_q_wrt_ws = object_q, \
                            pick_q_wrt_ws = object_q,\
                            right_arm_picks = right_arm_picks)

        # initialize problem
        flipping_task_init.init_prb(urdf_full_path, args.base_weight_pos, args.base_weight_rot,\
                            args.weight_global_manip,\
                            is_soft_pose_cnstr = args.soft_warmstart,\
                            tf_single_task = t_exec_task)

        # set constraints and costs
        flipping_task_init.setup_prb(rot_error_epsi)

        if solver_type != "ilqr":

            Transcriptor.make_method(transcription_method,\
                                    flipping_task_init.prb,\
                                    transcription_opts)  # setting the transcriptor for the initialization problem

        ## Creating the solver
        init_slvr = solver.Solver.make_solver(solver_type, flipping_task_init.prb, slvr_opt)

        if solver_type == "ilqr":

            init_slvr.set_iteration_callback()
    
    # publishing picking poses frames
    pose_pub = FramePub("frame_pub")
    for i in range(len(y_sampling)):
        
        frame_name = "/repair/obj_picking_pose" + str(i)
    
        pose_pub.add_pose(flipping_task.object_pos_lft[i], flipping_task.object_q_lft[i],\
                        frame_name, "working_surface_link")

    pose_pub.spin()

    # clear generated urdf file
    if exists(urdf_full_path): 

        os.remove(urdf_full_path)
    
    # inizialize a dumper object for post-processing
    if args.dump_sol: 

        sol_dumper = SolDumper()

    # generating initial guesses, based on the script arguments
    q_ig_main, q_dot_ig_main =  generate_ig(args, full_file_paths,\
                                            flipping_task,\
                                            n_glob_tests, ig_seed,\
                                            False)
    if args.warmstart:

        q_ig_init, q_dot_ig_init =  generate_ig(args, full_file_paths,\
                                            flipping_task_init,\
                                            n_glob_tests, ig_seed,\
                                            False)

    # some initializations before entering the solution loop
    init_solve_failed_array = [True] * n_glob_tests # defaults to failed
    solve_failed_array = [True] * n_glob_tests
    sol_costs = [1e10] * n_glob_tests
    solutions = [None] * n_glob_tests
    init_solutions = [None] * n_glob_tests
    cnstr_opt = [None] * n_glob_tests
    cnstr_opt_init =[None] * n_glob_tests

    # solving multiple problems
    for i in range(n_glob_tests):

        print("\n SOLVING PROBLEM N.: ", i + 1)
        print("\n")
        # print("With initial guesses:\n")
        # print("q_ig: ", q_ig_main[i], "\n")
        # print("q_dot_ig: ", q_dot_ig_main[i], "\n")
                    
        init_sol_failed, solve_failed = do_one_solve_pass(args,\
                                            flipping_task, slvr,\
                                            q_ig_main[i], q_dot_ig_main[i],\
                                            flipping_task_init, init_slvr,\
                                            q_ig_init[i], q_dot_ig_init[i])


        solutions[i] = slvr.getSolutionDict()
        print("Solution cost " + str(i) + ": ", solutions[i]["opt_cost"])
        sol_costs[i] = solutions[i]["opt_cost"]
        cnstr_opt[i] = slvr.getConstraintSolutionDict()

        if args.warmstart and (not init_sol_failed):

            init_solutions[i] = init_slvr.getSolutionDict() # extracting solution from initialization problem
            # print("Initialization problem solution cost" + str(i) + ": ", init_solutions[i]["opt_cost"])

        init_solve_failed_array[i] = init_sol_failed

        solve_failed_array[i] = solve_failed

    n_opt_sol = len(np.where(np.array(solve_failed_array)== False)[0])

    best_index = get_min_cost_index(sol_costs, solve_failed_array)

    other_stuff = {"dt": flipping_task.dt, "filling_nodes": flipping_task.filling_n_nodes,
                    "task_base_nnodes": flipping_task.task_base_n_nodes,
                    "right_arm_picks": flipping_task.rght_arm_picks, \
                    "wman_base": args.weight_global_manip, \
                    "wpo_bases": args.base_weight_pos, "wrot_base": args.base_weight_rot, \
                    "wman_actual": flipping_task.weight_glob_man, \
                    "wpos_actual": flipping_task.weight_pos, "wrot_actual": flipping_task.weight_rot, 
                    "solve_failed": solve_failed_array, 
                    "n_opt_sol": n_opt_sol, "n_unfeas_sol": n_glob_tests - n_opt_sol,
                    "sol_costs": sol_costs, "best_sol_index": best_index}
    
    if args.dump_sol: 

        sol_dumper.add_storer(other_stuff, results_path,\
                                            "additional_info_main", True)
    if args.warmstart:
        
        n_opt_init_sol = len(np.where(np.array(init_solve_failed_array) == False))

        other_stuff_init = {"dt": flipping_task_init.dt, "filling_nodes": flipping_task_init.filling_n_nodes,
                                "task_base_nnodes": flipping_task_init.task_base_n_nodes,
                                "right_arm_picks": flipping_task_init.rght_arm_picks, \
                                "wman_base": args.weight_global_manip, \
                                "wpo_bases": args.base_weight_pos, "wrot_base": args.base_weight_rot, \
                                "wman_actual": flipping_task_init.weight_glob_man, \
                                "wpos_actual": flipping_task_init.weight_pos, "wrot_actual": flipping_task_init.weight_rot,\
                                "init_solve_failed": init_solve_failed_array, 
                                "n_opt_sol": n_opt_init_sol, "n_unfeas_sol": n_glob_tests - n_opt_init_sol}

        sol_dumper.add_storer(other_stuff_init, results_path,\
                                        "additional_info_init", True)

    if args.dump_sol:

        for i in range(n_glob_tests):

            store_current_sol = wait_for_confirmation(do_something = "store the solution n." + str(i),\
                                or_do_something_else = "avoid storing it",\
                                on_confirmation = "Adding solution n." + str(i) + " to dumped data...",\
                                on_denial = "Solution" + str(i) + "will be discarted!")

            if store_current_sol:
                
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

                if args.warmstart:

                    full_solution_init = {**(init_solutions[i]),\
                        **(cnstr_opt_init[i]),\
                        **{"q_ig": q_ig_init[i], "q_dot_ig": q_dot_ig_init[i]}, 
                        **{"init_solution_index": i}}
                    
                    if not solve_failed_array[i]:
                        
                        sol_dumper.add_storer(full_solution_init, opt_results_path,\
                                            "flipping_repair_init_prb" + str(i), True)
                    
                    else:

                        sol_dumper.add_storer(full_solution_init, failed_results_path,\
                                            "flipping_repair_init_prb" + str(i), True)

            if i == (n_glob_tests - 1): # dump solution after last pass

                    sol_dumper.dump() 

                    print("\nSolutions dumped. \n")

    if args.replay_only_best:

        if args.rviz_replay:

            q_replay = None
            q_replay_init = None

            if args.resample_sol:
                
                dt_res = flipping_task.dt / refinement_scale

                q_replay = resampler(solutions[best_index]["q"], solutions[best_index]["q_dot"],\
                                        flipping_task.dt, dt_res,\
                                        {'x': flipping_task.q, 'p': flipping_task.q_dot,\
                                        'ode': flipping_task.q_dot, 'quad': 0})

                sol_replayer = ReplaySol(dt_res,
                                            joint_list = flipping_task.joint_names,
                                            q_replay = q_replay, \
                                            srt_msg = "\nReplaying best solution ( n." + str(best_index + 1) + " /" +\
                                            str(n_glob_tests) + " )...")

            else:
                
                q_replay = solutions[best_index]["q"]

                sol_replayer = ReplaySol(dt = flipping_task.dt,\
                                            joint_list = flipping_task.joint_names,\
                                            q_replay = q_replay, \
                                            srt_msg = "\nReplaying best solution ( n." + str(best_index + 1) + " /" +\
                                            str(n_glob_tests) + " )...") 

            if args.replay_init_and_main and args.warmstart:
                
                if args.resample_sol:

                    dt_res = flipping_task_init.dt / refinement_scale
                    q_replay_init = resampler(init_solutions[best_index]["q"], init_solutions[best_index]["q_dot"],\
                                        flipping_task_init.dt, dt_res,\
                                        {'x': flipping_task_init.q,\
                                            'p': flipping_task_init.q_dot,\
                                            'ode': flipping_task_init.q_dot,\
                                            'quad': 0})
                    
                    sol_replayer_init = ReplaySol(dt_res,\
                                            joint_list = flipping_task_init.joint_names,\
                                            q_replay = q_replay_init, srt_msg = "\nReplaying initialization trajectory") 

                else:

                    q_replay_init = init_solutions[best_index]["q"]

                    sol_replayer_init = ReplaySol(dt = flipping_task_init.dt,\
                                            joint_list = flipping_task_init.joint_names,\
                                            q_replay = q_replay_init, srt_msg = "\nReplaying initialization trajectory")
                

                while True:
                    
                    sol_replayer_init.sleep(1)
                    sol_replayer_init.replay(is_floating_base = False, play_once = True)
                    sol_replayer_init.sleep(0.5)
                    sol_replayer.replay(is_floating_base = False, play_once = True)
                    
            else:

                sol_replayer.sleep(0.5)
                sol_replayer.replay(is_floating_base = False, play_once = False)

    else: # replaying init solutions(if used) not supported yet

        best_index = get_min_cost_index(sol_costs, solve_failed_array)
        sol_replayer = [None] * n_glob_tests
        if args.rviz_replay:
            
            for i in range(n_glob_tests):
                
                q_replay = solutions[i]["q"]
                sol_replayer[i] = ReplaySol(dt = flipping_task.dt,\
                                            joint_list = flipping_task.joint_names,\
                                            q_replay = q_replay, \
                                            srt_msg = "\nReplaying solution ( n." + str(i + 1) + " /" +\
                                            str(n_glob_tests) + " ). Optimal is n." + str(best_index) + "...") 
            while True:

                for i in range(n_glob_tests):

                    sol_replayer[i].sleep(1)
                    sol_replayer[i].replay(is_floating_base = False, play_once = True)
                    

    # closing all child processes and exiting
    rviz_window.terminate()
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
                        help = 'whether to dump results to file', default = False)
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
    parser.add_argument('--warmstart', '-ws', type=str2bool,\
                        help = 'whether to first solve an initialization problem and then use that solution for the main one', default = False)
    parser.add_argument('--soft_warmstart', '-sws', type=str2bool,\
                        help = 'whether to use soft pose constraint for the initalization problem ', default = True)
    parser.add_argument('--replay_init_and_main', '-rsh', type=str2bool,\
                        help = 'whether to replay both initialization and main solution on RVIz (only valid if warmstart == True)', default = False)
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)
    parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
                        help = 'whether to load the initial guess from file', default = False)
    parser.add_argument('--replay_only_best', '-rplb', type=str2bool,\
                        help = 'whether to replay only the best solution or not', default = True)
    # parser.add_argument('--is_sliding_wrist', '-isw', type=str2bool,\
    #                     help = 'whether to add a sliding co-design d.o.f. on the second joint or the wrist', default = False)

    args = parser.parse_args()

    main(args)