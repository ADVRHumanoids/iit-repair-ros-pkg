#!/usr/bin/env python3

from operator import is_not
from horizon import problem
from horizon.utils.resampler_trajectory import resampler
from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver
import os, argparse
from os.path import exists

import numpy as np

from datetime import datetime
from datetime import date
import time

import subprocess

import rospkg

from codesign_pyutils.ros_utils import FramePub, ReplaySol
from codesign_pyutils.miscell_utils import str2bool, SolDumper,\
                                           wait_for_confirmation
from codesign_pyutils.horizon_utils import FlippingTaskGen

from horizon.utils import mat_storer

import warnings

## getting some useful information to be used for data storage
today = date.today()
today_is = today.strftime("%d-%m-%Y")
now = datetime.now()
current_time = now.strftime("_%H_%M_%S")

# useful paths
rospackage = rospkg.RosPack() # Only for taking the path to the leg package
urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
codesign_path = rospackage.get_path("repair_codesign")
results_path = codesign_path + "/test_results"

file_name = os.path.splitext(os.path.basename(__file__))[0]

# task-specific options
right_arm_picks = True
filling_n_nodes = 10
rot_error_epsi = 0.001

# generating samples along working surface y direction
n_y_samples = 5
y_sampl_ub = 0.3 
y_sampl_lb = - y_sampl_ub
dy = (y_sampl_ub - y_sampl_lb) / (n_y_samples - 1)

y_sampling = np.array( [0.0] * n_y_samples)
for i in range(n_y_samples):
    
    y_sampling[i] = y_sampl_lb + dy * i

# random init settings( if used)
seed = 1
np.random.seed(seed)

# resampler option (if used)
refinement_scale = 10

# longitudinal extension of grasped object
cocktail_size = 0.08

# solver options
solver_type = 'ipopt'
slvr_opt = {
    "ipopt.tol": 0.0001, 
    "ipopt.max_iter": 10000,
    "ilqr.verbose": True}

# loaded initial guess options
init_guess_filename = "init3.mat"
init_load_abs_path = results_path + "/init_guess/" + init_guess_filename


def solve_prb_standalone(task, slvr, q_init=None, prbl_name = "Problem",
                         on_failure = "\n Failed to solve problem!! \n'"):

    # standard routine for solving the problem

    solve_failed = False

    try:
        
        if q_init is not None:

            task.q.setInitialGuess(q_init) # random initialization

        t = time.time()

        slvr.solve()  # solving

        solution_time = time.time() - t

        print(f'\n {prbl_name} solved in {solution_time} s \n')
        
        solve_failed = False

    except:
        
        print(on_failure)

        solve_failed = True

    
    return solve_failed

def solve_main_prb_soft_init(args, task, slvr_init, slvr, q_init_hard=None,\
                             prbl_name = "Problem",\
                             on_failure = "\n Failed to solve problem with soft initialization!! \n'"):

    # Routine for solving the "hard" problem employing the solution 
    # from another solver.
    # If solution fails, the user can choose to try to solve the "hard"
    # problem without any initialization from the other solution

    task.q.setInitialGuess(slvr_init.getSolutionDict()["q"])
   
    solve_failed = False

    try: # try to solve the main problem with the results of the soft initialization
    
        t = time.time()

        slvr.solve()  # solving soft problem

        solution_time = time.time() - t 

        print(f'\n {prbl_name} solved in {solution_time} s \n')

        solve_failed = False
    
    except:
        
        print(on_failure)

        proceed_to_hard_anyway = wait_for_confirmation(\
                                 do_something = "try to solve the main problem without soft initialization",\
                                 or_do_something_else = "stop here", \
                                 on_confirmation = "Trying to solve the main problem  without soft initialization...",\
                                 on_denial = "Stopping here!")
        
        if proceed_to_hard_anyway:
            
            if args.use_init_guess:

                solve_failed = solve_prb_standalone(task, slvr, q_init_hard)

            else:

                solve_failed = solve_prb_standalone(task, slvr)

        else:

            solve_failed = True

    return solve_failed

def try_init_solve_or_go_on(args, init_task, init_slvr, task, slvr,\
                            q_init_hard=None, q_init_soft=None):
        
        # Routine for solving the initialization problem.
        # If solution fails, the user can choose to solve
        # the main problem anyway, without employing the 
        # solution the initialization problem.

        soft_sol_failed =  False
        sol_failed = False

        if args.use_init_guess:

            init_task.q.setInitialGuess(q_init_soft) # random initialization for soft problem (if flag enabled)

        try: # try solving the soft initialization problem
            
            t = time.time()

            init_slvr.solve()  # solving soft problem

            solution_time = time.time() - t 

            print(f'\n Soft initialization problem solved in {solution_time} s \n')

            soft_sol_failed = False
        
        except: # failed to solve the soft problem 

            soft_sol_failed = True

            print('\n Failed to solve the soft initialization problem!! \n')

            proceed_to_hard_anyway = wait_for_confirmation(\
                                     do_something = "try to solve the main problem anyway",\
                                     or_do_something_else = "stop here", \
                                     on_confirmation = "Trying to solve the main problem  ...",\
                                     on_denial = "Stopping here!")

            if proceed_to_hard_anyway:
                
                sol_failed = solve_prb_standalone(task, slvr, q_init_hard)
            
            else:

                sol_failed = True
            
        return soft_sol_failed, sol_failed

def build_multiple_flipping_tasks(args, flipping_task, y_sampling, right_arm_picks, urdf_full_path, is_soft_pose_cnstr = False, epsi = 0.001):

    # Routine for automatically adding multiple flipping tasks,
    # based on the provided y-axis sampling. 

    # add tasks to the task holder object
    next_node = 0 # used to place the next task on the right problem nodes
    for i in range(len(y_sampling)):

        next_node = flipping_task.add_in_place_flip_task(init_node = next_node,\
                                    object_pos_wrt_ws = np.array([0.0, y_sampling[i], 0.0]), \
                                    object_q_wrt_ws = np.array([0, 1, 0, 0]), \
                                    #  pick_q_wrt_ws = np.array([np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0]),\
                                    right_arm_picks = right_arm_picks)

    print("Flipping task node list: ", flipping_task.nodes_list)
    print("Total employed nodes: ", flipping_task.total_nnodes)
    print("Number of added subtasks:", flipping_task.n_of_tasks)

    # build the problem 
    flipping_task.init_prb(urdf_full_path, weight_glob_man = args.weight_global_manip,\
                            is_soft_pose_cnstr = is_soft_pose_cnstr, epsi = epsi)


def main(args):

    # preliminary ops
    if args.gen_urdf:

        try:

            xacro_gen = subprocess.check_call(["xacro", "-o",\
                                               urdf_full_path,\
                                               xacro_full_path])
            
        except:

            print('Failed to generate URDF.')

    if args.launch_rviz:

        try:

            rviz_window = subprocess.Popen(["roslaunch",\
                                            "repair_urdf",\
                                            "repair_full_markers.launch"])

        except:

            print('Failed to launch RViz.')
    
    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)

    if args.soft_warmstart:

        # "hard" flipping task
        flipping_task = FlippingTaskGen(cocktail_size = cocktail_size, filling_n_nodes = filling_n_nodes)
        build_multiple_flipping_tasks(args, flipping_task, y_sampling,\
                                    right_arm_picks, urdf_full_path,\
                                    is_soft_pose_cnstr = False, \
                                    epsi = rot_error_epsi)

        # "soft" flipping task to be used as initialization to the hard
        flipping_task_init = FlippingTaskGen(cocktail_size = cocktail_size, filling_n_nodes = filling_n_nodes)
        build_multiple_flipping_tasks(args, flipping_task_init, y_sampling,\
                                    right_arm_picks, urdf_full_path,\
                                    is_soft_pose_cnstr = True, \
                                    epsi = rot_error_epsi)

        transcription_method = 'multiple_shooting'
        transcription_opts_soft = dict(integrator='RK4')
        transcription_opts_hard = dict(integrator='RK4')

        if solver_type != "ilqr":

            Transcriptor.make_method(transcription_method,\
                                    flipping_task.prb,\
                                    transcription_opts_hard)  # setting the transcriptor for the main hard problem
            Transcriptor.make_method(transcription_method,\
                                    flipping_task_init.prb,\
                                    transcription_opts_soft)  # setting the transcriptor for the initialization problem

        ## Creating the solver
        slvr = solver.Solver.make_solver(solver_type, flipping_task.prb, slvr_opt)
        slvr_init = solver.Solver.make_solver(solver_type, flipping_task_init.prb, slvr_opt)

        if solver_type == "ilqr":

            slvr.set_iteration_callback()
            slvr_init.set_iteration_callback()

    else:
        # generate a single task depending on the arguments

        flipping_task = FlippingTaskGen(cocktail_size = cocktail_size, filling_n_nodes = filling_n_nodes)
        build_multiple_flipping_tasks(args, flipping_task, y_sampling,\
                                    right_arm_picks, urdf_full_path,\
                                    is_soft_pose_cnstr = args.soft_pose_cnstrnt, \
                                    epsi = rot_error_epsi)
    
        transcription_method = 'multiple_shooting'
        transcription_opts = dict(integrator='RK4')

        if solver_type != "ilqr":
            
            Transcriptor.make_method(transcription_method,\
                                    flipping_task.prb,\
                                    transcription_opts)  # setting the transcriptor

        ## Creating the solver
        slvr = solver.Solver.make_solver(solver_type, flipping_task.prb, slvr_opt)

        if solver_type == "ilqr":

            slvr.set_iteration_callback()
    
    pose_pub = FramePub("frame_pub")
    init_frame_name = "/repair/init_pose"
    trgt_frame_name = "/repair/trgt_pose"
    # pose_pub.add_pose(flipping_task.object_pos_lft[0], flipping_task.object_q_lft[0],\
    #                   init_frame_name, "working_surface_link")
    # pose_pub.add_pose(flipping_task.lft_pick_pos[0], flipping_task.lft_pick_q[0],\
    #                   trgt_frame_name, "working_surface_link")
    pose_pub.spin()

    if exists(urdf_full_path): # clear generated urdf file

        os.remove(urdf_full_path)
    
    if args.dump_sol:

        sol_dumper = SolDumper(results_path)

    solve_failed = False
    if args.soft_warmstart:

        soft_sol_failed = False

    q_init_hard = None
    q_init_soft = None
    
    if args.use_init_guess:
        
        if args.load_initial_guess:
            
            try:
                                
                ms_ig_load = mat_storer.matStorer(init_load_abs_path)

                q_init_hard = ms_ig_load.load()["q"]
                q_init_soft = q_init_hard
            
            except:
                
                warnings.warn("Failed to load initial guess from file! I will use random intialization.")

                q_init_hard = np.random.uniform(flipping_task.lbs, flipping_task.ubs,\
                                        (1, flipping_task.nq)).flatten()
                q_init_soft = q_init_hard


            # insert check on correct initialization dimensions!!!!!
                        
        else:

            q_init_hard = np.random.uniform(flipping_task.lbs, flipping_task.ubs,\
                                        (1, flipping_task.nq)).flatten()
            q_init_soft = q_init_hard

        # if not args.soft_warmstart:

        #     print("Initialization for hard problem: ", q_init_hard)
        
        # else:
            
        #     print("Initalization for soft problem: ", q_init_soft)
        #     print("Initalization for hard problem: ", q_init_hard)

    # Solve
    if not args.soft_warmstart:

        solve_failed = solve_prb_standalone(flipping_task, slvr, q_init_hard)

    else:

        soft_sol_failed, solve_failed = try_init_solve_or_go_on(args,\
                                        flipping_task_init, slvr_init,\
                                        flipping_task, slvr,\
                                        q_init_hard = q_init_hard,\
                                        q_init_soft = q_init_soft)

        if not soft_sol_failed: # soft solution succedeed

            solve_failed = solve_main_prb_soft_init(args, flipping_task,\
                                                    slvr_init, slvr)

    # Postprocessing
    if not solve_failed:
        
        solution = slvr.getSolutionDict() # extracting solution
        print("Solution cost: ", solution["opt_cost"])
        
        if args.soft_warmstart and (soft_sol_failed != True):

            solution_soft = slvr_init.getSolutionDict() # extracting solution from soft problem
            print("Soft solution cost: ", solution_soft["opt_cost"])

        if args.dump_sol:

            store_current_sol = wait_for_confirmation(do_something = "store the current solution",\
                                or_do_something_else = "avoid storing it",\
                                on_confirmation = "Storing current solution  ...",\
                                on_denial = "Current solution will be discarted!")

            if store_current_sol:
                
                cnstr_opt = slvr.getConstraintSolutionDict()
                other_stuff = {"dt": flipping_task.dt, "filling_nodes": flipping_task.filling_n_nodes,
                                "task_base_nnodes": flipping_task.task_base_n_nodes,
                                "right_arm_picks": flipping_task.rght_arm_picks, \
                                "wman_base": args.weight_global_manip, \
                                "wpo_bases": args.base_weight_pos, "wrot_base": args.base_weight_rot, \
                                "wman_actual": flipping_task.weight_glob_man, \
                                "wpos_actual": flipping_task.weight_pos, "wrot_actual": flipping_task.weight_rot}

                full_solution = {**solution, **cnstr_opt, **other_stuff}
                

                sol_dumper.add_storer(full_solution, results_path,\
                                      "flipping_repair", True)

                if args.soft_warmstart and (soft_sol_failed != True):

                    cnstr_opt_soft = slvr.getConstraintSolutionDict()
                    other_stuff = {"dt": flipping_task.dt, "filling_nodes": flipping_task.filling_n_nodes,
                                "task_base_nnodes": flipping_task.task_base_n_nodes,
                                "right_arm_picks": flipping_task.rght_arm_picks, \
                                "wman_base": args.weight_global_manip, \
                                "wpo_bases": args.base_weight_pos, "wrot_base": args.base_weight_rot, \
                                "wman_actual": flipping_task.weight_glob_man, \
                                "wpos_actual": flipping_task.weight_pos, "wrot_actual": flipping_task.weight_rot}
                                
                    full_solution_soft = {**solution_soft, **cnstr_opt_soft, **other_stuff}
                    
                    sol_dumper.add_storer(full_solution_soft, results_path,\
                                          "flipping_repair_soft", True)

                sol_dumper.dump() 

                print("\nSolutions dumped. \n")
        
        if args.rviz_replay:

            q_replay = None
            q_replay_soft = None

            if args.resample_sol:
                
                dt_res = flipping_task.dt / refinement_scale

                q_replay = resampler(solution["q"], solution["q_dot"],\
                                     flipping_task.dt, dt_res,\
                                     {'x': flipping_task.q, 'p': flipping_task.q_dot,\
                                     'ode': flipping_task.q_dot, 'quad': 0})

                sol_replayer = ReplaySol(dt_res,
                                         joint_list = flipping_task.joint_names,
                                         q_replay = q_replay) 

            else:
                
                q_replay = solution["q"]

                sol_replayer = ReplaySol(dt = flipping_task.dt,\
                                         joint_list = flipping_task.joint_names,\
                                         q_replay = q_replay) 

            if args.replay_soft_and_hard and (not soft_sol_failed):
                
                if args.resample_sol:

                    dt_res = flipping_task_init.dt / refinement_scale
                    q_replay_soft = resampler(solution_soft["q"], solution_soft["q_dot"],\
                                        flipping_task_init.dt, dt_res,\
                                        {'x': flipping_task_init.q,\
                                         'p': flipping_task_init.q_dot,\
                                         'ode': flipping_task_init.q_dot,\
                                         'quad': 0})
                    
                    sol_replayer_soft = ReplaySol(dt_res,\
                                            joint_list = flipping_task_init.joint_names,\
                                            q_replay = q_replay_soft, srt_msg = "\nReplaying soft trajectory") 

                else:

                    q_replay_soft = solution_soft["q"]

                    sol_replayer_soft = ReplaySol(dt = flipping_task_init.dt,\
                                            joint_list = flipping_task_init.joint_names,\
                                            q_replay = q_replay_soft, srt_msg = "\nReplaying soft trajectory")
                

                while True:
                    
                    sol_replayer_soft.sleep(1)
                    sol_replayer_soft.replay(is_floating_base = False, play_once = True)
                    sol_replayer_soft.sleep(0.5)
                    sol_replayer.replay(is_floating_base = False, play_once = True)
                    
            
            else:

                sol_replayer.sleep(0.5)
                sol_replayer.replay(is_floating_base = False, play_once = False)

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
    parser.add_argument('--soft_bartender_cnstrnt', '-sbc', type=str2bool,\
                        help = 'whether to use soft bartender constraints', default = False)
    parser.add_argument('--soft_pose_cnstrnt', '-spc', type=str2bool,\
                        help = 'whether to use soft pose constraints or not', default = False)
    parser.add_argument('--base_weight_pos', '-wp', type = np.double,\
                        help = 'base weight for position tracking (if using soft constraints)', default = 0.001)
    parser.add_argument('--base_weight_rot', '-wr', type = np.double,\
                        help = 'base weight for orientation tracking (if using soft constraints)', default = 0.001)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--soft_warmstart', '-sws', type=str2bool,\
                        help = 'whether to use the solution to the soft problem as initialization for the hard one', default = False)
    parser.add_argument('--replay_soft_and_hard', '-rsh', type=str2bool,\
                        help = 'whether to replay both soft and hard solution on RVIz (only valid if soft_warmstart == True)', default = False)
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)
    parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
                        help = 'whether to load the initial guess from file', default = False)

    args = parser.parse_args()

    main(args)