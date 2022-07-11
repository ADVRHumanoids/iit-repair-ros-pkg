#!/usr/bin/env python3

from horizon import problem
from horizon.utils import mat_storer
from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
import os, argparse
from os.path import exists

import numpy as np
import casadi as cs

from datetime import datetime
from datetime import date
import time

import subprocess

import rospkg

from codesign_pyutils.ros_utils import MarkerGen, FramePub, ReplaySol
from codesign_pyutils.miscell_utils import str2bool, SolDumper, wait_for_confirmation
from codesign_pyutils.horizon_utils import add_pose_cnstrnt, add_bartender_cnstrnt, FlippingTaskGen
from codesign_pyutils.math_utils import quat2rot

## Getting/setting some useful variables
today = date.today()
today_is = today.strftime("%d-%m-%Y")
now = datetime.now()
current_time = now.strftime("_%H_%M_%S")

rospackage = rospkg.RosPack() # Only for taking the path to the leg package
urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
codesign_path = rospackage.get_path("repair_codesign")
results_path = codesign_path + "/test_results"

file_name = os.path.splitext(os.path.basename(__file__))[0]

solver_type = 'ipopt'
slvr_opt = {"ipopt.tol": 0.0001, "ipopt.max_iter": 1000} 

def main(args):

    # preliminary ops
    if args.gen_urdf:

        try:

            xacro_gen = subprocess.check_call(["xacro", "-o", urdf_full_path, xacro_full_path])
            
        except:

            print('Failed to generate URDF.')

    if args.launch_rviz:

        try:

            rviz_window = subprocess.Popen(["roslaunch", "repair_urdf", "repair_full_markers.launch"])

        except:

            print('Failed to launch RViz.')
    
    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)

    # creating a flipping task
    flipping_task = FlippingTaskGen()
    flipping_task.add_task(init_node = 0, filling_n_nodes = 30)
    flipping_task.init_prb(urdf_full_path, weight_pos = args.weight_pos, weight_rot = args.weight_rot,\
                           weight_glob_man = args.weight_global_manip, is_soft_pose_cnstr = args.soft_pose_cnstrnt, epsi = 0.001)

    transcription_method = 'multiple_shooting'
    transcription_opts = dict(integrator='RK4')
    Transcriptor.make_method(transcription_method, flipping_task.prb, transcription_opts)  # setting the transcriptor
    
    ## Creating the solver and solving the problem
    slvr = solver.Solver.make_solver(solver_type, flipping_task.prb, slvr_opt) 

    pose_pub = FramePub("frame_pub")
    init_frame_name = "/repair/init_pose"
    trgt_frame_name = "/repair/trgt_pose"
    pose_pub.add_pose(flipping_task.rght_pick_pos_wrt_ws_default, flipping_task.rght_pick_q_wrt_ws_default, init_frame_name, "working_surface_link")
    pose_pub.add_pose(flipping_task.lft_pick_pos_wrt_ws_default, flipping_task.lft_pick_q_wrt_ws_default, trgt_frame_name, "working_surface_link")
    pose_pub.spin()

    if exists(urdf_full_path): # clear generated urdf file

        os.remove(urdf_full_path)
    
    if args.dump_sol:

        sol_dumper = SolDumper(results_path)

    solve_failed = False
    t = time.time()

    q_init = np.random.uniform(flipping_task.lbs, flipping_task.ubs, (1, flipping_task.nq))
        
    try:
        
        flipping_task.q.setInitialGuess(q_init.flatten())

        slvr.solve()  # solving

        solution_time = time.time() - t
        print(f'solved in {solution_time} s')

    except:
        
        print('\n Failed to solve problem!! \n')

        solve_failed = True
        
    if not solve_failed:
        
        solution = slvr.getSolutionDict() # extracting solution

        print(solution["opt_cost"])

        q_sol = solution["q"]

        if args.dump_sol:

            store_current_sol = wait_for_confirmation(do_something = "store the current solution", or_do_something_else = "avoid storing it", \
                                                    on_confirmation = "Storing current solution  ...", on_denial = "Current solution will be discarted!")

            if store_current_sol:
            
                cnstr_opt = slvr.getConstraintSolutionDict()

                full_solution = {**solution, **cnstr_opt}

                sol_dumper.add_storer(full_solution, results_path, "flipping_repair", True)

                sol_dumper.dump() 

                print("\n Done \n")
        
        if args.rviz_replay:

            sol_replayer = ReplaySol(dt = flipping_task.dt, joint_list = flipping_task.joint_names, q_replay = q_sol) 
            # sol_replayer.sleep(1.0)
            # sol_replayer.publish_joints(q_sol, is_floating_base = False, base_link = "world")
            sol_replayer.replay(is_floating_base = False, play_once = False)

        counter = counter + 1

    # closing all child processes and exiting
    rviz_window.terminate()
    exit()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--gen_urdf', '-g', type=str2bool, help = 'whether to generate urdf from xacro', default = True)
    parser.add_argument('--launch_rviz', '-rvz', type=str2bool, help = 'whether to launch rviz or not', default = True)
    parser.add_argument('--rviz_replay', '-rpl', type=str2bool, help = 'whether to replay the solution on RViz', default = True)
    parser.add_argument('--dump_sol', '-ds', type=str2bool, help = 'whether to dump results to file', default = False)
    parser.add_argument('--use_init_guess', '-ig', type=str2bool, help = 'whether to use initial guesses between solution loops', default = True)
    parser.add_argument('--soft_bartender_cnstrnt', '-sbc', type=str2bool, help = 'whether to use soft bartender constraints', default = False)
    parser.add_argument('--soft_pose_cnstrnt', '-spc', type=str2bool, help = 'whether to use soft pose constraints or not', default = False)
    parser.add_argument('--weight_pos', '-wp', type = np.double, help = 'weight for position tracking (if soft_pose_cnstrnt == True)', default = 10000)
    parser.add_argument('--weight_rot', '-wr', type = np.double, help = 'weight for orientation tracking (if soft_pose_cnstrnt == True)', default = 10000)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double, help = 'weight for global manipulability cost function', default = 100)

    args = parser.parse_args()
    main(args)