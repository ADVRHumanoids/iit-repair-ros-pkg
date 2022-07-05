#!/usr/bin/env python3

from curses import KEY_B2
from sympy import Q
from horizon import problem
from horizon.utils import utils, kin_dyn, plotter, mat_storer
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
import shutil
import time

import rospkg
import rospy

import subprocess

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

arm_dofs = 7 

solver_type = 'ipopt'
slvr_opt = {"ipopt.tol": 0.0001, "ipopt.max_iter": 1000} 

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def main(args):

    # preliminary ops
    if args.gen_urdf:
        try:
            xacro_gen = subprocess.check_call(["xacro", "-o", urdf_full_path, xacro_full_path])
        except:
            print('Failed to generate URDF.')

    if args.dump_mat:
        ms = mat_storer.matStorer(results_path + f'/{file_name}.mat')

    if args.launch_rviz:
        try:

            rviz_window = subprocess.Popen(["roslaunch", "repair_urdf", "repair_full.launch"])

        except:
            print('Failed to launch RViz.')

    # load urdf
    urdf = open(urdf_full_path, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # joint names
    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

    # dangerous move
    design_var_names = [joint_names[0], joint_names[1], joint_names[2], joint_names[11], joint_names[3], joint_names[12]]
    arm1_jnt_names = joint_names[13:(13 + arm_dofs)]
    arm2_jnt_names = joint_names[4:(4 + arm_dofs)]
    
    print("Design vars: ", joint_names)
    print("Design vars: ", design_var_names)
    print("Arm1 joints: ", arm1_jnt_names)
    print("Arm2 joints: ", arm2_jnt_names)

    # parameters
    n_q = kindyn.nq()
    n_v = kindyn.nv()
    tf = 3
    n_nodes = 1000
    dt = tf / n_nodes
    lbs = kindyn.q_min() 
    ubs = kindyn.q_max()
    
    prb = problem.Problem(n_nodes)
    q = prb.createStateVariable('q', n_q)
    q_dot = prb.createInputVariable('q_dot', n_v)
    q_design = q[1, 2, 3, 2 + arm_dofs, 3 + arm_dofs] # design vector
    
    q_init = np.zeros((n_q, 1)).flatten()
    q_aux = np.array([0, 0.5, 0.3, -0.2, -0.4, -1, 0.8, -2, -0.4, 0.4, 1.4, 0.3, -0.2, 0.25, -1.2, -1.45, -1.3, 0.5, -0.72, 0.3])
    
    prb.setDynamics(q_dot)
    prb.setDt(dt)  

    transcription_method = 'multiple_shooting'
    transcription_opts = dict(integrator='RK4')
    trscptr = Transcriptor.make_method(transcription_method, prb, transcription_opts)  # setting the transcriptor

    # getting some useful kinematic quantities
    fk_ws = cs.Function.deserialize(kindyn.fk("working_surface_link"))
    ws_link_pos = fk_ws(q = np.zeros((n_q, 1)).flatten())["ee_pos"] # w.r.t. world
    ws_tcp_rot = fk_ws(q = np.zeros((n_q, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)

    fk_dummy_link = cs.Function.deserialize(kindyn.fk("dummy_link"))
    dummy_link_pos = fk_dummy_link(q = q)["ee_pos"] # w.r.t. world
    dummy_link_rot = fk_dummy_link(q = q)["ee_rot"] # w.r.t. world (3x3 rot matrix)

    fk_arm_r = cs.Function.deserialize(kindyn.fk("arm_1_tcp")) 
    rarm_tcp_pos = fk_arm_r(q = q)["ee_pos"] # w.r.t. world
    rarm_tcp_rot = fk_arm_r(q = q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
    rarm_tcp_pos_wrt_ws = rarm_tcp_pos - ws_link_pos # pose w.r.t. working surface
    rarm_tcp_rot_wrt_ws = cs.inv(ws_tcp_rot) * rarm_tcp_rot # orient w.r.t. working surface

    fk_arm_l = cs.Function.deserialize(kindyn.fk("arm_2_tcp"))  
    larm_tcp_pos = fk_arm_l(q = q)["ee_pos"] # w.r.t. world
    larm_tcp_rot = fk_arm_l(q = q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
    larm_tcp_pos_wrt_ws = larm_tcp_pos - ws_link_pos # pose w.r.t. working surface
    larm_tcp_rot_wrt_ws = cs.inv(ws_tcp_rot) * larm_tcp_rot # orient w.r.t. working surface

    # fixed x constraint
    prb.createConstraint("fixed_x", q[0])

    # roll and shouldeer vars equal
    prb.createConstraint("same_roll", q[3] - q[3 + arm_dofs])
    prb.createConstraint("same_shoulder_w", q[2] - q[2 + arm_dofs])
    # design vars equal on all nodes 
    prb.createConstraint("single_var_cnstrnt", q_design - q_design.getVarOffset(-1), nodes = range(1, n_nodes))

    # lower and upper bounds for design variables and joint variables
    q.setBounds(lbs, ubs) 

    # TCPs above working surface

    # keep_tcp1_above_ground = prb.createConstraint("keep_tcp1_above_ground", rarm_tcp_pos_wrt_ws[2], nodes = range(1, n_nodes - 1))
    # keep_tcp1_above_ground.setLowerBounds(0)
    # keep_tcp2_above_ground = prb.createConstraint("keep_tcp2_above_ground", larm_tcp_pos_wrt_ws[2], nodes = range(1, n_nodes - 1))
    # keep_tcp2_above_ground.setLowerBounds(0)
    # COSTS

    # min inputs 

    prb.createIntermediateCost("min_q_dot", 0.01 * cs.sumsqr(q_dot))  # minimizing the joint accelerations ("responsiveness" of the trajectory)
    
    # left arm pose error
    prb.createIntermediateCost("init_left_tcp_pos_error", 100000000 * cs.sumsqr(larm_tcp_pos - fk_arm_l(q = q_init)["ee_pos"]), nodes = 0)
    prb.createFinalCost("final_left_tcp_pos_error", 100000000 * cs.sumsqr(larm_tcp_pos - fk_arm_l(q = q_aux)["ee_pos"]))

    # right arm pose error
    prb.createIntermediateCost("init_right_tcp_pos_error", 100000000 * cs.sumsqr(rarm_tcp_pos - fk_arm_r(q = q_init)["ee_pos"]), nodes = 0)
    prb.createFinalCost("final_right_tcp_pos_error", 100000000 * cs.sumsqr(rarm_tcp_pos - fk_arm_r(q = q_aux)["ee_pos"]))

    ## Creating the solver and solving the problem
    slvr = solver.Solver.make_solver(solver_type, prb, slvr_opt) 
    t = time.time()
    slvr.solve()  # solving
    solution_time = time.time() - t
    print(f'solved in {solution_time} s')
    solution = slvr.getSolutionDict() # extracting solution
    cnstr_opt = slvr.getConstraintSolutionDict()

    q_sol = solution["q"]
    ms.store({**solution, **cnstr_opt})

    if args.rviz_replay and args.launch_rviz:
        rpl_traj = replay_trajectory(dt = dt, joint_list = joint_names, q_replay = q_sol)  
        rpl_traj.sleep(1.)
        rpl_traj.replay(is_floating_base = False)

    if exists(urdf_full_path): # clear generated urdf file

        os.remove(urdf_full_path)

    # rviz_window.kill()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--gen_urdf', '-g', type=str2bool, help = 'whether to generate urdf from xacro', default = True)
    parser.add_argument('--dump_mat', '-dm', type=str2bool, help = 'whether to dump results to mat file', default = True)
    parser.add_argument('--launch_rviz', '-rvz', type=str2bool, help = 'whether to launch rviz or not', default = True)
    parser.add_argument('--rviz_replay', '-rpl', type=str2bool, help = 'whether to replay the solution on RViz', default = True)

    args = parser.parse_args()
    main(args)