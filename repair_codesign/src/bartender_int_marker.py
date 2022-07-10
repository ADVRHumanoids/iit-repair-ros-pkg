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
import multiprocessing

import rospkg

from codesign_pyutils.ros_utils import MarkerGen
from codesign_pyutils.miscell_utils import str2bool
from codesign_pyutils.math_utils import quat2rot, rot_error, rot_error2, get_cocktail_aux_rot

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

cocktail_size = 0.04

soft_tracking = False

def add_bartender_cnstrnt(index, prb, nodes, posl, posr, rotl, rotr, is_soft = False):

    pos_cnstrnt = prb.createConstraint("keep_baretender_pos" + str(index), posr - posl, nodes = nodes)
    rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index), rot_error2( get_cocktail_aux_rot(rotr), rotl), nodes = nodes)
    
    return pos_cnstrnt, rot_cnstrnt

def add_lpose_cnstrnt(index, prb, nodes, pos, rot, pos_ref, rot_ref, weight_pos = 1.0, weight_rot = 1.0, is_only_pos = False, is_soft = False):

    if is_soft:

        pos_cnstrnt = prb.createConstraint("init_left_tcp_pos" + str(index), pos - pos_ref, nodes = nodes)
    
        rot_cnstrnt = None

        if not is_only_pos:

            rot_cnstrnt = prb.createConstraint("init_left_tcp_rot" + str(index), rot_error2(rot, rot_ref, epsi = 0.001), nodes = nodes)

    else:

        pos_cnstrnt = prb.createIntermediateCost("init_left_tcp_pos_soft" + str(index), weight_pos * cs.sumsqr(pos - pos_ref), nodes = nodes)

        rot_cnstrnt = None

        if not is_only_pos:

            rot_cnstrnt = prb.createIntermediateCost("init_left_tcp_soft" + str(index), weight_rot * cs.sumsqr(rot_error2(rot, rot_ref, epsi = 0.001)), nodes = nodes)

    return pos_cnstrnt, rot_cnstrnt

def add_rpose_cnstrnt(index, prb, nodes, pos, rot, pos_ref, rot_ref, weight_pos = 1.0, weight_rot = 1.0, is_only_pos = False, is_soft = False):

    if is_soft:

        pos_cnstrnt = prb.createConstraint("init_right_tcp_pos" + str(index), pos - pos_ref, nodes = nodes)
        
        rot_cnstrnt = None
        
        if not is_only_pos:

            rot_cnstrnt = prb.createConstraint("init_right_tcp_rot" + str(index), rot_error2(rot, rot_ref, epsi = 0.001), nodes = nodes)

    else:

        pos_cnstrnt = prb.createIntermediateCost("init_right_tcp_pos_soft" + str(index), weight_pos * cs.sumsqr(pos - pos_ref), nodes = nodes)

        rot_cnstrnt = None

        if not is_only_pos:

            rot_cnstrnt = prb.createIntermediateCost("init_right_tcp_rot_soft" + str(index), weight_rot * cs.sumsqr(rot_error2(rot, rot_ref, epsi = 0.001)), nodes = nodes)

    return pos_cnstrnt, rot_cnstrnt

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
    
    # load urdf
    urdf = open(urdf_full_path, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # joint names
    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

    # parameters
    n_q = kindyn.nq()
    n_v = kindyn.nv()
    tf = 5.0
    n_nodes = 1
    dt = tf / n_nodes
    lbs = kindyn.q_min() 
    ubs = kindyn.q_max()

    prb = problem.Problem(n_nodes)

    q = prb.createStateVariable('q', n_q)
    q_dot = prb.createInputVariable('q_dot', n_v)
    q_design = q[1, 2, 3, 2 + (arm_dofs + 2), 3 + (arm_dofs + 2)] # design vector
    
    # targets (parameters of the problem)
    init_pos = prb.createParameter('init_pos', 3)
    trgt_pos = prb.createParameter('trgt_pos', 3)
    init_rot = prb.createParameter('init_rot', 4)
    trgt_rot = prb.createParameter('trgt_rot', 4)

    prb.setDynamics(q_dot)
    prb.setDt(dt)  

    transcription_method = 'multiple_shooting'
    transcription_opts = dict(integrator='RK4')
    Transcriptor.make_method(transcription_method, prb, transcription_opts)  # setting the transcriptor

    # getting some useful kinematic quantities
    fk_ws = cs.Function.deserialize(kindyn.fk("working_surface_link"))
    ws_link_pos = fk_ws(q = np.zeros((n_q, 1)).flatten())["ee_pos"] # w.r.t. world
    ws_tcp_rot = fk_ws(q = np.zeros((n_q, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)

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

    rarm_cocktail_pos = rarm_tcp_pos + rarm_tcp_rot @ cs.vertcat(0, 0, cocktail_size)
    rarm_cocktail_rot = rarm_tcp_rot
    larm_cocktail_pos = larm_tcp_pos + larm_tcp_rot @ cs.vertcat(0, 0, cocktail_size)
    larm_cocktail_rot = larm_tcp_rot

    # roll and shoulder vars equal
    prb.createConstraint("same_roll", q[3] - q[3 + (arm_dofs + 2)])
    prb.createConstraint("same_shoulder_w", q[2] - q[2 + (arm_dofs + 2)])

    # design vars equal on all nodes 
    prb.createConstraint("single_var_cnstrnt", q_design - q_design.getVarOffset(-1), nodes = range(1, n_nodes + 1))

    # lower and upper bounds for design variables and joint variables
    q.setBounds(lbs, ubs) 

    # TCPs above working surface
    keep_tcp1_above_ground = prb.createConstraint("keep_tcp1_above_ground", rarm_tcp_pos_wrt_ws[2])
    keep_tcp1_above_ground.setBounds(0, cs.inf)
    keep_tcp2_above_ground = prb.createConstraint("keep_tcp2_above_ground", larm_tcp_pos_wrt_ws[2])
    keep_tcp2_above_ground.setBounds(0, cs.inf)

    # keep baretender pose throught the trajectory
    add_bartender_cnstrnt(0, prb, range(0, n_nodes + 1), larm_cocktail_pos,  rarm_cocktail_pos, larm_cocktail_rot, rarm_cocktail_rot)

    # with baretender constraint only one reference pose is sufficient for both arms --> right arm conventionally chosen as base
    add_rpose_cnstrnt(0, prb, 0, rarm_cocktail_pos, rarm_cocktail_rot, init_pos, quat2rot(init_rot), is_only_pos = True, is_soft = False)
    add_rpose_cnstrnt(1, prb, n_nodes, rarm_cocktail_pos, rarm_cocktail_rot, trgt_pos, quat2rot(trgt_rot), is_only_pos = True, is_soft = False)

    # add_lpose_cnstrnt(0, prb, 0, larm_cocktail_pos, larm_cocktail_rot, init_pos, quat2rot(init_rot), is_only_pos = True, is_soft = False)
    # add_lpose_cnstrnt(1, prb, n_nodes, larm_cocktail_pos, larm_cocktail_rot, trgt_pos, quat2rot(trgt_rot), is_only_pos = True, is_soft = False)
    
    # min inputs 

    prb.createIntermediateCost("max_global_manipulability", 0.01 * cs.sumsqr(q_dot))  # minimizing the joint accelerations ("responsiveness" of the trajectory)
    
    ## Creating the solver and solving the problem
    slvr = solver.Solver.make_solver(solver_type, prb, slvr_opt) 

    init_pose_marker_topic = "init_pose"
    trgt_pose_marker_topic = "trgt_pose"
    
    # process = multiprocessing.Process(target = trgt_poses_pub)
    # process.start()

    rviz_marker_gen = MarkerGen()
    rviz_marker_gen.add_marker("world", [0.7, 0.2, 0.7], init_pose_marker_topic, 0.3) 
    rviz_marker_gen.add_marker("world", [0.7, 0.2, 0.8], trgt_pose_marker_topic, 0.3) 
    rviz_marker_gen.spin()

    if exists(urdf_full_path): # clear generated urdf file

        os.remove(urdf_full_path)
    
    input("Press Enter to start solving the problem!!!\n")

    while True:
    
        init_pos_trgt, init_rot_trgt = rviz_marker_gen.getPose(init_pose_marker_topic)
        trgt_pos_trgt, trgt_rot_trgt = rviz_marker_gen.getPose(trgt_pose_marker_topic)
        
        init_pos.assign(init_pos_trgt)
        init_rot.assign(init_rot_trgt)
        init_pos.assign(trgt_pos_trgt)
        init_rot.assign(trgt_rot_trgt)

        t = time.time()

        try:

            slvr.solve()  # solving

            solution_time = time.time() - t
            print(f'solved in {solution_time} s')

            init_pos = trgt_pos
            trgt_rot = trgt_rot

            solution = slvr.getSolutionDict() # extracting solution
            cnstr_opt = slvr.getConstraintSolutionDict()

            q_sol = solution["q"]

            if args.rviz_replay and args.launch_rviz:

                rpl_traj = replay_trajectory(dt = dt, joint_list = joint_names, q_replay = q_sol)  
                # rpl_traj.publish_joints(q_sol, is_floating_base = False)
                rpl_traj.replay(is_floating_base = False)

        except:
            
            print('Failed to solve problem')

            continue

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--gen_urdf', '-g', type=str2bool, help = 'whether to generate urdf from xacro', default = True)
    parser.add_argument('--launch_rviz', '-rvz', type=str2bool, help = 'whether to launch rviz or not', default = True)
    parser.add_argument('--rviz_replay', '-rpl', type=str2bool, help = 'whether to replay the solution on RViz', default = True)

    args = parser.parse_args()
    main(args)