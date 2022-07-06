#!/usr/bin/env python3

from curses import KEY_B2
import multiprocessing
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

from geometry_msgs.msg import PoseStamped

import multiprocessing.process as Process
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

def rot2quat(R):

    # convert matrix to quaternion representation

    # quaternion Q ={eta, Epsi}
    # where eta = cos(theta/2), with theta belogs [- PI, PI] 
    # Epsi = sin(theta/2) * r, where r is the axis of rotation

    eta = 1/2 * cs.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
    epsi_1 = 1/2 * cs.sign(R[2, 1] - R[1, 2]) * cs.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1) 
    epsi_2 = 1/2 * cs.sign(R[0, 2] - R[2, 0]) * cs.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1) 
    epsi_3 = 1/2 * cs.sign(R[1, 0] - R[0, 1]) * cs.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)
    
    Q = cs.vertcat(eta, epsi_1, epsi_2, epsi_3)
    return Q

def rot_error(R_trgt, R_actual):

    Q_trgt = rot2quat(R_trgt)
    Q_actual = rot2quat(R_actual)

    rot_err1 = Q_trgt[0] * Q_actual[1] - Q_actual[0] * Q_trgt[1] + Q_actual[3] * Q_trgt[2] - Q_actual[2] * Q_trgt[3]
    rot_err2 = Q_trgt[0] * Q_actual[2] - Q_actual[0] * Q_trgt[2] - Q_actual[3] * Q_trgt[1] + Q_actual[1] * Q_trgt[3]
    rot_err3 = Q_trgt[0] * Q_actual[3] - Q_actual[0] * Q_trgt[3] + Q_actual[2] * Q_trgt[1] - Q_actual[1] * Q_trgt[2]

    return (rot_err1 * rot_err1 + rot_err2 * rot_err2 + rot_err3 * rot_err3)

def rot_error2(R_trgt, R_actual):

    R_err = R_trgt * cs.transpose(R_actual) # R_trgt * R_actual^T should be the identity matrix if error = 0

    I = np.zeros((3, 3))
    I[0, 0] = 1
    I[1, 1] = 1
    I[2, 2] = 1

    Err = R_err - I

    err = Err[0, 0] * Err[0, 0] + Err[1, 0] * Err[1, 0] + Err[2, 0] * Err[2, 0] + \
          Err[0, 1] * Err[0, 1] + Err[1, 1] * Err[1, 1] + Err[2, 1] * Err[2, 1] + \
          Err[0, 2] * Err[0, 2] + Err[1, 2] * Err[1, 2] + Err[2, 2] * Err[2, 2] 

    return err

def trgt_pose_publisher(pos_left, pos_rght, rot_lft, rot_rght):

    Q_lft = rot2quat(rot_lft)
    Q_rght = rot2quat(rot_rght)
    
    rospy.init_node('trgt_pose_pub', anonymous = False)

    publisher_lft = rospy.Publisher("/repair/trgt_pose_lft", PoseStamped, queue_size = 10)
    publisher_rght = rospy.Publisher("/repair/trgt_pose_rght", PoseStamped, queue_size = 10)

    pose_lft = PoseStamped()
    pose_rght = PoseStamped()

    pose_lft.header.stamp = rospy.Time.now()
    pose_lft.header.frame_id = "world"
    pose_lft.pose.position.x = pos_left[0]
    pose_lft.pose.position.y = pos_left[1]
    pose_lft.pose.position.z = pos_left[2]
    pose_lft.pose.orientation.x = Q_lft[1]
    pose_lft.pose.orientation.y = Q_lft[2]
    pose_lft.pose.orientation.z = Q_lft[3]
    pose_lft.pose.orientation.w = Q_lft[0]

    pose_rght.header.stamp = rospy.Time.now()
    pose_rght.header.frame_id = "world"
    pose_rght.pose.position.x = pos_rght[0]
    pose_rght.pose.position.y = pos_rght[1]
    pose_rght.pose.position.z = pos_rght[2]
    pose_rght.pose.orientation.x = Q_rght[1]
    pose_rght.pose.orientation.y = Q_rght[2]
    pose_rght.pose.orientation.z = Q_rght[3]
    pose_rght.pose.orientation.w = Q_rght[0]
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():

        publisher_lft.publish(pose_lft)
        publisher_rght.publish(pose_rght)
        rate.sleep()


    return True


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

            rviz_window = subprocess.Popen(["roslaunch", "repair_urdf", "repair_full_trgt_poses.launch"])

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
    tf = 5
    n_nodes = 200
    dt = tf / n_nodes
    lbs = kindyn.q_min() 
    ubs = kindyn.q_max()
    
    prb = problem.Problem(n_nodes)
    q = prb.createStateVariable('q', n_q)
    q_dot = prb.createInputVariable('q_dot', n_v)
    q_design = q[1, 2, 3, 2 + (arm_dofs + 2), 3 + (arm_dofs + 2)] # design vector
    
    q_init = np.zeros((n_q, 1)).flatten()
    q_init[1] = 0.6
    q_aux = np.array([0, 0.6, 0.3, 0, -1, -1, -0.7, -0.2, -1.5, 0.6, 0.4, 0.3, 0, -0.9, -0.8, 0, -1, 1, -0.8, -2])
    
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
    # prb.createConstraint("fixed_x", q[0])

    # roll and shouldeer vars equal
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

    # prb.createFinalConstraint("final_left_tcp_pos_error", larm_tcp_pos - fk_arm_l(q = q_aux)["ee_pos"])
    # prb.createFinalConstraint("final_right_tcp_pos_error", rarm_tcp_pos - fk_arm_r(q = q_aux)["ee_pos"])

    # COSTS

    # min inputs 

    prb.createIntermediateCost("min_q_dot", 0.01 * cs.sumsqr(q_dot))  # minimizing the joint accelerations ("responsiveness" of the trajectory)
    
    # left arm pose error
    prb.createIntermediateCost("init_left_tcp_pos_error", 10 * cs.sumsqr(larm_tcp_pos - fk_arm_l(q = q_init)["ee_pos"]), nodes = 0)
    prb.createFinalCost("final_left_tcp_pos_error", 1000 * cs.sumsqr(larm_tcp_pos - fk_arm_l(q = q_aux)["ee_pos"]))
    prb.createFinalCost("final_left_tcp_rot_error", 1 * rot_error(fk_arm_l(q = q_aux)["ee_rot"], larm_tcp_rot))

    # right arm pose error
    prb.createIntermediateCost("init_right_tcp_pos_error", 10 * cs.sumsqr(rarm_tcp_pos - fk_arm_r(q = q_init)["ee_pos"]), nodes = 0)
    prb.createFinalCost("final_right_tcp_pos_error",  1000 * cs.sumsqr(rarm_tcp_pos - fk_arm_r(q = q_aux)["ee_pos"]))
    prb.createFinalCost("final_right_tcp_rot_error", 1 * rot_error(fk_arm_r(q = q_aux)["ee_rot"], rarm_tcp_rot))

    ## Creating the solver and solving the problem
    slvr = solver.Solver.make_solver(solver_type, prb, slvr_opt) 
    t = time.time()
    slvr.solve()  # solving
    solution_time = time.time() - t
    print(f'solved in {solution_time} s')
    solution = slvr.getSolutionDict() # extracting solution
    cnstr_opt = slvr.getConstraintSolutionDict()

    q_sol = solution["q"]
    
    tcp_pos = {"rTCP_pos": fk_arm_r(q = q_sol)["ee_pos"].toarray() , "lTCP_pos": fk_arm_l(q = q_sol)["ee_pos"].toarray() }
    tcp_rot = {"rTCP_rot": fk_arm_r(q = q_sol)["ee_rot"].toarray() , "lTCP_rot": fk_arm_l(q = q_sol)["ee_rot"].toarray() }
    tcp_pos_trgt = {"rTCP_trgt_pos": fk_arm_r(q = q_aux)["ee_pos"].toarray() , "lTCP_trgt_pos": fk_arm_l(q = q_aux)["ee_pos"].toarray() }
    tcp_rot_trgt = {"rTCP_trgt_rot": fk_arm_r(q = q_aux)["ee_rot"].toarray() , "lTCP_trgt_rot": fk_arm_l(q = q_aux)["ee_pos"].toarray() }

    ms.store({**solution, **cnstr_opt, **tcp_pos, **tcp_rot, **tcp_pos_trgt, **tcp_rot_trgt})

    if args.rviz_replay and args.launch_rviz:

        process = multiprocessing.Process(target = trgt_pose_publisher, args=(fk_arm_l(q = q_aux)["ee_pos"], fk_arm_r(q = q_aux)["ee_pos"], fk_arm_l(q = q_aux)["ee_rot"], fk_arm_r(q = q_aux)["ee_rot"]))
        process.start()

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