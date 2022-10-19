# Defaults values used across multiple scripts

import casadi as cs

import numpy as np

import yaml

epsi_default = 0.000001

rot2trasl_man_scl_fact = 5.0 # man_trasl * rot2trasl_man_scl_fact will scale 
# man_trasl so that man_rot and man_trasl have comparable magnitude 

super_high_cost = 1e25

def get_design_map():

    # THIS DEFINITIONS CAN CHANGE IF THE URDF CHANGES --> MIND THE URDF!!!
    arm_dofs = 7
    coll_dof_n = 1
    d_var_map = {}
    d_var_map["mount_h"] = 1
    d_var_map["should_w_l"] = 2
    d_var_map["should_roll_l"] = 3
    d_var_map["wrist_off_l"] = 3 + arm_dofs
    d_var_map["should_w_r"] = \
        d_var_map["should_w_l"] + (arm_dofs + coll_dof_n + 3)
    d_var_map["should_roll_r"] = \
        d_var_map["should_roll_l"] + (arm_dofs + coll_dof_n + 3)
    d_var_map["wrist_off_r"] = \
        d_var_map["wrist_off_l"] + (arm_dofs + coll_dof_n + 3)

    return d_var_map

def get_actuated_jnts_indxs(n_jnts):

    d_var_map = get_design_map()
    coll_var_map = get_coll_joint_map()

    index_array = np.arange(0, n_jnts)

    non_act_jnt_indxs = np.array([d_var_map["mount_h"],\
                        d_var_map["should_w_l"], d_var_map["should_roll_l"], d_var_map["wrist_off_l"], \
                        d_var_map["should_w_r"], d_var_map["should_roll_r"], d_var_map["wrist_off_r"],
                        coll_var_map["link5_coll_joint_l"], coll_var_map["link5_coll_joint_r"]])

    return np.delete(index_array, non_act_jnt_indxs)

def parse_weights_yaml(path):

    yaml_data = None
    
    with open(path, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    torque_weights = np.array(yaml_data["torque"])
    vel_weights = np.array(yaml_data["vel"])

    clman_rot = yaml_data["cl_man"]["rot"]
    clman_trsl = yaml_data["cl_man"]["trasl"]

    return torque_weights, vel_weights, clman_rot, clman_trsl

def get_coll_joint_map():

    # THIS DEFINITIONS CAN CHANGE IF THE URDF CHANGES --> MIND THE URDF!!!
    arm_dofs = 7
    coll_dof_n = 1
    coll_dof_map = {}
    coll_dof_map["link5_coll_joint_l"] = 12 
    coll_dof_map["link5_coll_joint_r"] =  coll_dof_map["link5_coll_joint_l"] + arm_dofs + coll_dof_n + 3

    return coll_dof_map

def get_crossed_handover_local_rot():
    
    frame_rot = cs.DM([[0.0, -1.0, 0.0],\
                        [-1.0, 0.0, 0.0],\
                        [0.0, 0.0, -1.0]]) 
    
    return frame_rot

def get_bimanual_frame_rot():
    
    frame_rot = cs.DM([[1.0, 0.0, 0.0],\
                        [0.0, -1.0, 0.0],\
                        [0.0, 0.0, -1.0]]) 
    
    return frame_rot