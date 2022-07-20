# Defaults values used across multiple scripts

epsi_default = 0.000001

def get_design_map():

    # THIS DEFINITIONS CAN CHANGE IF THE URDF CHANGES --> MIND THE URDF!!!
    arm_dofs = 7
    d_var_map = {}
    d_var_map["mount_h"] = 1
    d_var_map["should_wl"] = 2
    d_var_map["should_roll_l"] = 3
    d_var_map["wrist_off_l"] = 3 + arm_dofs
    d_var_map["should_wr"] = \
        d_var_map["should_wl"] + (arm_dofs + 3)
    d_var_map["should_roll_r"] = \
        d_var_map["should_roll_l"] + (arm_dofs + 3)
    d_var_map["wrist_off_r"] = \
        d_var_map["wrist_off_l"] + (arm_dofs + 3)

    return d_var_map