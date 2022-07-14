import numpy as np

import casadi as cs

from codesign_pyutils.math_utils import quat2rot, rot_error, rot_error2, rot_error3, get_cocktail_matching_rot

from codesign_pyutils.miscell_utils import check_str_list, wait_for_confirmation

from horizon import problem

from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn

import time

import warnings

rot_error_approach = "traversaro" # options: "siciliano", "arturo", "traversaro"

###############
# obsolete
def add_bartender_cnstrnt(index, prb, nodes, posl, posr, rotl, rotr, is_pos = True,\
                          is_rot = True, weight_pos = 1.0, weight_rot = 1.0,\
                          is_soft = False, epsi = 0.001):

    if not is_soft:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_cnstrnt = prb.createConstraint("keep_baretender_pos" + str(index),\
                                                posr - posl, nodes = nodes)

        if is_rot:
            
            if rot_error_approach == "siciliano":

                rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index),\
                                    rot_error( get_cocktail_matching_rot(rotr), rotl, epsi),\
                                    nodes = nodes)

            if rot_error_approach == "arturo":
                
                rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index),\
                                    rot_error2( get_cocktail_matching_rot(rotr), rotl, epsi),\
                                    nodes = nodes)
            
            if rot_error_approach == "traversaro":

                rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index),\
                                    rot_error3( get_cocktail_matching_rot(rotr), rotl, epsi),\
                                    nodes = nodes)

            if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
                and rot_error_approach != "siciliano":

                raise Exception('Choose a valid rotation error computation approach')

    
    else:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:
            
            pos_cnstrnt = prb.createIntermediateCost("keep_baretender_pos_soft" + str(index),\
                                                     weight_pos * cs.sumsqr(posr - posl),
                                                     nodes = nodes)

        if is_rot:
            
            if rot_error_approach == "siciliano":
                
                rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot_soft" + str(index),\
                    weight_rot * cs.sumsqr(rot_error( get_cocktail_matching_rot(rotr), rotl, epsi)),\
                    nodes = nodes)

            if rot_error_approach == "arturo":
                
                rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot_soft" + str(index),\
                    weight_rot * cs.sumsqr(rot_error2( get_cocktail_matching_rot(rotr), rotl, epsi)),\
                    nodes = nodes)
            
            if rot_error_approach == "traversaro":

                rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot_soft" + str(index),\
                    weight_rot * (rot_error3( get_cocktail_matching_rot(rotr), rotl, epsi)),\
                    nodes = nodes)

            if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
                 and rot_error_approach != "siciliano":

                raise Exception('Choose a valid rotation error computation approach')
                

    return pos_cnstrnt, rot_cnstrnt

#############

def add_pose_cnstrnt(index, prb, nodes, pos = None, rot = None, pos_ref = None, rot_ref = None,\
                     pos_selection = ["x", "y", "z"], rot_selection = ["x", "y", "z"],\
                     weight_pos = 1.0, weight_rot = 1.0, is_soft = False, epsi = 0.001):

    if ((pos is None) or (pos_ref is None)) and ( (rot is None) or (rot_ref is None) ):

        raise Exception("Missing necessary arguments for add_pose_cnstrnt()!!")

    # now it is guaranteed that at least one pair between [pos, pos_ref] and [rot, rot_ref] was provided
    is_pos = False
    is_rot = True

    if pos is not None:

        is_pos = True

    if rot is not None:

        is_rot = True

    if not is_soft:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_selector = check_str_list(comp_list = ["x", "y", "z"], input = pos_selection)
            pos_cnstrnt = prb.createConstraint("pos" + str(index),\
                                                pos[pos_selector] - pos_ref[pos_selector],\
                                                nodes = nodes)

        if is_rot:
            
            if rot_error_approach == "siciliano":
                
                rot_cnstrnt = prb.createConstraint("rot" + str(index),\
                                    rot_error(rot_ref, rot, epsi), nodes = nodes)

            if rot_error_approach == "arturo":
                
                rot_cnstrnt = prb.createConstraint("rot" + str(index),\
                                    rot_error2(rot_ref, rot, epsi), nodes = nodes)
            
            if rot_error_approach == "traversaro":

                rot_cnstrnt = prb.createConstraint("rot" + str(index),\
                                    rot_error3(rot_ref, rot, epsi), nodes = nodes)

            if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
                and rot_error_approach != "siciliano":

                raise Exception('Choose a valid rotation error computation approach')
            
    else:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_selector = check_str_list(comp_list = ["x", "y", "z"], input = pos_selection)
            
            if len(pos[pos_selector]) != 0: # necessary only if using soft constraints
            
                if len(pos[pos_selector]) == 1: # do not use cs.sumsqr
 
                    pos_cnstrnt = prb.createIntermediateCost("pos_soft" + str(index),\
                                                    weight_pos * cs.power(pos[pos_selector] - pos_ref[pos_selector], 2),\
                                                    nodes = nodes)

                else:

                    pos_cnstrnt = prb.createIntermediateCost("pos_soft" + str(index),\
                                                    weight_pos * cs.sumsqr(pos[pos_selector] - pos_ref[pos_selector]),\
                                                    nodes = nodes)

        if is_rot:
            
            if rot_error_approach == "siciliano":
                
                rot_cnstrnt = prb.createIntermediateCost("rot_soft" + str(index),\
                            weight_rot * cs.sumsqr(rot_error(rot_ref, rot, epsi)),\
                            nodes = nodes)

            if rot_error_approach == "arturo":
                
                rot_cnstrnt = prb.createIntermediateCost("rot_soft" + str(index),\
                            weight_rot * cs.sumsqr(rot_error2(rot_ref, rot, epsi)),\
                            nodes = nodes)
            
            if rot_error_approach == "traversaro":

                rot_cnstrnt = prb.createIntermediateCost("rot_soft" + str(index),\
                            weight_rot * (rot_error3(rot_ref, rot, epsi)),\
                            nodes = nodes)

            if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
                and rot_error_approach != "siciliano":

                raise Exception('Choose a valid rotation error computation approach')
      

    return pos_cnstrnt, rot_cnstrnt

def add_p2p_collision_task(index, prb, nodes, pos1, pos2, epsi = 0.001):

    # collision task between two points

    return False

class FlippingTaskGen:

    def __init__(self, cocktail_size = 0.05, filling_n_nodes = 0):
        
        self.weight_pos = 0
        self.weight_rot = 0
        self.weight_glob_man = 0

        self.urdf = None
        self.joint_names = None
        self.nq = 0
        self.nv = 0
        self.kindyn = None
        self.lbs = None
        self.ubs = None

        self.q = None
        self.q_dot = None
        self.q_design = None
        self.q_design_dot = None

        self.cocktail_size = cocktail_size
        self.arm_dofs = 7

        self.task_base_n_nodes = 0
        self.phase_number = 0 # number of phases of the task
        self.total_nnodes = 0
        self.filling_n_nodes = filling_n_nodes # filling nodes in between two base task nodes
        self.nodes_list = [] # used to iteratre through flipping task and nodes of each defined task
        self.final_node = None
        self.n_of_tasks = 0

        self.dt = 0.0
        self.tf = 0.0

        self.prb = None

        self.rght_arm_picks = []
        self.contact_heights = []
        self.hor_offsets = []

        # object TCP picking orientation (position == position of the object)
        self.lft_pick_q = []
        self.rght_pick_q = []

        self.object_pos_lft = [] # object pose on working surface
        self.object_q_lft = []
        self.object_pos_rght = []
        self.object_q_rght = []

        self.lft_inward_q = np.array([- np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0])
        self.rght_inward_q = np.array([- np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0, 0.0])

        self.in_place_flip = True

        self.was_init_called = False

        self.available_task_stack = ["in_place_flip", "pick_and_place"]
        self.employed_task = ""

        self.coll_links_pos_lft = []
        self.coll_links_pos_rght = []

    def get_main_nodes_offset(self, total_task_nnodes):
    
        # simple method to get the offset of the main nodes of a single flipping task
    
        main_nodes_offset = int((total_task_nnodes - self.task_base_n_nodes)\
                               /(self.task_base_n_nodes - 1)) + 1

        return main_nodes_offset

    def compute_nodes(self, init_node, filling_n_nodes):

        # HOW THE PROBLEM IS BUILT:
        #                    
        #   a b                       c d 
        # |                         |     |                       |
        # |     |     |     |       |     |     |     |     |     |
        # | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 
        # 0     1     2     3       4    1.0  1.1   1.2   1.3   1.4

        # 0 - 1 - 2 - 3 - 4 base task nodes
        # from 1.0 to 1.4 successive task
        # a and b are examples of filling nodes in a task
        # c and d are filling nodes between two successive task

        total_n_task_nodes = self.phase_number * filling_n_nodes + self.task_base_n_nodes # total number of nodes required to perform a single task

        if self.n_of_tasks != 1: # adding additional nodes for the transition between tasks (if filling nodes were added); only from second task

            self.total_nnodes = self.total_nnodes + total_n_task_nodes  + filling_n_nodes

            next_task_node = init_node + total_n_task_nodes  + filling_n_nodes  # from the second task on, filling nodes between tasks are included in the nodes array

            self.nodes_list.append(list(range(init_node + filling_n_nodes, next_task_node + 1)))

        else: # first added task
            
            self.total_nnodes = total_n_task_nodes

            next_task_node = init_node + total_n_task_nodes  
            
            self.nodes_list.append(list(range(init_node, next_task_node + 1)))
        
        return next_task_node

    def add_pick_and_place_task(self, init_node,\
                                right_arm_picks = True,\
                                object_pos_lft_wrt_ws = np.array([0, 0.18, 0.04]),\
                                object_q_lft_wrt_ws = np.array([0.0, 1.0, 0.0, 0.0]),\
                                object_pos_rght_wrt_ws = np.array([0, - 0.18, 0.04]),\
                                object_q_rght_wrt_ws = np.array([0.0, 1.0, 0.0, 0.0]),\
                                lft_pick_q_wrt_ws = np.array([0.0, 1.0, 0.0, 0.0]),\
                                rght_pick_q_wrt_ws = np.array([0.0, 1.0, 0.0, 0.0]),\
                                contact_height = 0.4):

        if self.was_init_called :

            raise Exception("You can only add tasks before calling init_prb(*)!!")

        if self.employed_task != "pick_and_place" and self.employed_task != "":
            
            raise Exception("Adding multiple heterogeneous tasks is not supported yet.")

        self.employed_task = "pick_and_place"

        self.n_of_tasks = self.self.n_of_tasks + 1 # counter for the number of tasks

        self.in_place_flip = False
        
        self.task_base_n_nodes = 6
        self.phase_number = self.task_base_n_nodes - 1 # number of phases of the task

        final_node = self.compute_nodes(init_node, self.filling_n_nodes)

        self.object_pos_lft.append(object_pos_lft_wrt_ws)
        self.object_q_lft.append(object_q_lft_wrt_ws)
        self.object_pos_rght.append(object_pos_rght_wrt_ws)
        self.object_q_rght.append(object_q_rght_wrt_ws)  
        
        self.lft_pick_q.append(lft_pick_q_wrt_ws)
        self.rght_pick_q.append(rght_pick_q_wrt_ws)

        self.contact_heights.append(contact_height)
        
        self.rght_arm_picks.append(right_arm_picks)

        return final_node

    def add_in_place_flip_task(self, init_node,\
                               right_arm_picks = True,\
                               object_pos_wrt_ws = np.array([0.0, 0.0, 0.0]),\
                               object_q_wrt_ws = np.array([0.0, 1.0, 0.0, 0.0]),\
                               pick_q_wrt_ws = np.array([0.0, 1.0, 0.0, 0.0]),\
                               contact_height = 0.4, hor_offset = 0.2):

        if self.was_init_called :

            raise Exception("You can only add tasks before calling init_prb(*)!!")

        if self.employed_task != "in_place_flip" and self.employed_task != "":
            
            raise Exception("Adding multiple heterogeneous tasks is not supported yet.")

        self.n_of_tasks = self.n_of_tasks + 1 # counter for the number of tasks

        self.employed_task = "in_place_flip"

        self.in_place_flip = True

        self.task_base_n_nodes = 9
        self.phase_number = self.task_base_n_nodes - 1 # number of phases of the task

        self.object_pos_lft.append(object_pos_wrt_ws)
        self.object_q_lft.append(object_q_wrt_ws)
        self.object_pos_rght.append(object_pos_wrt_ws)
        self.object_q_rght.append(object_q_wrt_ws)

        self.lft_pick_q.append(pick_q_wrt_ws)
        self.rght_pick_q.append(pick_q_wrt_ws)

        self.contact_heights.append(contact_height)
        self.hor_offsets.append(hor_offset)
        self.rght_arm_picks.append(right_arm_picks)

        next_task_node = self.compute_nodes(init_node, self.filling_n_nodes)

        return next_task_node

    def init_prb(self, urdf_full_path, weight_pos = 0.001, weight_rot = 0.001,\
                 weight_glob_man = 0.0001, is_soft_pose_cnstr = True,\
                 epsi = 0.0, tf_single_task = 10):
        
        self.was_init_called  = True

        self.weight_pos = weight_pos / ( self.total_nnodes ) # scaling weights on the basis of the problem dimension
        self.weight_rot = weight_rot / ( self.total_nnodes )
        self.weight_glob_man = weight_glob_man / ( self.total_nnodes )

        n_int = self.total_nnodes - 1 # adding addditional filling nodes between nodes of two successive tasks
        self.prb = problem.Problem(n_int) 

        self.urdf = open(urdf_full_path, 'r').read()
        self.kindyn = cas_kin_dyn.CasadiKinDyn(self.urdf)
        self.tf = tf_single_task * len(self.nodes_list)
        self.dt = self.tf / n_int
        
        self.joint_names = self.kindyn.joint_names()
        if 'universe' in self.joint_names: self.joint_names.remove('universe')
        if 'floating_base_joint' in self.joint_names: self.joint_names.remove('floating_base_joint')
        
        self.nq = self.kindyn.nq()
        self.nv = self.kindyn.nv()

        self.lbs = self.kindyn.q_min() 
        self.ubs = self.kindyn.q_max()

        self.q = self.prb.createStateVariable('q', self.nq)
        self.q_dot = self.prb.createInputVariable('q_dot', self.nv)
        self.q_design = self.q[1, 2, 3, 2 + (self.arm_dofs + 2), 3 + (self.arm_dofs + 2)] # design vector
        self.q_design_dot = self.q[1, 2, 3, 2 + (self.arm_dofs + 2), 3 + (self.arm_dofs + 2)] # design vector

        self.prb.setDynamics(self.q_dot)
        self.prb.setDt(self.dt)  

        # getting useful kinematic quantities
        fk_ws = cs.Function.deserialize(self.kindyn.fk("working_surface_link"))
        ws_link_pos = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_pos"] # w.r.t. world
        ws_link_rot = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)

        fk_ws_arm1_link6 = cs.Function.deserialize(self.kindyn.fk("arm_1_link_6"))
        arm1_link6_pos = fk_ws_arm1_link6(q = self.q)["ee_pos"] # w.r.t. world
        arm1_link6_rot = fk_ws_arm1_link6(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        
        fk_ws_arm2_link6 = cs.Function.deserialize(self.kindyn.fk("arm_2_link_6"))
        arm2_link6_pos = fk_ws_arm2_link6(q = self.q)["ee_pos"] # w.r.t. world
        arm2_link6_rot = fk_ws_arm2_link6(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)

        fk_arm_r = cs.Function.deserialize(self.kindyn.fk("arm_1_tcp")) 
        rarm_tcp_pos = fk_arm_r(q = self.q)["ee_pos"] # w.r.t. world
        rarm_tcp_rot = fk_arm_r(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        rarm_tcp_pos_wrt_ws = rarm_tcp_pos - ws_link_pos # pos w.r.t. working surface in world frame

        fk_arm_l = cs.Function.deserialize(self.kindyn.fk("arm_2_tcp"))  
        larm_tcp_pos = fk_arm_l(q = self.q)["ee_pos"] # w.r.t. world
        larm_tcp_rot = fk_arm_l(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        larm_tcp_pos_wrt_ws = larm_tcp_pos - ws_link_pos # pos w.r.t. working surface in world frame

        rarm_cocktail_pos = rarm_tcp_pos + \
                            rarm_tcp_rot @ cs.vertcat(0, 0, self.cocktail_size / 2.0)
        rarm_cocktail_rot = rarm_tcp_rot
        rarm_cocktail_rot_wrt_ws = ws_link_rot.T @ rarm_cocktail_rot
        rarm_cocktail_pos_wrt_ws = ws_link_rot.T @ (rarm_cocktail_pos - ws_link_pos)

        larm_cocktail_pos = larm_tcp_pos + \
                            larm_tcp_rot @ cs.vertcat(0, 0, self.cocktail_size / 2.0)
        larm_cocktail_rot = larm_tcp_rot
        larm_cocktail_rot_wrt_ws = ws_link_rot.T @ larm_cocktail_rot
        larm_cocktail_pos_wrt_ws = ws_link_rot.T @ (larm_cocktail_pos - ws_link_pos)

        self.lft_tcp_pos_wrt_ws = larm_cocktail_pos_wrt_ws
        self.lft_tcp_rot_wrt_ws = larm_cocktail_rot_wrt_ws
        self.rght_tcp_pos_wrt_ws = rarm_cocktail_pos_wrt_ws
        self.rght_tcp_rot_wrt_ws = rarm_cocktail_rot_wrt_ws

        self.coll_links_pos_rght = ws_link_rot.T @ (arm1_link6_pos - ws_link_pos)
        self.coll_links_pos_lft = ws_link_rot.T @ (arm2_link6_pos - ws_link_pos)

        # roll and shoulder vars equal
        self.prb.createConstraint("same_roll", \
                                  self.q[3] - self.q[3 + (self.arm_dofs + 2)])
        self.prb.createConstraint("same_shoulder_w",\
                                  self.q[2] - self.q[2 + (self.arm_dofs + 2)])

        # design vars equal on all nodes 
        self.prb.createConstraint("single_var_cntrnt",\
            self.q_dot[1, 2, 3, 2 + (self.arm_dofs + 2), 3 + (self.arm_dofs + 2)],\
            nodes = range(0, (self.total_nnodes - 1)))

        # lower and upper bounds for design variables and joint variables
        self.q.setBounds(self.lbs, self.ubs)

        # TCPs above working surface
        keep_tcp1_above_ground = self.prb.createConstraint("keep_tcp1_above_ground",\
                                                           rarm_tcp_pos_wrt_ws[2])
        keep_tcp1_above_ground.setBounds(0, cs.inf)
        keep_tcp2_above_ground = self.prb.createConstraint("keep_tcp2_above_ground",\
                                                           larm_tcp_pos_wrt_ws[2])
        keep_tcp2_above_ground.setBounds(0, cs.inf)

        # min inputs 
        self.prb.createIntermediateCost("max_global_manipulability",\
                        weight_glob_man * cs.sumsqr(self.q_dot)) # minimizing the joint accelerations ("responsiveness" of the trajectory)

        # add p2p collision task on y axis
        coll = self.prb.createConstraint("avoid_collision_on_y", \
                                self.coll_links_pos_lft[1] - self.coll_links_pos_rght[1])
        coll.setBounds(self.hor_offsets[0] - 0.001, cs.inf)

        # here the "custom" task is added to the problem
        self.build_task(is_soft_pose_cnstr = is_soft_pose_cnstr, epsi = epsi,\
                        in_place_flip = self.in_place_flip)

    def build_in_place_flip_task(self, is_soft_pose_cnstr = True, epsi = 0.00001):

        for i in range(len(self.nodes_list)): # iterate through multiple flipping tasks
            
            delta_offset = self.get_main_nodes_offset(len(self.nodes_list[i])) # = 1 if no intermediate node between base nodes was inserted

            for j in range(self.task_base_n_nodes): # for each task, iterate through each base node (!= task total number of nodes)
                
                # hand-crafted state machine (TB improved in the future)

                constraint_unique_id_rght = j + 2 * self.task_base_n_nodes * i # index used to give different names to each constraint (rght arm)
                constraint_unique_id_lft = j + self.task_base_n_nodes + 2 * self.task_base_n_nodes * i # index used to give different names to each constraint (lft arm)
                
                cnstrnt_node_index = self.nodes_list[i][j * delta_offset] # index of the constraint (traslated by the number of filling nodes)

                if j == 0: # ARM 1: waiting pose | ARM 2: waiting pose
                                        
                    if (self.rght_arm_picks[i]): # right arm picks
                        

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        
                        
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                        
                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 1: # ARM 1: goes to pick position | ARM 2: waiting pose

                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i], quat2rot(self.rght_pick_q[i]),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index,\
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i], quat2rot(self.lft_pick_q[i]),
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 2: # ARM 1: goes up from the picking pose | ARM 2: waits
                    
                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 3: # ARM 1: inward rotation from contact height | ARM 2: waits
                    
                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index,  self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.lft_inward_q),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 4: # ARM 1: waits for contact | ARM 2: rotates inward
                    
                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.lft_inward_q),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index,  self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        pos_selection = ["x", "y", "z"],\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
                                        self.object_pos_lft[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.lft_inward_q),
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                                         
                if j == 5: # ARM 1: waits for contact | ARM 2: makes contact with bartender constraint
                    
                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        
                        # relative constraint
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index,\
                                        pos = self.lft_tcp_pos_wrt_ws, rot = self.lft_tcp_rot_wrt_ws,
                                        pos_ref = self.rght_tcp_pos_wrt_ws, rot_ref = get_cocktail_matching_rot(self.rght_tcp_rot_wrt_ws),
                                        pos_selection = ["x", "y", "z"],\
                                        weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # # relative constraint
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index,\
                                        pos = self.lft_tcp_pos_wrt_ws, rot = self.lft_tcp_rot_wrt_ws,\
                                        pos_ref = self.rght_tcp_pos_wrt_ws, rot_ref = get_cocktail_matching_rot(self.rght_tcp_rot_wrt_ws),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.lft_inward_q),    
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 6: # ARM 1: back to waiting pose but rotated inward| ARM 2: rotates back to waiting orientation

                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index,\
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, 0.0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.lft_inward_q),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 7: # ARM 1: back to waiting pose | ARM 2: down to picking  pose

                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i], quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i], quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "y", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 8: # ARM 1: waits | ARM 2: back to waiting pose

                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index,\
                                        self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                        self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                        pos_selection = ["x", "z"],\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                        is_soft = is_soft_pose_cnstr, epsi = epsi)

    def build_pick_and_place_task(self, is_soft_pose_cnstr = True, epsi = 0.00001):

        for i in range(len(self.nodes_list)): # iterate through multiple flipping tasks
            
            delta_offset = self.get_main_nodes_offset(len(self.nodes_list[i])) # = 1 if no intermediate node between base nodes was inserted

            for j in range(self.task_base_n_nodes): # for each task, iterate through each base node (!= task total number of nodes)
                
                # hand-crafted state machine (TB improved in the future)

                cnstrnt_node_index = self.nodes_list[i][j * delta_offset] + i * self.filling_n_nodes # index of the constraint (traslated by the number of filling nodes)

                constraint_unique_id_rght = j + 2 * self.task_base_n_nodes * i # index used to give different names to each constraint (rght arm)
                constraint_unique_id_lft = j + self.task_base_n_nodes + 2 * self.task_base_n_nodes * i # index used to give different names to each constraint (lft arm)

                if j == 0: # ARM 1: picking pose | ARM 2: waiting pose
                    
                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i], quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)

                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft,\
                                         self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i], quat2rot(self.object_q_lft[i]),\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                         is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 1: # ARM 1: uplift pose | ARM 2: inward rotation pose

                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_inward_q),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 2: # ARM 1: inward rotation pose | ARM 2: approach pose ----> contact ----> baretender constraint
                    
                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = epsi)
                        
                        # relative constraint
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index,\
                                         pos = self.lft_tcp_pos_wrt_ws, rot = self.lft_tcp_rot_wrt_ws,
                                         pos_ref = self.rght_tcp_pos_wrt_ws, rot_ref = get_cocktail_matching_rot(self.rght_tcp_rot_wrt_ws),
                                         weight_rot = self.weight_rot,\
                                         is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # relative constraint
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index,\
                                         pos = self.lft_tcp_pos_wrt_ws, rot = self.lft_tcp_rot_wrt_ws,
                                         pos_ref = self.rght_tcp_pos_wrt_ws, rot_ref = get_cocktail_matching_rot(self.rght_tcp_rot_wrt_ws),
                                         weight_rot = self.weight_rot,\
                                         is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_inward_q),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                if j == 3: # ARM 1: back to picking pose | ARM 2: back to inward rotation pose
                    
                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                        self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_inward_q),\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                         is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_inward_q),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.object_pos_lft[i], quat2rot(self.object_q_lft[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True,\
                                         is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                if j == 4: # ARM 1: still at picking pose | ARM 2: back to waiting pose

                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i], quat2rot(self.object_q_lft[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                if j == 5: # ARM 1: still at picking pose | ARM 2: down to picking  pose

                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                        self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.object_pos_lft[i], quat2rot(self.object_q_lft[i]),\
                                        weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                         is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(constraint_unique_id_rght, self.prb, cnstrnt_node_index, \
                                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                         self.object_pos_rght[i], quat2rot(self.object_q_rght[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(constraint_unique_id_lft, self.prb, cnstrnt_node_index, \
                                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                         self.object_pos_lft[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.object_q_lft[i]),\
                                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                          is_soft = is_soft_pose_cnstr, epsi = 0.00001)

        
    def build_task(self, is_soft_pose_cnstr = True, epsi = 0.00001, in_place_flip = True):

            if in_place_flip:

                self.build_in_place_flip_task(is_soft_pose_cnstr, epsi)
            
            else:
                
                self.build_pick_and_place_task(is_soft_pose_cnstr, epsi)
