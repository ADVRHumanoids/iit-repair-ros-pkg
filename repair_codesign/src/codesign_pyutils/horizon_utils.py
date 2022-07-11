import numpy as np

import casadi as cs

from codesign_pyutils.math_utils import quat2rot, rot_error, rot_error2, get_cocktail_aux_rot

from horizon import problem

from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn

def add_bartender_cnstrnt(index, prb, nodes, posl, posr, rotl, rotr, is_pos = True, is_rot = True, weight_pos = 1.0, weight_rot = 1.0, is_soft = False, epsi = 0.001):

    if not is_soft:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_cnstrnt = prb.createConstraint("keep_baretender_pos" + str(index), posr - posl, nodes = nodes)

        if is_rot:

            rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index), rot_error2( get_cocktail_aux_rot(rotr), rotl, epsi), nodes = nodes)
    
    else:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:
            
            pos_cnstrnt = prb.createIntermediateCost("keep_baretender_pos" + str(index), weight_pos * cs.sumsqr(posr - posl), nodes = nodes)

        if is_rot:
            
            rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot" + str(index), weight_rot * cs.sumsqr(rot_error2( get_cocktail_aux_rot(rotr), rotl, epsi)), nodes = nodes)

    return pos_cnstrnt, rot_cnstrnt

def add_pose_cnstrnt(index, prb, nodes, pos, rot, pos_ref, rot_ref, weight_pos = 1.0, weight_rot = 1.0, is_pos = True, is_rot = True, is_soft = False, epsi = 0.001):

  if not is_soft:
  
        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_cnstrnt = prb.createConstraint("pos" + str(index), pos - pos_ref, nodes = nodes)

        if is_rot:

            rot_cnstrnt = prb.createConstraint("rot" + str(index), rot_error2(rot, rot_ref, epsi), nodes = nodes)

  else:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_cnstrnt = prb.createIntermediateCost("pos_soft" + str(index), weight_pos * cs.sumsqr(pos - pos_ref), nodes = nodes)

        if is_rot:

            rot_cnstrnt = prb.createIntermediateCost("rot_soft" + str(index), weight_rot * cs.sumsqr(rot_error2(rot, rot_ref, epsi)), nodes = nodes)

  return pos_cnstrnt, rot_cnstrnt


class FlippingTaskGen:

    def __init__(self, ):
        
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

        self.cocktail_size = 0.05
        self.arm_dofs = 7

        self.task_base_n_nodes = 6
        self.phase_number = self.task_base_n_nodes - 1 # number of phases of the task
        self.total_nnodes = 0
        self.nodes_list = [] # used to iteratre through flipping task and nodes of each defined task
        self.final_node = None

        self.dt = 0.0
        self.tf = 0.0

        self.prb = None

        self.rght_arm_picks = []
        self.contact_heights = []

        self.lft_pick_pos = []
        self.lft_pick_q = []
        self.rght_pick_pos = []
        self.rght_pick_q = []

        self.lft_inward_q = np.array([- np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0])
        self.rght_inward_q = np.array([- np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0, 0.0])

        self.lft_pick_pos_wrt_ws_default = np.array([0, - 0.18, 0.04])
        self.lft_pick_q_wrt_ws_default = np.array([0.0, 1.0, 0.0, 0.0])
        self.rght_pick_pos_wrt_ws_default = np.array([0, 0.18, 0.04])
        self.rght_pick_q_wrt_ws_default = np.array([0.0, 1.0, 0.0, 0.0])

        self.contact_height_default = 0.4

    def get_main_nodes_offset(self, total_task_nnodes):
    
        # simple method to get the offset of the main nodes of a single flipping task (if )
    
        main_nodes_offset = int((total_task_nnodes - self.task_base_n_nodes) / (self.task_base_n_nodes - 1)) + 1

        return main_nodes_offset

    def add_task(self, init_node, filling_n_nodes = 0, right_arm_picks = True, lft_pick_pos_wrt_ws = None, lft_pick_q_wrt_ws = None, rght_pick_pos_wrt_ws = None, rght_pick_q_wrt_ws = None, contact_height = None):
   
        total_n_nodes = self.phase_number * filling_n_nodes + self.task_base_n_nodes # total number of nodes required to perform the task
        final_node = init_node + total_n_nodes - 1 # node index at the end of the task

        nodes_array = list(range(init_node, final_node + 1))

        self.nodes_list.append(nodes_array)

        self.total_nnodes = self.total_nnodes + total_n_nodes

        if lft_pick_pos_wrt_ws == None:
            self.lft_pick_pos.append(self.lft_pick_pos_wrt_ws_default)
        else:
            self.lft_pick_pos.append(lft_pick_pos_wrt_ws)
        
        if lft_pick_q_wrt_ws == None:
            self.lft_pick_q.append(self.lft_pick_q_wrt_ws_default)
        else:
            self.lft_pick_q.append(lft_pick_q_wrt_ws)
        
        if rght_pick_pos_wrt_ws == None:
            self.rght_pick_pos.append(self.rght_pick_pos_wrt_ws_default)
        else:
            self.rght_pick_pos.append(rght_pick_pos_wrt_ws)
        
        if rght_pick_q_wrt_ws == None:
            self.rght_pick_q.append(self.rght_pick_q_wrt_ws_default)
        else:
            self.rght_pick_q.append(rght_pick_q_wrt_ws)  
        

        if contact_height == None:
            self.contact_heights.append(self.contact_height_default)
        else:
            self.contact_heights.append(contact_height)
        
        self.rght_arm_picks.append(right_arm_picks)

        return final_node

    def init_prb(self, urdf_full_path, weight_pos = 1.0, weight_rot = 1.0, weight_glob_man = 0.1, is_soft_pose_cnstr = True, epsi = 0.0, tf = 10):
        
        self.weight_pos = weight_pos
        self.weight_rot = weight_rot
        self.weight_glob_man = weight_glob_man

        n_int = (self.total_nnodes - 1)
        self.prb = problem.Problem(n_int) 

        self.urdf = open(urdf_full_path, 'r').read()
        self.kindyn = cas_kin_dyn.CasadiKinDyn(self.urdf)
        self.tf = tf
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
        q_design = self.q[1, 2, 3, 2 + (self.arm_dofs + 2), 3 + (self.arm_dofs + 2)] # design vector

        self.prb.setDynamics(self.q_dot)
        self.prb.setDt(self.dt)  

        fk_ws = cs.Function.deserialize(self.kindyn.fk("working_surface_link"))
        ws_link_pos = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_pos"] # w.r.t. world
        ws_link_rot = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)

        fk_arm_r = cs.Function.deserialize(self.kindyn.fk("arm_1_tcp")) 
        rarm_tcp_pos = fk_arm_r(q = self.q)["ee_pos"] # w.r.t. world
        rarm_tcp_rot = fk_arm_r(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        rarm_tcp_pos_wrt_ws = rarm_tcp_pos - ws_link_pos # pose w.r.t. working surface

        fk_arm_l = cs.Function.deserialize(self.kindyn.fk("arm_2_tcp"))  
        larm_tcp_pos = fk_arm_l(q = self.q)["ee_pos"] # w.r.t. world
        larm_tcp_rot = fk_arm_l(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        larm_tcp_pos_wrt_ws = larm_tcp_pos - ws_link_pos # pose w.r.t. working surface

        rarm_cocktail_pos = rarm_tcp_pos + rarm_tcp_rot @ cs.vertcat(0, 0, self.cocktail_size / 2.0)
        rarm_cocktail_rot = rarm_tcp_rot
        rarm_cocktail_rot_wrt_ws = cs.transpose(ws_link_rot) @ rarm_cocktail_rot
        rarm_cocktail_pos_wrt_ws = ws_link_rot @ (rarm_cocktail_pos - ws_link_pos)

        larm_cocktail_pos = larm_tcp_pos + larm_tcp_rot @ cs.vertcat(0, 0, self.cocktail_size / 2.0)
        larm_cocktail_rot = larm_tcp_rot
        larm_cocktail_rot_wrt_ws = cs.transpose(ws_link_rot) @ larm_cocktail_rot
        larm_cocktail_pos_wrt_ws = ws_link_rot @ (larm_cocktail_pos - ws_link_pos)

        self.lft_tcp_pos_wrt_ws = larm_cocktail_pos_wrt_ws
        self.lft_tcp_rot_wrt_ws = larm_cocktail_rot_wrt_ws
        self.rght_tcp_pos_wrt_ws = rarm_cocktail_pos_wrt_ws
        self.rght_tcp_rot_wrt_ws = rarm_cocktail_rot_wrt_ws

        # roll and shoulder vars equal
        self.prb.createConstraint("same_roll", self.q[3] - self.q[3 + (self.arm_dofs + 2)])
        self.prb.createConstraint("same_shoulder_w", self.q[2] - self.q[2 + (self.arm_dofs + 2)])

        # design vars equal on all nodes 
        self.prb.createConstraint("single_var_cnstrnt", q_design - q_design.getVarOffset(-1), nodes = range(1, n_int + 1))

        # lower and upper bounds for design variables and joint variables
        self.q.setBounds(self.lbs, self.ubs)

        # TCPs above working surface
        keep_tcp1_above_ground = self.prb.createConstraint("keep_tcp1_above_ground", rarm_tcp_pos_wrt_ws[2])
        keep_tcp1_above_ground.setBounds(0, cs.inf)
        keep_tcp2_above_ground = self.prb.createConstraint("keep_tcp2_above_ground", larm_tcp_pos_wrt_ws[2])
        keep_tcp2_above_ground.setBounds(0, cs.inf)

        # min inputs 

        self.prb.createIntermediateCost("max_global_manipulability", weight_glob_man * cs.sumsqr(self.q_dot))  # minimizing the joint accelerations ("responsiveness" of the trajectory)

        self.build_task(is_soft_pose_cnstr = is_soft_pose_cnstr, epsi = epsi)

    def build_task(self, is_soft_pose_cnstr = True, epsi = 0.00001):

        for i in range(len(self.nodes_list)): # iterate through multiple flipping tasks
            
            delta_offset = self.get_main_nodes_offset(len(self.nodes_list[i])) # = 1 if no intermediate node between base nodes was inserted

            for j in range(self.task_base_n_nodes): # for each task, iterate through each base node (!= task total number of nodes)
                
                # hand-crafted state machine (TB improved in the future)

                if j == 0: # ARM 1: picking pose | ARM 2: waiting pose
                    
                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i], quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i], quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 1: # ARM 1: uplift pose | ARM 2: inward rotation pose

                    if (self.rght_arm_picks[i]): # right arm picks

                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_inward_q), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_inward_q), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)

                if j == 2: # ARM 1: inward rotation pose | ARM 2: approach pose ----> contact ----> baretender constraint
                    
                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_inward_q), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = epsi)
                        # left arm
                        add_bartender_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.rght_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.rght_tcp_rot_wrt_ws, is_pos = True, is_rot = True, weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_soft = is_soft_pose_cnstr,  epsi = epsi)

                    else: # left arm picks
                        
                        # right arm
                        add_bartender_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.rght_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.rght_tcp_rot_wrt_ws, is_pos = True, is_rot = True, weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_soft = is_soft_pose_cnstr,  epsi = epsi)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_inward_q), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                if j == 3: # ARM 1: back to picking pose | ARM 2: back to inward rotation pose
                    
                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_inward_q), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_inward_q), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i], quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                if j == 4: # ARM 1: still at picking pose | ARM 2: back to waiting pose

                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i], quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                if j == 5: # ARM 1: still at picking pose | ARM 2: down to picking  pose

                    if (self.rght_arm_picks[i]): # right arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i], quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

                    else: # left arm picks
                        
                        # right arm
                        add_pose_cnstrnt(i + j, self.prb, self.nodes_list[i][j * delta_offset], self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws, self.rght_pick_pos[i] + np.array([0, 0, self.contact_heights[i]]), quat2rot(self.rght_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)
                        # left arm
                        add_pose_cnstrnt(i + j + self.task_base_n_nodes * len(self.nodes_list), self.prb, self.nodes_list[i][j * delta_offset], self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws, self.lft_pick_pos[i], quat2rot(self.lft_pick_q[i]), weight_pos = self.weight_pos, weight_rot = self.weight_rot, is_pos = True, is_rot = True, is_soft = is_soft_pose_cnstr, epsi = 0.00001)

        

        


    
