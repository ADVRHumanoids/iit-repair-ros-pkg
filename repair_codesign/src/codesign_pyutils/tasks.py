from codesign_pyutils.math_utils import quat2rot

from codesign_pyutils.misc_definitions import epsi_default, get_bimanual_frame_rot,\
                                              rot2trasl_man_scl_fact, get_crossed_handover_local_rot

from codesign_pyutils.horizon_utils import add_pose_cnstrnt, SimpleCollHandler

from codesign_pyutils.ros_utils import MarkerGen, ReplaySol

from codesign_pyutils.misc_definitions import get_design_map, get_coll_joint_map, get_actuated_jnts_indxs,\
                                            parse_weights_yaml

from horizon import problem
from horizon.utils import kin_dyn

import casadi_kin_dyn

import numpy as np

import casadi as cs

import time

from horizon.solvers import solver

from horizon.transcriptions.transcriptor import Transcriptor

class CodesCodesTaskGen:
    
    # single class for generaing the whole co-design problem
    # for he RePAIR co-design

    def __init__(self, filling_n_nodes = 0, \
                is_sliding_wrist = False,\
                sliding_wrist_offset = 0.0, \
                coll_yaml_path = "", 
                cost_weights_yaml_path = ""):
        
        self.coll_yaml_path = coll_yaml_path
        
        self.cost_weights_yaml_path = cost_weights_yaml_path

        self.torque_weights, self.vel_weights,\
            self.weight_clman_rot, self.weight_clman_trasl = parse_weights_yaml(self.cost_weights_yaml_path)

        self.is_sliding_wrist = is_sliding_wrist # whether to add a fictitious d.o.f. on the wrist or not
        self.sliding_wrist_offset =  sliding_wrist_offset # mounting offset for wrist auxiliary joint

        self.is_soft_pose_cnstrnt = False # whether soft constraints are used

        self.weight_pos = 0 # actual weight assigned to the positional error cost (if using soft constraints)
        self.weight_rot = 0 # actual weight assigned to the orientation error cost (if using soft constraints)
        self.weight_glob_man = 0 # actual weight assigned to the global manipulability cost
        self.weight_classical_man = 0 # actual weight assigned to the classical manipulability cost
        self.weight_static_tau = 0 

        self.urdf = None # opened urdf file
        self.joint_names = None # joint names
        self.nq = 0 # number of positional states
        self.nv = 0 # number of velocity  states
        self.urdf_kin_dyn = None # CasadiKinDyn object
        self.lbs = None # holds the lower (positional) bounds on state variables
        self.ubs = None # holds the upper (positional) bounds on state variables

        self.q = None # positional state
        self.q_dot = None # velocity state
        self.q_design = None # positional state of design variables
        self.q_design_dot = None # velocity state of design variables
        self.q_codes_ref = None
        self.wrist_off_ref = None

        self.d_var_map = get_design_map() # retrieving design map (dangerous method)
        self.coll_aux_joint_map = get_coll_joint_map()

        self.total_nnodes = 0 # total number of nodes of the problem (not intervals!)
        self.filling_n_nodes = filling_n_nodes # filling nodes in between two base task nodes 
        # (these nodes are not added between two successive tasks)
        self.nodes_list = [] # nodes list to keep track of the problem structure
        self.task_list = [] # list of IDs specifying the task type for each self.nodes_list[i]
        self.final_node = None
        self.n_of_tasks = 0 # number of tasks added to this object, counting multiplicity
        self.n_of_base_tasks = 0 # number of base tasks added to this object (multiplicity not counted)

        self.dt = 0.0 # dt of the problem (for now constant)
        self.tf = 0.0 # final time of the traj. opt. problem
        self.n_int = 0

        self.prb = None # problem object

        self.rght_arm_picks = [] # whether the right or left arm picks 
        self.contact_heights = [] # reference heights
        self.hor_offsets = [] # horizontal offsets w.r.t. the object (not used now)

        self.was_init_called = False # flag to check if the init() method was called

        self.task_names = ["in_place_flip", "bimanual"] # definition of task names (for now, TB modified manually)
        self.task_base_n_nodes_dict = {} # base task number of nodes (populated automatically)
        self.was_task_already_added = [False, False] # list of flags to be used to check whether it is the first time the task is added

        self.task_dict = {} # dictionary mapping self.task_names[i] to a unique ID
        
        # kinematic quantities

        self.lft_inward_q = np.array([- np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0]) # inward orientation for left arm
        self.rght_inward_q = np.array([- np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0, 0.0]) # inward orientation for right arm

        self.lft_pick_q = [] # where left arm will pick
        self.rght_pick_q = [] # where right arm will pick

        self.object_pos_lft = [] # object pose on working surface
        self.object_q_lft = []
        self.object_pos_rght = []
        self.object_q_rght = []  
        self.object_size = [] # used by the bimanual task

        self.rarm_tcp_pos_rel_ws = None # relative positions and orientations to the ws frame (but not expressed wrt to it)
        self.larm_tcp_pos_rel_ws = None 
        self.rarm_tcp_rot_wrt_ws = None
        self.larm_tcp_rot_wrt_ws = None

        self.lft_tcp_pos_wrt_ws = None # relative positions and orientations to the ws expressed w.r.t. that frame
        self.lft_tcp_rot_wrt_ws = None
        self.rght_tcp_pos_wrt_ws = None
        self.rght_tcp_rot_wrt_ws = None
        self.rght_tcp_pos_wrt_lft_tcp = None
        self.rght_tcp_rot_wrt_lft_tcp = None

        self.jac_arm_r = None
        self.jac_arm_l = None
        self.rarm_tcp_jacobian_sym = None
        self.larm_tcp_jacobian_sym = None

        self.handover_rot_loc = get_crossed_handover_local_rot()
        # constant matrix to define the relative orientation between hand frames when 
        # performing the handover maneuver  
        self.bimanual_rot_loc = get_bimanual_frame_rot()
        # constant matrix to define the relative orientation between hand frames when 
        # performing the bimanual pickup

        self.coll_handler = None
        self.tcp_contact_nodes = [] # used to let the collision handler know when to relax tcp collision 
        # avoidance constraint

        self.rot2trasl_man_scl_fact = rot2trasl_man_scl_fact

        self.lift_height_biman = 0.4

    def get_main_nodes_offset(self, total_task_nnodes: int, task_base_n_nodes: int):
    
        # simple method to get the offset of the main nodes of a single flipping task
    
        main_nodes_offset = int((total_task_nnodes - task_base_n_nodes)\
                               /(task_base_n_nodes - 1)) + 1

        return main_nodes_offset

    def compute_nodes(self, init_node: int, filling_n_nodes: int, task_name: str):

        # HOW THE PROBLEM IS BUILT:
        #                    
        #   a b    (task1)                   (task2)    
        # |                       |  |                       |
        # |     |     |     |     |  |     |     |     |     |
        # | | | | | | | | | | | | |  | | | | | | | | | | | | | 
        # 0     1     2     3     4  0     1     2     3     4

        # 0 - 1 - 2 - 3 - 4 are the base nodes of the first task
        # 1.0 to 1.4 are the base nodes of the successive task,...
        # which is attached to the previous by filling nodes c, d (if present)
        # a and b are examples of filling nodes within a task
        # by default the same number of filling nodes is used for all internal filling nodes  across all tasks ...
        # and also for the junctions between successive tasks

        # This method returns next_task_node, which can be used 
        # as the init_node argument when calling the add_task method 
        # on the next task.

        task_base_nnodes = self.task_base_n_nodes_dict[task_name]# base number of nodes of the selected task

        total_n_task_nodes = (task_base_nnodes - 1) * filling_n_nodes +  task_base_nnodes # total number of nodes required to perform a single task

        self.total_nnodes = self.total_nnodes + total_n_task_nodes # incrementing total nodes counter

        next_task_node = init_node + total_n_task_nodes  # computing the node of the next task

        self.nodes_list.append(list(range(init_node, next_task_node))) # appending task node to the nodes list (from init_node to next_task_node - 1)

        self.task_list.append(self.task_dict[task_name]) # appending task ID to the task list

        return next_task_node

    def set_ig(self, q_ig = None, q_dot_ig = None):

        if q_ig is not None:

            self.q.setInitialGuess(q_ig)

        if q_dot_ig is not None:

            self.q_dot.setInitialGuess(q_dot_ig)

    def compute_cl_man(self, J):

        J_trasl = J[0:3, :]
        J_rot = J[3:(J[:,0].shape[0]), :]
    
        cl_man_trasl = cs.sqrt(cs.det(J_trasl @ J_trasl.T))
        cl_man_rot = cs.sqrt(cs.det(J_rot @ J_rot.T))
        cl_man_tot = cs.sqrt(cs.det(J @ J.T))

        return cl_man_trasl, cl_man_rot, cl_man_tot
    
    def compute_cl_man_cost(self, J):

        # modified version to avoid unstable sqrt method
        # we basically use the 4th power of the classical 
        # index

        J_trasl = J[0:3, :]
        J_rot = J[3:(J[:,0].shape[0]), :]
    
        cl_man_trasl_cost = (cs.det(J_trasl @ J_trasl.T)) **2
        cl_man_rot_cost = (cs.det(J_rot @ J_rot.T)) **2
        cl_man_tot_cost = (cs.det(J @ J.T)) **2

        return cl_man_trasl_cost, cl_man_rot_cost, cl_man_tot_cost

    def compute_nonloc_man(self, weights, q_dot):

        non_local_man = q_dot.T @ weights @ q_dot

        return non_local_man

    def compute_static_tau(self):
        
        static_tau = kin_dyn.InverseDynamics(self.urdf_kin_dyn, [], \
                casadi_kin_dyn.py3casadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(self.q,\
                                                        np.array([0] * self.nq), np.array([0] * self.nv)) 

        return static_tau[get_actuated_jnts_indxs(self.nq)]

    def add_cl_man_cost(self, epsi_cl_man = 1e-5):

        Jl = self.larm_tcp_jacobian_sym
        Jr = self.rarm_tcp_jacobian_sym

        man_l_trasl_cost, man_l_rot_cost, _1 = self.compute_cl_man_cost(Jl)
        man_r_trasl_cost, man_r_rot_cost, _2 = self.compute_cl_man_cost(Jr)

        # max cl. manipulability (can be defined on all task nodes without problems,
        # because it involves q and not q_dot)
        # same weights between left and right

        self.prb.createIntermediateCost("max_clman_l_trasl",\
                        self.weight_classical_man * self.weight_clman_trasl / (man_l_trasl_cost + epsi_cl_man))
        self.prb.createIntermediateCost("max_clman_l_rot",\
                        self.weight_classical_man * self.weight_clman_rot / (man_l_rot_cost + epsi_cl_man))

        self.prb.createIntermediateCost("max_clman_r_trasl",\
                        self.weight_classical_man * self.weight_clman_trasl / (man_r_trasl_cost + epsi_cl_man))
        self.prb.createIntermediateCost("max_clman_r_rot",\
                        self.weight_classical_man * self.weight_clman_rot / (man_r_rot_cost + epsi_cl_man))

    def add_manip_cost(self, is_classical_man = False):
        
        n_of_tasks = len(self.nodes_list)
        
        # the global man cost has to be removed from the final node of each task
        # so that the transition nodes between them do not influence the optimization
        # This is not needed for the classical man cost because it does not involve joint velocity

        for i in range(n_of_tasks): # iterate through each task
            
            start_node_idx = self.nodes_list[i][0] # first node index of task i
            last_node_idx = self.nodes_list[i][-1] # last node index of task i
            
            # min inputs 
            self.prb.createIntermediateCost("max_nonlocal_manipulability" + str(i),\
                            self.weight_glob_man * (self.compute_nonloc_man(self.vel_weights, self.q_dot_actuated_jnts)),
                            nodes = range(start_node_idx, last_node_idx)) 

        if is_classical_man:

            self.add_cl_man_cost()           

    def add_static_tau_cost(self):
        
        static_tau = self.compute_static_tau()
        # min inputs
        self.prb.createIntermediateCost("min_static_torque",\
                        self.weight_static_tau * (static_tau.T @ self.weight_static_tau @ static_tau))         

    def init_prb(self, urdf_full_path: str, weight_pos = 0.001, weight_rot = 0.001,\
                weight_glob_man = 0.0001, weight_class_man = 0.0001, weight_static_tau = 0.0001,\
                is_soft_pose_cnstr = False,\
                tf_single_task = 10):

        ## All the main initializations for the prb are performed here ##

        self.is_soft_pose_cnstrnt = is_soft_pose_cnstr

        self.was_init_called  = True

        self.weight_pos = weight_pos / ( self.total_nnodes ) # scaling weights on the basis of the problem dimension
        self.weight_rot = weight_rot / ( self.total_nnodes )
        self.weight_glob_man = weight_glob_man / ( self.total_nnodes )
        self.weight_classical_man = weight_class_man / ( self.total_nnodes )
        self.weight_static_tau = weight_static_tau / ( self.total_nnodes )

        self.n_int = self.total_nnodes - 1 # adding addditional filling nodes between nodes of two successive tasks
        self.prb = problem.Problem(self.n_int) 

        self.urdf = open(urdf_full_path, 'r').read()
        self.urdf_kin_dyn = casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn(self.urdf)
        self.tf = tf_single_task * len(self.nodes_list)
        self.dt = self.tf / self.n_int
        
        self.joint_names = self.urdf_kin_dyn.joint_names()
        if 'universe' in self.joint_names: self.joint_names.remove('universe')
        if 'floating_base_joint' in self.joint_names: self.joint_names.remove('floating_base_joint')
        
        print(self.joint_names)
        print("\n")

        self.nq = self.urdf_kin_dyn.nq()
        self.nv = self.urdf_kin_dyn.nv()

        self.lbs = self.urdf_kin_dyn.q_min() 
        self.ubs = self.urdf_kin_dyn.q_max()

        self.q = self.prb.createStateVariable('q', self.nq)
        self.q_dot = self.prb.createInputVariable('q_dot', self.nv)

        # THIS DEFINITIONS CAN CHANGE IF THE URDF CHANGES --> MIND THE URDF!!!

        self.q_design = self.q[self.d_var_map["mount_h"],\
                            self.d_var_map["should_w_l"], self.d_var_map["should_roll_l"], self.d_var_map["wrist_off_l"], \
                            self.d_var_map["should_w_r"], self.d_var_map["should_roll_r"], self.d_var_map["wrist_off_r"]] # design vector

        self.q_design_dot = self.q_dot[self.d_var_map["mount_h"],\
                            self.d_var_map["should_w_l"], self.d_var_map["should_roll_l"], self.d_var_map["wrist_off_l"], \
                            self.d_var_map["should_w_r"], self.d_var_map["should_roll_r"], self.d_var_map["wrist_off_r"]]
   
        self.q_actuated_jnts = self.q[get_actuated_jnts_indxs(self.nq)]
      
        self.q_dot_actuated_jnts = self.q_dot[get_actuated_jnts_indxs(self.nv)]

        self.wrist_off_ref = self.prb.createParameter('wrist_off_ref', 1)

        self.q_codes_ref = self.prb.createParameter('q_codes_ref', 4)

        self.prb.setDynamics(self.q_dot)
        self.prb.setDt(self.dt)  

        # getting useful kinematic quantities
        fk_ws = self.urdf_kin_dyn.fk("working_surface_link")
        ws_link_pos = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_pos"] # w.r.t. world
        ws_link_rot = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)
        
        fk_arm_r = self.urdf_kin_dyn.fk("arm_1_tcp")
        rarm_tcp_pos = fk_arm_r(q = self.q)["ee_pos"] # w.r.t. world
        rarm_tcp_rot = fk_arm_r(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        self.rarm_tcp_pos_rel_ws = rarm_tcp_pos - ws_link_pos # pos vector relative to working surface in world frame (w.r.t. world)
        
        self.jac_arm_r =self.urdf_kin_dyn.jacobian("arm_1_tcp",\
                        casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        self.rarm_tcp_jacobian_sym = self.jac_arm_r(q = self.q)["J"]

        fk_arm_l = self.urdf_kin_dyn.fk("arm_2_tcp")
        larm_tcp_pos = fk_arm_l(q = self.q)["ee_pos"] # w.r.t. world
        larm_tcp_rot = fk_arm_l(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        self.larm_tcp_pos_rel_ws = larm_tcp_pos - ws_link_pos # pos vector relative to working surface in world frame (w.r.t. world)

        self.jac_arm_l = self.urdf_kin_dyn.jacobian("arm_2_tcp",\
                        casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        self.larm_tcp_jacobian_sym = self.jac_arm_l(q = self.q)["J"]

        self.rght_tcp_rot_wrt_ws = ws_link_rot.T @ rarm_tcp_rot
        self.rght_tcp_pos_wrt_ws = ws_link_rot.T @ (rarm_tcp_pos - ws_link_pos)

        self.lft_tcp_rot_wrt_ws = ws_link_rot.T @ larm_tcp_rot
        self.lft_tcp_pos_wrt_ws = ws_link_rot.T @ (larm_tcp_pos - ws_link_pos)

        self.rght_tcp_pos_wrt_lft_tcp = self.lft_tcp_rot_wrt_ws.T @ (self.rght_tcp_pos_wrt_ws - self.lft_tcp_pos_wrt_ws)
        self.rght_tcp_rot_wrt_lft_tcp = self.lft_tcp_rot_wrt_ws.T @ self.rght_tcp_rot_wrt_ws

    def setup_prb(self,\
                epsi = epsi_default,\
                q_ig = None, q_dot_ig = None, 
                is_classical_man = False,
                is_static_tau = False, 
                is_second_lev_opt = False):
         
        ## All the constraints and costs are set here ##

        # setting initial guesses
        if q_ig is not None:

            self.q.setInitialGuess(q_ig)

        if q_dot_ig is not None:
            
            self.q_dot.setInitialGuess(q_dot_ig)

        # lower and upper bounds for design variables and joint variables
        self.q.setBounds(self.lbs, self.ubs)

        # we don't need vel. limits since we are doing only a "positional"
        # kinematic opt 
        
        # add also velocity bounds? 
        
        # adapt link5 coll model to the wrist offset so that future design 
        # changes of link5 are accounted for in the optimzation

        self.prb.createConstraint("adapt_link5_coll_model_l", \
                                self.q[self.coll_aux_joint_map["link5_coll_joint_l"]] - \
                                self.q[self.d_var_map["wrist_off_l"]])
        self.prb.createConstraint("adapt_link5_coll_model_r", \
                                self.q[self.coll_aux_joint_map["link5_coll_joint_r"]] - \
                                self.q[self.d_var_map["wrist_off_r"]])

        # roll and shoulder vars equal
        self.prb.createConstraint("same_roll", \
                                self.q[self.d_var_map["should_roll_l"]] - \
                                self.q[self.d_var_map["should_roll_r"]])

        self.prb.createConstraint("same_shoulder_w",\
                                self.q[self.d_var_map["should_w_l"]] - \
                                self.q[self.d_var_map["should_w_r"]])

        self.prb.createConstraint("same_wrist_offset",\
                                self.q[self.d_var_map["wrist_off_l"]] - \
                                self.q[self.d_var_map["wrist_off_r"]])

        # design vars equal on all nodes 
        self.prb.createConstraint("single_var_cntrnt",\
                            self.q_design_dot,\
                            nodes = range(0, (self.total_nnodes - 1)))

        if is_second_lev_opt: # second level opt --> assign codesign var to parameter

            self.prb.createConstraint("codesign_values",\
                    self.q_design[0:4] - self.q_codes_ref,\
                    nodes = 0) # is sufficient to put it on the first node and only on one arm
                    # since there are already constraints which impose simmetry of design vars

        else:

            if not self.is_sliding_wrist: # setting value for wrist offsets (basically
            # removing the wrist from the design variables)

                self.prb.createConstraint("wrist_offset_value",\
                        self.q[self.d_var_map["wrist_off_l"]] - self.wrist_off_ref,\
                        nodes = 0) # is sufficient to put it on the first node and only on one arm 
              

        # TCPs inside working volume (assumed to be a box)
        ws_ub = np.array([1.25, 0.5, 1.0])
        ws_lb = np.array([-1.25, -0.5, 0.0])

        keep_tcp_in_ws_rght = self.prb.createConstraint("keep_tcp_in_ws_rght",\
                                                           self.rght_tcp_pos_wrt_ws)
        keep_tcp_in_ws_rght.setBounds(ws_lb, ws_ub)
        keep_tcp_in_ws_lft = self.prb.createConstraint("keep_tcp_in_ws_lft",\
                                                           self.lft_tcp_pos_wrt_ws)
        keep_tcp_in_ws_lft.setBounds(ws_lb, ws_ub)

        # kinemaic costs
        self.add_manip_cost(is_classical_man = is_classical_man)

        # static tau cost
        if is_static_tau:

            self.add_static_tau_cost()

        # here the "custom" task is added to the problem
        self.build_tasks(is_soft_pose_cnstr = self.is_soft_pose_cnstrnt, epsi = epsi)

        # Simple p2p collision avoidance cnstraints 
        self.coll_handler = SimpleCollHandler(self.urdf_kin_dyn, self.q, self.prb, \
                                            self.coll_yaml_path,\
                                            tcp_contact_nodes = self.tcp_contact_nodes)
    
    def add_tasks(self, y_sampling: list,
                    right_arm_picks: list, 
                    is_in_place_flip = True, is_bimanual_pick = False):

        next_node = 0 # used to place the next task on the right problem nodes
        # # in place flip task
        if is_in_place_flip:

            for i in range(len(y_sampling[0])): # first row of y sampling is for in place flip

                next_node = self.add_in_place_flip_task(init_node = next_node,\
                                object_pos_wrt_ws = np.array([0.0, y_sampling[0][i], 0.0]), \
                                right_arm_picks = bool(right_arm_picks[i]))
        # bimanual task
        if is_bimanual_pick:
            
            for j in range(len(y_sampling[1])): # second row of y sampling is for biman pick
                
                next_node = self.add_bimanual_pick_task(init_node = next_node,\
                                object_pos_wrt_ws = np.array([0.0, y_sampling[1][j], 0.0]))

    def build_tasks(self, is_soft_pose_cnstr = False, epsi = epsi_default):\
        
        base_nnodes = 0
        base_nnodes_previous = [0] * (len(self.nodes_list)) # auxiliary variable
        base_nnodes_previous[0] = 0
        delta_offset = 1

        for i in range(len(self.nodes_list)): # iterate through multiple tasks

            if self.task_names[0] in self.task_dict: # if the task was added

                if self.task_list[i] == self.task_dict[self.task_names[0]]: # in place flip task
                    
                    base_nnodes = self.task_base_n_nodes_dict[self.task_names[0]] # reading task n. of base nodes

            if self.task_names[1] in self.task_dict: # if the task was added

                if self.task_list[i] == self.task_dict[self.task_names[1]]: # bimanual task
                    
                    base_nnodes = self.task_base_n_nodes_dict[self.task_names[1]]
            

            delta_offset = self.get_main_nodes_offset(len(self.nodes_list[i]), base_nnodes) 
            # = 1 if no intermediate node between base nodes was inserted

            base_nodes_aux = np.sum(np.array(base_nnodes_previous))

            for j in range(base_nnodes): # iterate through base task nodes
                                
                constraint_unique_id_rght = j + 2 * base_nodes_aux 
                # index used to give different names to each constraint (rght arm)
                constraint_unique_id_lft = j + base_nnodes + 2 * base_nodes_aux 
                # index used to give different names to each constraint (lft arm)
        
                # The way pose constraint names are assigned (example with task made of 9 base nodes):
                # -----------------------------------------------
                # R arm: | 0  1  2  3  4  5  6  7  8 | 18 19 --->
                # -----------------------------------------------
                # L arm: | 9 10 11 12 13 14 15 16 17 | 27 28 --->
                # -----------------------------------------------
                # node:  | 0  1  2  3  4  5  6  7  8 | 9  10 --->
                # -----------------------------------------------
                
                cnstrnt_node_index = self.nodes_list[i][j * delta_offset] 
                # index of the constraint (traslated by the number of filling nodes)

                if self.task_names[0] in self.task_dict: # if the task was added

                    if self.task_list[i] == self.task_dict[self.task_names[0]]:
                        
                        self.build_in_place_flip_task(i, j, cnstrnt_node_index,\
                                                    constraint_unique_id_rght, constraint_unique_id_lft, \
                                                    is_soft_pose_cnstr = is_soft_pose_cnstr,\
                                                    epsi = epsi)
                
                if self.task_names[1] in self.task_dict: # if the task was added

                    if self.task_list[i] == self.task_dict[self.task_names[1]]:
                        
                        self.build_bimanual_task(i, j, cnstrnt_node_index,\
                                                    constraint_unique_id_rght, constraint_unique_id_lft, \
                                                    delta_offset,\
                                                    base_nnodes,\
                                                    is_soft_pose_cnstr = is_soft_pose_cnstr,\
                                                    epsi = epsi)

            if i!=(len(self.nodes_list) - 1):

                base_nnodes_previous[i + 1] = base_nnodes

    def add_in_place_flip_task(self, init_node: int,\
                               right_arm_picks = True,\
                               object_pos_wrt_ws = np.array([0.0, 0.0, 0.0]),\
                               object_q_wrt_ws = np.array([1, 0, 0, 0]),\
                               pick_q_wrt_ws = np.array([1, 0, 0, 0]),\
                               contact_height = 0.4, hor_offset = 0.2, 
                               task_base_nodes = 3):

        if self.was_init_called : # we do not support adding more tasks after having built the problem!

            raise Exception("You can only add tasks before calling init_prb(*)!!")

        if not self.was_task_already_added[0]:

            self.was_task_already_added[0] =  True # setting flag so that next

            self.n_of_base_tasks = self.n_of_base_tasks + 1 # incrementing base tasks counter 

            # assigning unique code to task when the method is called the first time
            self.task_dict[self.task_names[0]] = self.n_of_base_tasks -1

            self.task_base_n_nodes_dict[self.task_names[0]] = task_base_nodes

        self.n_of_tasks = self.n_of_tasks + 1 # incrementing counter for the number of tasks

        self.object_pos_lft.append(object_pos_wrt_ws)
        self.object_q_lft.append(object_q_wrt_ws)
        self.object_pos_rght.append(object_pos_wrt_ws)
        self.object_q_rght.append(object_q_wrt_ws)

        self.lft_pick_q.append(pick_q_wrt_ws)
        self.rght_pick_q.append(pick_q_wrt_ws)

        self.contact_heights.append(contact_height)
        self.hor_offsets.append(hor_offset)
        self.rght_arm_picks.append(right_arm_picks)

        next_task_node = self.compute_nodes(init_node, self.filling_n_nodes, self.task_names[0])

        return next_task_node

    def add_bimanual_pick_task(self, init_node: int,\
                                right_arm_picks = True,\
                                object_pos_wrt_ws = np.array([0.0, 0.0, 0.0]),\
                                object_q_wrt_ws = np.array([1, 0, 0, 0]),\
                                pick_q_wrt_ws = np.array([1, 0, 0, 0]),\
                                contact_height = 0.1, hor_offset = 0.0,
                                object_size = 0.05, 
                                lift_height = 0.2, 
                                task_base_nodes = 3):

        if self.was_init_called : # we do not support adding more tasks after having built the problem!

            raise Exception("You can only add tasks before calling init_prb(*)!!")

        if not self.was_task_already_added[1]:

            self.was_task_already_added[1] =  True # setting flag so that next

            self.n_of_base_tasks = self.n_of_base_tasks + 1 # incrementing base tasks counter 

            # assigning unique code to task when the method is called the first time
            self.task_dict[self.task_names[1]] = self.n_of_base_tasks -1

            self.task_base_n_nodes_dict[self.task_names[1]] = task_base_nodes

        self.n_of_tasks = self.n_of_tasks + 1 # incrementing counter for the number of tasks

        self.object_pos_lft.append(object_pos_wrt_ws)
        self.object_q_lft.append(object_q_wrt_ws)
        self.object_pos_rght.append(object_pos_wrt_ws)
        self.object_q_rght.append(object_q_wrt_ws)

        self.lft_pick_q.append(pick_q_wrt_ws)
        self.rght_pick_q.append(pick_q_wrt_ws)

        self.contact_heights.append(contact_height)
        self.hor_offsets.append(hor_offset)
        self.rght_arm_picks.append(True) # not used

        self.object_size.append(object_size)

        next_task_node = self.compute_nodes(init_node, self.filling_n_nodes, self.task_names[1])

        self.lift_height_biman = lift_height
    
        return next_task_node

    def add_pick_and_place_task(self, init_node: int,\
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

    def build_in_place_flip_task(self, i: int, j: int,\
                                cnstrnt_node_index: int,\
                                cnstrnt_id_rght: int, cnstrnt_id_lft: int,\
                                is_soft_pose_cnstr = True, epsi = epsi_default):

     
        # if j == 0: # ARM 1: waiting | ARM 2: waiting (starting from a reference height)
                                
        #     if (self.rght_arm_picks[i]): # right arm picks
                
        #         # right arm
        #         add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
        #                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
        #                         self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_rght[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)
                
                
        #         # left arm
        #         add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
        #                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
        #                         self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_lft[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)

                
        #     else: # left arm picks
                
        #         # right arm
        #         add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
        #                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
        #                         self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_rght[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)
        #         # left arm
        #         add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
        #                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,
        #                         self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_lft[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)

        if j == 0: # ARM 1: goes to pick position | ARM 2: waiting

            if (self.rght_arm_picks[i]): # right arm picks
                
                # right arm (rotation transposed so that orientation err is comp. wrt ws frame)
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws.T,\
                                self.object_pos_rght[i], quat2rot(self.rght_pick_q[i]).T,\
                                pos_selection = ["x",  "y", "z"],\
                                rot_selection = ["x",  "y"],\
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)
                # left arm
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
                                self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]),\
                                quat2rot(self.object_q_lft[i]),\
                                pos_selection = [],\
                                rot_selection = [], \
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)

            else: # left arm picks
                
                # right arm
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index,\
                                self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]),\
                                quat2rot(self.object_q_rght[i]),\
                                pos_selection = [],\
                                rot_selection = [], \
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)
                # left arm
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
                                self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws.T,
                                self.object_pos_lft[i], quat2rot(self.lft_pick_q[i]).T,
                                pos_selection = ["x",  "y", "z"],\
                                rot_selection = ["x",  "y"],\
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)
                                    
        if j == 1: # ARM 1: waits for contact | ARM 2: makes contact with bartender constraint
            
            if (self.rght_arm_picks[i]): # right arm picks

                # exchange has to happen with horizontal hands
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                    self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                    self.object_pos_rght[i] + np.array([0.0, 0.0, self.contact_heights[i]]),\
                                    quat2rot(self.object_q_rght[i]),\
                                    pos_selection = [],\
                                    rot_selection = ["x", "y"],\
                                    weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                    is_soft = is_soft_pose_cnstr, epsi = epsi)

                # relative constraint
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index,\
                            pos = self.rght_tcp_pos_wrt_lft_tcp, rot = self.rght_tcp_rot_wrt_lft_tcp,
                            pos_ref = np.array([0.0] * 3), rot_ref = self.handover_rot_loc,
                            pos_selection = ["x", "y", "z"],\
                            rot_selection = ["x", "y",  "z"],\
                            weight_rot = self.weight_rot,\
                            is_soft = is_soft_pose_cnstr, epsi = epsi)

            else:

                # exchange has to happen with horizontal hands
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
                                    self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                    self.object_pos_lft[i] + np.array([0.0, 0.0, self.contact_heights[i]]),\
                                    quat2rot(self.object_q_lft[i]),\
                                    pos_selection = [],\
                                    rot_selection = ["x", "y"],\
                                    weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                    is_soft = is_soft_pose_cnstr, epsi = epsi)


                # relative constraint
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index,\
                            pos = self.rght_tcp_pos_wrt_lft_tcp, rot = self.rght_tcp_rot_wrt_lft_tcp,
                            pos_ref = np.array([0.0] * 3), rot_ref = self.handover_rot_loc,
                            pos_selection = ["x", "y", "z"],\
                            rot_selection = ["x", "y",  "z"],\
                            weight_rot = self.weight_rot,\
                            is_soft = is_soft_pose_cnstr, epsi = epsi)

            self.tcp_contact_nodes.append(cnstrnt_node_index) # signaling to the tcp collision avoidance handler
            # to relax the constraint

        if j == 2: # ARM 1: back to waiting pose | ARM 2: down to picking  pose

            if (self.rght_arm_picks[i]): # right arm picks

                # right arm
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]),\
                                quat2rot(self.object_q_rght[i]),\
                                pos_selection = [],\
                                rot_selection = [], \
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)
                # left arm
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
                                self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws.T,\
                                self.object_pos_lft[i], quat2rot(self.lft_pick_q[i]).T,\
                                pos_selection = ["x", "y", "z"],\
                                rot_selection = ["x",  "y"],\
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)

            else: # left arm picks
                
                # right arm
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws.T,\
                                self.object_pos_rght[i], quat2rot(self.rght_pick_q[i]).T,\
                                pos_selection = ["x", "y", "z"],\
                                rot_selection = ["x",  "y"],\
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)
                # left arm
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
                                self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
                                self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]),\
                                quat2rot(self.object_q_lft[i]),\
                                pos_selection = [],\
                                rot_selection = [], \
                                weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)

        # if j == 4: # ARM 1: waits | ARM 2: back to waiting pose

        #     if (self.rght_arm_picks[i]): # right arm picks

        #         # right arm
        #         add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
        #                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
        #                         self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_rght[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)
        #         # left arm
        #         add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
        #                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
        #                         self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_lft[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)

        #     else: # left arm picks
                
        #         # right arm
        #         add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index,\
        #                         self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
        #                         self.object_pos_rght[i] + np.array([0.0, - self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_rght[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)
        #         # left arm
        #         add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index, \
        #                         self.lft_tcp_pos_wrt_ws, self.lft_tcp_rot_wrt_ws,\
        #                         self.object_pos_lft[i] + np.array([0.0, self.hor_offsets[i], self.contact_heights[i]]),\
        #                         quat2rot(self.object_q_lft[i]),\
        #                         pos_selection = [],\
        #                         rot_selection = [], \
        #                         weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
        #                         is_soft = is_soft_pose_cnstr, epsi = epsi)
    
    def build_bimanual_task(self, i: int, j: int,\
                            cnstrnt_node_index: int,\
                            cnstrnt_id_rght: int, cnstrnt_id_lft: int,\
                            delta_offset: int,
                            base_nnodes: int,
                            is_soft_pose_cnstr = True, epsi = epsi_default, 
                            ):
        

        if j == 0: # ARM 1: waiting pose | ARM 2: waiting pose

                next_base_node_index = self.nodes_list[i][(j + 1) * delta_offset] # used
                # to set bimanual pose also on nodes between task base nodes

                # exchange has to happen with vertical hands
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                    self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                    self.object_pos_rght[i] + np.array([0.0, - self.object_size[0]/2, self.lift_height_biman]),\
                                    cs.DM([[0.0, 1.0, 0.0],\
                                            [0.0, 0.0, -1.0],\
                                            [-1.0, 0.0, 0.0]]), \
                                    pos_selection = ["x", "y", "z"],\
                                    rot_selection = ["x", "y", "z"],\
                                    weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                    is_soft = is_soft_pose_cnstr, epsi = epsi)
                
                
                # relative constraint
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, list(range(cnstrnt_node_index, next_base_node_index)),\
                                pos = self.rght_tcp_pos_wrt_lft_tcp, rot = self.rght_tcp_rot_wrt_lft_tcp,
                                pos_ref = np.array([0.0, 0.0, - self.object_size[0]]), rot_ref = self.bimanual_rot_loc,
                                pos_selection = ["x", "y", "z"],\
                                rot_selection = ["x", "y",  "z"],\
                                weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)

        if j == 1: # ARM 1: goes to pick position | ARM 2: goes to pick position
                
                next_base_node_index = self.nodes_list[i][(j + 1) * delta_offset]

                # exchange has to happen with vertical hands
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                    self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                    self.object_pos_rght[i] + np.array([0.0, - self.object_size[0]/2, self.contact_heights[i]]),\
                                    cs.DM([[0.0, 1.0, 0.0],\
                                            [0.0, 0.0, -1.0],\
                                            [-1.0, 0.0, 0.0]]), \
                                    pos_selection = ["x", "y", "z"],\
                                    rot_selection = ["x", "y", "z"],\
                                    weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                    is_soft = is_soft_pose_cnstr, epsi = epsi)

                # relative constraint
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, list(range(cnstrnt_node_index, next_base_node_index)),\
                                pos = self.rght_tcp_pos_wrt_lft_tcp, rot = self.rght_tcp_rot_wrt_lft_tcp,
                                pos_ref = np.array([0.0, 0.0, - self.object_size[0]]), rot_ref = self.bimanual_rot_loc,
                                pos_selection = ["x", "y", "z"],\
                                rot_selection = ["x", "y",  "z"],\
                                weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)

                
        if j == 2: 
            
                add_pose_cnstrnt(cnstrnt_id_rght, self.prb, cnstrnt_node_index, \
                                    self.rght_tcp_pos_wrt_ws, self.rght_tcp_rot_wrt_ws,\
                                    self.object_pos_rght[i] + np.array([0.0, - self.object_size[0]/2, self.lift_height_biman]),\
                                    cs.DM([[0.0, 1.0, 0.0],\
                                            [0.0, 0.0, -1.0],\
                                            [-1.0, 0.0, 0.0]]), \
                                    pos_selection = ["x", "y", "z"],\
                                    rot_selection = ["x", "y", "z"],\
                                    weight_pos = self.weight_pos, weight_rot = self.weight_rot,\
                                    is_soft = is_soft_pose_cnstr, epsi = epsi)

                # relative constraint
                add_pose_cnstrnt(cnstrnt_id_lft, self.prb, cnstrnt_node_index,\
                                pos = self.rght_tcp_pos_wrt_lft_tcp, rot = self.rght_tcp_rot_wrt_lft_tcp,
                                pos_ref = np.array([0.0, 0.0, - self.object_size[0]]), rot_ref = self.bimanual_rot_loc,
                                pos_selection = ["x", "y", "z"],\
                                rot_selection = ["x", "y",  "z"],\
                                weight_rot = self.weight_rot,\
                                is_soft = is_soft_pose_cnstr, epsi = epsi)

class HighClManGen:

    # class for geneating reference number for high-classical manipulability configurations
    def __init__(self,
                sliding_wrist_offset = 0.0, \
                coll_yaml_path = ""):
        
        self.coll_yaml_path = coll_yaml_path

        self.sliding_wrist_offset =  sliding_wrist_offset # mounting offset for wrist auxiliary joint

        self.weight_classical_man = 0 # actual weight assigned to the classical manipulability cost

        self.urdf = None # opened urdf file
        self.joint_names = None # joint names
        self.nq = 0 # number of positional states
        self.nv = 0 # number of velocity  states
        self.urdf_kin_dyn = None # CasadiKinDyn object
        self.lbs = None # holds the lower (positional) bounds on state variables
        self.ubs = None # holds the upper (positional) bounds on state variables

        self.q = None # positional state
        self.q_dot = None # velocity state
        self.q_design = None # positional state of design variables
        self.q_design_dot = None # velocity state of design variables
        self.q_codes_ref = None
        self.wrist_off_ref = None

        self.d_var_map = get_design_map() # retrieving design map (dangerous method)

        self.total_nnodes = 2 # total number of nodes of the problem (not intervals!)

        self.dt = 0.0 # dt of the problem (for now constant)
        self.tf = 0.0 # final time of the traj. opt. problem
        self.n_int = 0

        self.prb = None # problem object

        self.rght_arm_picks = [] # whether the right or left arm picks 
        self.contact_heights = [] # reference heights
        self.hor_offsets = [] # horizontal offsets w.r.t. the object (not used now)

        self.was_init_called = False # flag to check if the init() method was called

        # kinematic quantities

        self.lft_inward_q = np.array([- np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0]) # inward orientation for left arm
        self.rght_inward_q = np.array([- np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0, 0.0]) # inward orientation for right arm

        self.lft_pick_q = [] # where left arm will pick
        self.rght_pick_q = [] # where right arm will pick

        self.object_pos_lft = [] # object pose on working surface
        self.object_q_lft = []
        self.object_pos_rght = []
        self.object_q_rght = []  
        self.object_size = [] # used by the bimanual task

        self.rarm_tcp_pos_rel_ws = None # relative positions and orientations to the ws frame (but not expressed wrt to it)
        self.larm_tcp_pos_rel_ws = None 
        self.rarm_tcp_rot_wrt_ws = None
        self.larm_tcp_rot_wrt_ws = None

        self.lft_tcp_pos_wrt_ws = None # relative positions and orientations to the ws expressed w.r.t. that frame
        self.lft_tcp_rot_wrt_ws = None
        self.rght_tcp_pos_wrt_ws = None
        self.rght_tcp_rot_wrt_ws = None
        self.rght_tcp_pos_wrt_lft_tcp = None
        self.rght_tcp_rot_wrt_lft_tcp = None

        self.jac_arm_r = None
        self.jac_arm_l = None
        self.rarm_tcp_jacobian_sym = None
        self.larm_tcp_jacobian_sym = None

        self.coll_handler = None

        self.rot2trasl_man_scl_fact = rot2trasl_man_scl_fact

    def set_ig(self, q_ig = None, q_dot_ig = None):

        if q_ig is not None:

            self.q.setInitialGuess(q_ig)

        if q_dot_ig is not None:

            self.q_dot.setInitialGuess(q_dot_ig)

    def compute_cl_man(self, J):

        J_trasl = J[0:3, :]
        J_rot = J[3:(J[:,0].shape[0]), :]
    
        cl_man_trasl = cs.sqrt(cs.det(J_trasl @ J_trasl.T))
        cl_man_rot = cs.sqrt(cs.det(J_rot @ J_rot.T))
        cl_man_tot = cs.sqrt(cs.det(J @ J.T))

        return cl_man_trasl, cl_man_rot, cl_man_tot

    def compute_nonloc_man(self, q_dot):

        node_man = cs.sumsqr(q_dot)

        return node_man

    def add_cl_man_cost(self):

        Jl = self.larm_tcp_jacobian_sym
        Jr = self.rarm_tcp_jacobian_sym

        man_l_trasl, man_l_rot, _1 = self.compute_cl_man(Jl)
        man_r_trasl, man_r_rot, _2 = self.compute_cl_man(Jr)

        self.prb.createIntermediateCost("max_clman_l_trasl",\
                        self.weight_classical_man / (man_l_trasl)**2, nodes=1)
        self.prb.createIntermediateCost("max_clman_l_rot",\
                        self.weight_classical_man / (man_l_rot)**2, nodes=1)

        self.prb.createIntermediateCost("max_clman_r_trasl",\
                        self.weight_classical_man / (man_r_trasl)**2, nodes=1)
        self.prb.createIntermediateCost("max_clman_r_rot",\
                        self.weight_classical_man / (man_r_rot)**2, nodes=1)
        
        # self.prb.createIntermediateCost("max_clman_l_trasl",\
        #                 self.weight_classical_man / (man_l_trasl)**2)
        # self.prb.createIntermediateCost("max_clman_l_rot",\
        #                 self.weight_classical_man / (man_l_rot)**2)

        # self.prb.createIntermediateCost("max_clman_r_trasl",\
        #                 self.weight_classical_man / (man_r_trasl)**2)
        # self.prb.createIntermediateCost("max_clman_r_rot",\
        #                 self.weight_classical_man / (man_r_rot)**2)

    def init_prb(self, urdf_full_path: str, weight_class_man = 1.0,\
                tf_single_task = 10):

        ## All the main initializations for the prb are performed here ##

        self.was_init_called  = True

        self.weight_classical_man = weight_class_man / ( self.total_nnodes )

        self.n_int = self.total_nnodes - 1 # adding addditional filling nodes between nodes of two successive tasks

        self.prb = problem.Problem(self.n_int) 

        self.urdf = open(urdf_full_path, 'r').read()
        self.urdf_kin_dyn = casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn(self.urdf)
        self.tf = tf_single_task
        self.dt = self.tf / self.n_int
        
        self.joint_names = self.urdf_kin_dyn.joint_names()
        if 'universe' in self.joint_names: self.joint_names.remove('universe')
        if 'floating_base_joint' in self.joint_names: self.joint_names.remove('floating_base_joint')
        
        print(self.joint_names)
        print("\n")

        self.nq = self.urdf_kin_dyn.nq()
        self.nv = self.urdf_kin_dyn.nv()

        self.lbs = self.urdf_kin_dyn.q_min() 
        self.ubs = self.urdf_kin_dyn.q_max()

        self.q = self.prb.createStateVariable('q', self.nq)
        self.q_dot = self.prb.createInputVariable('q_dot', self.nv)

        # THIS DEFINITIONS CAN CHANGE IF THE URDF CHANGES --> MIND THE URDF!!!
    
        self.q_design = self.q[self.d_var_map["mount_h"],\
                            self.d_var_map["should_w_l"], self.d_var_map["should_roll_l"], self.d_var_map["wrist_off_l"], \
                            self.d_var_map["should_w_r"], self.d_var_map["should_roll_r"], self.d_var_map["wrist_off_r"]] # design vector

        self.q_design_dot = self.q_dot[self.d_var_map["mount_h"],\
                            self.d_var_map["should_w_l"], self.d_var_map["should_roll_l"], self.d_var_map["wrist_off_l"], \
                            self.d_var_map["should_w_r"], self.d_var_map["should_roll_r"], self.d_var_map["wrist_off_r"]]
        
        self.wrist_off_ref = self.prb.createParameter('wrist_off_ref', 1)

        self.q_codes_ref = self.prb.createParameter('q_codes_ref', 4)

        self.prb.setDynamics(self.q_dot)
        self.prb.setDt(self.dt)  

        # getting useful kinematic quantities
        fk_ws = self.urdf_kin_dyn.fk("working_surface_link")
        ws_link_pos = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_pos"] # w.r.t. world
        ws_link_rot = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)
        
        fk_arm_r = self.urdf_kin_dyn.fk("arm_1_tcp")
        rarm_tcp_pos = fk_arm_r(q = self.q)["ee_pos"] # w.r.t. world
        rarm_tcp_rot = fk_arm_r(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        self.rarm_tcp_pos_rel_ws = rarm_tcp_pos - ws_link_pos # pos vector relative to working surface in world frame (w.r.t. world)
        
        self.jac_arm_r = self.urdf_kin_dyn.jacobian("arm_1_tcp",\
                        casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        self.rarm_tcp_jacobian_sym = self.jac_arm_r(q = self.q)["J"]

        fk_arm_l = self.urdf_kin_dyn.fk("arm_2_tcp")
        larm_tcp_pos = fk_arm_l(q = self.q)["ee_pos"] # w.r.t. world
        larm_tcp_rot = fk_arm_l(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        self.larm_tcp_pos_rel_ws = larm_tcp_pos - ws_link_pos # pos vector relative to working surface in world frame (w.r.t. world)

        self.jac_arm_l = self.urdf_kin_dyn.jacobian("arm_2_tcp",\
                        casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        self.larm_tcp_jacobian_sym = self.jac_arm_l(q = self.q)["J"]

        self.rght_tcp_rot_wrt_ws = ws_link_rot.T @ rarm_tcp_rot
        self.rght_tcp_pos_wrt_ws = ws_link_rot.T @ (rarm_tcp_pos - ws_link_pos)

        self.lft_tcp_rot_wrt_ws = ws_link_rot.T @ larm_tcp_rot
        self.lft_tcp_pos_wrt_ws = ws_link_rot.T @ (larm_tcp_pos - ws_link_pos)

        self.rght_tcp_pos_wrt_lft_tcp = self.lft_tcp_rot_wrt_ws.T @ (self.rght_tcp_pos_wrt_ws - self.lft_tcp_pos_wrt_ws)
        self.rght_tcp_rot_wrt_lft_tcp = self.lft_tcp_rot_wrt_ws.T @ self.rght_tcp_rot_wrt_ws

    def setup_prb(self,\
                q_ig = None, q_dot_ig = None):
         
        ## All the constraints and costs are set here ##

        # setting initial guesses
        if q_ig is not None:

            self.q.setInitialGuess(q_ig)

        if q_dot_ig is not None:
            
            self.q_dot.setInitialGuess(q_dot_ig)

        # lower and upper bounds for design variables and joint variables
        self.q.setBounds(self.lbs, self.ubs)

        # design vars equal on all nodes 
        # self.prb.createConstraint("single_var_cntrnt",\
        #                     self.q_design_dot,\
        #                     nodes = range(0, (self.total_nnodes - 1)))

        # roll and shoulder vars equal
        self.prb.createConstraint("same_roll", \
                                self.q[self.d_var_map["should_roll_l"]] - \
                                self.q[self.d_var_map["should_roll_r"]])

        self.prb.createConstraint("same_shoulder_w",\
                                self.q[self.d_var_map["should_w_l"]] - \
                                self.q[self.d_var_map["should_w_r"]])

        self.prb.createConstraint("same_wrist_offset",\
                                self.q[self.d_var_map["wrist_off_l"]] - \
                                self.q[self.d_var_map["wrist_off_r"]])

        self.prb.createConstraint("codesign_values",\
                self.q_design[0:4] - self.q_codes_ref,\
                nodes = 0) # is sufficient to put it on the first node and only on one arm

        self.prb.createConstraint("wrist_offset_value",\
                self.q[self.d_var_map["wrist_off_l"]] - self.wrist_off_ref,\
                nodes = 0) # is sufficient to put it on the first node and only on one arm 

        # min inputs 
        self.add_cl_man_cost()      

        # adding regularization 
        self.prb.createIntermediateCost("regular",\
                        self.compute_nonloc_man(self.q_dot))

class DoubleArmCartTask:
    
    # task geneator for testing the pose constraints on the RePAIR 
    # platform using interactive markers
    # OBSOLETE (migh be broken)

    def __init__(self,\
        rviz_process,\
        should_w = 0.26, should_roll = 2.0, mount_h = 0.5, \
        filling_n_nodes = 0,\
        collision_margin = 0.1, 
        is_sliding_wrist = False,\
        sliding_wrist_offset = 0.0):

        self.is_sliding_wrist = is_sliding_wrist
        self.sliding_wrist_offset =  sliding_wrist_offset

        self.collision_margin = collision_margin

        self.rviz_process = rviz_process

        self.lft_default_q_wrt_ws = np.array([- np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0])
        self.rght_default_q_wrt_ws = np.array([- np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0, 0.0])
        self.lft_default_p_wrt_ws = np.array([0.0, 0.2, 0.3])
        self.rght_default_p_wrt_ws = np.array([0.0, - 0.2, 0.3])

        self.should_w = should_w
        self.should_roll = should_roll
        self.mount_h = mount_h
        
        self.is_soft_pose_cnstrnt = False

        self.weight_pos = 0
        self.weight_rot = 0
        self.weight_glob_man = 0

        self.urdf = None
        self.joint_names = None
        self.nq = 0
        self.nv = 0
        self.urdf_kin_dyn = None
        self.lbs = None
        self.ubs = None

        self.q = None
        self.q_dot = None
        self.q_design = None
        self.q_design_dot = None
        
        self.d_var_map = get_design_map() # retrieving design map (dangerous method)

        self.task_base_n_nodes = 2
        self.phase_number = 1 # number of phases of the task
        self.total_nnodes = 0
        self.filling_n_nodes = filling_n_nodes # filling nodes in between two base task nodes
        self.nodes_list = [] 

        self.dt = 0.0
        self.tf = 0.0

        self.prb = None

        self.solver_type = 'ipopt'
        self.slvr_opt = {"ipopt.tol": 0.00001,
                        "ipopt.max_iter": 1000,
                        "ipopt.constr_viol_tol": 0.1} 

        self.transcription_method = 'multiple_shooting'
        self.transcription_opts = dict(integrator='RK4') 

        self.slvr = None

        # kinematic quantities

        self.lft_inward_q = np.array([- np.sqrt(2.0)/2.0, - np.sqrt(2.0)/2.0, 0.0, 0.0])
        self.rght_inward_q = np.array([- np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0, 0.0])

        self.lft_pick_q = [] # where left arm will pick
        self.rght_pick_q = [] # where right arm will pick

        self.ws_link_pos = None
        self.ws_link_rot = None
        self.rarm_tcp_pos_rel_ws = None
        self.larm_tcp_pos_rel_ws = None
        self.rarm_tcp_rot_wrt_ws = None
        self.larm_tcp_rot_wrt_ws = None

        self.lft_tcp_pos_wrt_ws = None
        self.lft_tcp_rot_wrt_ws = None
        self.rght_tcp_pos_wrt_ws = None
        self.rght_tcp_rot_wrt_ws = None


        self.p_ref_left_init = None
        self.p_ref_right_init  = None
        self.q_ref_left_init  = None
        self.q_ref_right_init  = None

        self.p_ref_left_trgt = None
        self.p_ref_right_trgt = None
        self.q_ref_left_trgt = None
        self.q_ref_right_trgt = None

        self.coll_links_pos_rght = None
        self.coll_links_pos_lft = None

        self.compute_nodes(0, self.filling_n_nodes)
    
    def create_markers(self):

        self.lft_trgt_marker_topic = "repair/lft_trgt"
        self.rght_trgt_marker_topic = "repair/rght_trgt"
        self.rviz_marker_gen = MarkerGen(node_name = "marker_gen_cart_task")
        self.rviz_marker_gen.add_marker("working_surface_link", [0.0, 0.3, 0.4], self.lft_trgt_marker_topic,\
                                "left_trgt", 0.3) 
        self.rviz_marker_gen.add_marker("working_surface_link", [0.0, - 0.3, 0.4], self.rght_trgt_marker_topic,\
                                "rght_trgt", 0.3) 
        self.rviz_marker_gen.spin()

    def compute_nodes(self, init_node, filling_n_nodes):

        # HOW THE PROBLEM IS BUILT:
        #                    
        #                     
        # |                   |   
        # |                   |   
        # | | | | | | | | | | |  
        # 0                   1   

        # This simple cartesian task has only two base nodes: initial node and target node

        total_n_task_nodes = self.phase_number * filling_n_nodes + self.task_base_n_nodes # total number of nodes required to perform a single task
 
        self.total_nnodes = total_n_task_nodes

        final_node = init_node + total_n_task_nodes - 1 
        
        self.nodes_list.append(list(range(init_node, final_node + 1)))
        
        return final_node
    
    def set_ig(self, q_ig = None, q_dot_ig = None):

        if q_ig is not None:

            self.q.setInitialGuess(q_ig)

        if q_dot_ig is not None:

            self.q_dot.setInitialGuess(q_dot_ig)

    def init_prb(self, urdf_full_path, weight_pos = 0.001, weight_rot = 0.001,\
            weight_glob_man = 0.0001, is_soft_pose_cnstr = False,\
            tf_task = 4.0):

        ## All the main initializations for the prb are performed here ##

        self.is_soft_pose_cnstrnt = is_soft_pose_cnstr

        self.weight_pos = weight_pos / ( self.total_nnodes ) # scaling weights on the basis of the problem dimension
        self.weight_rot = weight_rot / ( self.total_nnodes )
        self.weight_glob_man = weight_glob_man / ( self.total_nnodes )

        n_int = self.total_nnodes - 1 # adding addditional filling nodes between nodes of two successive tasks
        self.prb = problem.Problem(n_int) 

        self.urdf = open(urdf_full_path, 'r').read()
        self.urdf_kin_dyn = casadi_kin_dyn.pycasadi_kin_dyn.CasadiKinDyn(self.urdf)
        self.tf = tf_task * len(self.nodes_list)
        self.dt = self.tf / n_int
        
        self.joint_names = self.urdf_kin_dyn.joint_names()
        if 'universe' in self.joint_names: self.joint_names.remove('universe')
        if 'floating_base_joint' in self.joint_names: self.joint_names.remove('floating_base_joint')
        
        self.nq = self.urdf_kin_dyn.nq()
        self.nv = self.urdf_kin_dyn.nv()

        self.lbs = self.urdf_kin_dyn.q_min() 
        self.ubs = self.urdf_kin_dyn.q_max()

        self.q = self.prb.createStateVariable('q', self.nq)
        self.q_dot = self.prb.createInputVariable('q_dot', self.nv)

        # THIS DEFINITIONS CAN CHANGE IF THE URDF CHANGES --> MIND THE URDF!!!
    
        self.q_design = self.q[self.d_var_map["mount_h"],\
                            self.d_var_map["should_w_l"], self.d_var_map["should_roll_l"], self.d_var_map["wrist_off_l"], \
                            self.d_var_map["should_w_r"], self.d_var_map["should_roll_r"], self.d_var_map["wrist_off_r"]] # design vector

        self.q_design_dot = self.q_dot[self.d_var_map["mount_h"],\
                            self.d_var_map["should_w_l"], self.d_var_map["should_roll_l"], self.d_var_map["wrist_off_l"], \
                            self.d_var_map["should_w_r"], self.d_var_map["should_roll_r"], self.d_var_map["wrist_off_r"]]

        self.p_ref_left_init = self.prb.createParameter('p_ref_left_init', 3)
        self.p_ref_right_init  = self.prb.createParameter('p_ref_right_init ', 3)
        self.q_ref_left_init  = self.prb.createParameter('q_ref_left_init ', 4)
        self.q_ref_right_init  = self.prb.createParameter('q_ref_right_init ', 4)

        self.p_ref_left_trgt = self.prb.createParameter('p_ref_left_trgt', 3)
        self.p_ref_right_trgt = self.prb.createParameter('p_ref_right_trgt', 3)
        self.q_ref_left_trgt = self.prb.createParameter('q_ref_left_trgt', 4)
        self.q_ref_right_trgt = self.prb.createParameter('q_ref_right_trgt', 4)

        self.prb.setDynamics(self.q_dot)
        self.prb.setDt(self.dt)  

        # getting useful kinematic quantities
        fk_ws = self.urdf_kin_dyn.fk("working_surface_link")
        self.ws_link_pos = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_pos"] # w.r.t. world
        self.ws_link_rot = fk_ws(q = np.zeros((self.nq, 1)).flatten())["ee_rot"] # w.r.t. world (3x3 rot matrix)

        fk_ws_arm1_link6 = self.urdf_kin_dyn.fk("arm_1_link_6")
        arm1_link6_pos = fk_ws_arm1_link6(q = self.q)["ee_pos"] # w.r.t. world
        arm1_link6_rot = fk_ws_arm1_link6(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        
        fk_ws_arm2_link6 = self.urdf_kin_dyn.fk("arm_2_link_6")
        arm2_link6_pos = fk_ws_arm2_link6(q = self.q)["ee_pos"] # w.r.t. world
        arm2_link6_rot = fk_ws_arm2_link6(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)

        fk_arm_r = self.urdf_kin_dyn.fk("arm_1_tcp")
        rarm_tcp_pos = fk_arm_r(q = self.q)["ee_pos"] # w.r.t. world
        rarm_tcp_rot = fk_arm_r(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        self.rarm_tcp_pos_rel_ws = rarm_tcp_pos - self.ws_link_pos # pos w.r.t. working surface in world frame

        fk_arm_l = self.urdf_kin_dyn.fk("arm_2_tcp")
        larm_tcp_pos = fk_arm_l(q = self.q)["ee_pos"] # w.r.t. world
        larm_tcp_rot = fk_arm_l(q = self.q)["ee_rot"] # w.r.t. world (3x3 rot matrix)
        self.larm_tcp_pos_rel_ws = larm_tcp_pos - self.ws_link_pos # pos w.r.t. working surface in world frame

        rarm_cocktail_pos = rarm_tcp_pos 
        rarm_cocktail_rot = rarm_tcp_rot
        self.rght_tcp_rot_wrt_ws = self.ws_link_rot.T @ rarm_cocktail_rot
        self.rght_tcp_pos_wrt_ws = self.ws_link_rot.T @ (rarm_cocktail_pos - self.ws_link_pos)

        larm_cocktail_pos = larm_tcp_pos
        larm_cocktail_rot = larm_tcp_rot
        self.lft_tcp_rot_wrt_ws = self.ws_link_rot.T @ larm_cocktail_rot
        self.lft_tcp_pos_wrt_ws = self.ws_link_rot.T @ (larm_cocktail_pos - self.ws_link_pos)

        self.coll_links_pos_rght = self.ws_link_rot.T @ (arm1_link6_pos - self.ws_link_pos)
        self.coll_links_pos_lft = self.ws_link_rot.T @ (arm2_link6_pos - self.ws_link_pos)

    def setup_prb(self,\
                epsi = epsi_default,\
                q_ig = None, q_dot_ig = None):
         
        self.epsi = epsi

        ## All the constraints and costs are set here ##

        # setting initial guesses
        if q_ig is not None:

            self.q.setInitialGuess(q_ig)

        if q_dot_ig is not None:
            
            self.q_dot.setInitialGuess(q_dot_ig)

        # lower and upper bounds for design variables and joint variables
        self.q.setBounds(self.lbs, self.ubs)
        
        # adapt collision model of the bent link (link5) to the wrist offset value
        # this is to account for variations link5's footprint as a results of the optimal
        # wrist offset 

        self.prb.createConstraint("adapt_link5_coll_model", \
                                self.q[self.d_var_map["should_roll_l"]] - \
                                self.q[self.d_var_map["should_roll_r"]])

        # roll and shoulder vars equal
        self.prb.createConstraint("same_roll", \
                                self.q[self.d_var_map["should_roll_l"]] - \
                                self.q[self.d_var_map["should_roll_r"]])

        self.prb.createConstraint("same_shoulder_w",\
                                self.q[self.d_var_map["should_w_l"]] - \
                                self.q[self.d_var_map["should_w_r"]])

        self.prb.createConstraint("same_wrist_offset",\
                                self.q[self.d_var_map["wrist_off_l"]] - \
                                self.q[self.d_var_map["wrist_off_r"]])

        # design vars equal on all nodes 
        self.prb.createConstraint("single_var_cntrnt",\
                            self.q_design_dot,\
                            nodes = range(0, (self.total_nnodes - 1)))

        if not self.is_sliding_wrist: # setting value for wrist offsets

            self.prb.createConstraint("wrist_offset_value",\
                    self.q[self.d_var_map["wrist_off_l"]] - self.sliding_wrist_offset,\
                    nodes = range(0, (self.total_nnodes - 1)))

        # TCPs inside working volume (assumed to be a box)
        ws_ub = np.array([1.2, 0.8, 1.0])
        ws_lb = np.array([-1.2, -0.8, 0.0])

        keep_tcp_in_ws_rght = self.prb.createConstraint("keep_tcp_in_ws_rght",\
                                                           self.rght_tcp_pos_wrt_ws)
        keep_tcp_in_ws_rght.setBounds(ws_lb, ws_ub)
        keep_tcp_in_ws_lft = self.prb.createConstraint("keep_tcp_in_ws_lft",\
                                                           self.lft_tcp_pos_wrt_ws)
        keep_tcp_in_ws_lft.setBounds(ws_lb, ws_ub)

        # TCPs above working surface
        keep_tcp1_above_ground = self.prb.createConstraint("keep_tcp1_above_ground",\
                                                           self.rarm_tcp_pos_rel_ws[2])
        keep_tcp1_above_ground.setBounds(0, cs.inf)
        keep_tcp2_above_ground = self.prb.createConstraint("keep_tcp2_above_ground",\
                                                           self.larm_tcp_pos_rel_ws[2])
        keep_tcp2_above_ground.setBounds(0, cs.inf)

        # add p2p collision task on y axis
        coll = self.prb.createConstraint("avoid_collision_on_y", \
                                self.coll_links_pos_lft[1] - self.coll_links_pos_rght[1])
        coll.setBounds(self.collision_margin - 0.001, cs.inf)

        # min inputs --> also corresponds to penalize large joint excursions
        self.prb.createIntermediateCost("max_global_manipulability",\
                        self.weight_glob_man * cs.sumsqr(self.q_dot))
        
        add_pose_cnstrnt("right_arm_init", self.prb, 0,\
                        pos = self.rght_tcp_pos_wrt_ws, rot = self.rght_tcp_rot_wrt_ws,\
                        pos_ref = self.p_ref_right_init, rot_ref = quat2rot(self.q_ref_right_init),\
                        pos_selection = ["x", "y", "z"],\
                        rot_selection = ["x", "y", "z"])

        add_pose_cnstrnt("right_arm_trgt", self.prb, self.total_nnodes - 1,\
                        pos = self.rght_tcp_pos_wrt_ws, rot = self.rght_tcp_rot_wrt_ws,\
                        pos_ref = self.p_ref_right_trgt, rot_ref = quat2rot(self.q_ref_right_trgt),\
                        pos_selection = ["x", "y", "z"],\
                        rot_selection = ["x", "y", "z"])
        
        add_pose_cnstrnt("left_arm_init", self.prb, 0,\
                        pos = self.lft_tcp_pos_wrt_ws, rot = self.lft_tcp_rot_wrt_ws,\
                        pos_ref = self.p_ref_left_init, rot_ref = quat2rot(self.q_ref_left_init), \
                        pos_selection = ["x", "y", "z"],\
                        rot_selection = ["x", "y", "z"])

        add_pose_cnstrnt("left_arm_trgt", self.prb, self.total_nnodes - 1,\
                        pos = self.lft_tcp_pos_wrt_ws, rot = self.lft_tcp_rot_wrt_ws,\
                        pos_ref = self.p_ref_left_trgt, rot_ref = quat2rot(self.q_ref_left_trgt), \
                        pos_selection = ["x", "y", "z"],\
                        rot_selection = ["x", "y", "z"])
        
        Transcriptor.make_method(self.transcription_method,\
                                self.prb,\
                                self.transcription_opts)

        self.slvr = solver.Solver.make_solver(self.solver_type, self.prb, self.slvr_opt)

    def start_loop(self):

        self.create_markers() # spawn markers node

        mark_lft_pos_trgt = None
        mark_lft_q_trgt = None
        mark_rght_pos_trgt = None
        mark_rght_q_trgt = None

        self.p_ref_left_init.assign(self.lft_default_p_wrt_ws)
        self.p_ref_right_init.assign(self.rght_default_p_wrt_ws)
        self.q_ref_left_init.assign(self.lft_default_q_wrt_ws)
        self.q_ref_right_init.assign(self.rght_default_q_wrt_ws)

        print("\n \nPlease move both markers in order to start the solution loop!!\n \n ")

        while (mark_lft_pos_trgt is None) or (mark_lft_q_trgt is None) \
            or (mark_rght_pos_trgt is None) or (mark_rght_q_trgt is None):
            
            # continue polling the positions until they become valid
            mark_lft_pos_trgt, mark_lft_q_trgt = self.rviz_marker_gen.getPose(self.lft_trgt_marker_topic)
            mark_rght_pos_trgt, mark_rght_q_trgt = self.rviz_marker_gen.getPose(self.rght_trgt_marker_topic)

            time.sleep(0.1)

        print("\n \nValid feedback from markers received! Starting solution loop ...\n \n ")

        while True:
            
            # continue polling
            mark_lft_pos_trgt, mark_lft_q_trgt = self.rviz_marker_gen.getPose(self.lft_trgt_marker_topic)
            mark_rght_pos_trgt, mark_rght_q_trgt = self.rviz_marker_gen.getPose(self.rght_trgt_marker_topic)

            self.p_ref_left_trgt.assign(mark_lft_pos_trgt)
            self.p_ref_right_trgt.assign(mark_rght_pos_trgt)
            self.q_ref_left_trgt.assign(mark_lft_q_trgt)
            self.q_ref_right_trgt.assign(mark_rght_q_trgt)

            t = time.time()

            self.slvr.solve()  # solving

            solution_time = time.time() - t

            print(f'\n Problem solved in {solution_time} s \n')
                    
            self.p_ref_left_init.assign(mark_lft_pos_trgt)
            self.p_ref_right_init.assign(mark_rght_pos_trgt)
            self.q_ref_left_init.assign(mark_lft_q_trgt)
            self.q_ref_right_init.assign(mark_rght_q_trgt)

            solution = self.slvr.getSolutionDict() # extracting solution
            q_sol = solution["q"]

            sol_replayer = ReplaySol(dt = self.dt, joint_list = self.joint_names, q_replay = q_sol) 
            sol_replayer.replay(is_floating_base = False, play_once = True)


   