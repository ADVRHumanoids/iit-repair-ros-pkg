import numpy as np

import casadi as cs

from codesign_pyutils.math_utils import quat2rot, rot_error, rot_error2, rot_error3, get_cocktail_matching_rot

from codesign_pyutils.miscell_utils import check_str_list, rot_error_axis_sel_not_supp

rot_error_approach = "arturo" # options: "siciliano", "arturo", "traversaro"

from codesign_pyutils.misc_definitions import epsi_default

#############

def add_pose_cnstrnt(unique_id, prb, nodes, pos = None, rot = None, pos_ref = None, rot_ref = None,\
                     pos_selection = ["x", "y", "z"], rot_selection = ["x", "y", "z"],\
                     weight_pos = 1.0, weight_rot = 1.0, is_soft = False, epsi = epsi_default):

    if ((pos is None) or (pos_ref is None)) and ( (rot is None) or (rot_ref is None) ):

        raise Exception("Missing necessary arguments for add_pose_cnstrnt()!!")

    # now it is guaranteed that at least one pair between [pos, pos_ref] and [rot, rot_ref] was provided
    is_pos = False
    is_rot = False

    if pos is not None:

        is_pos = True

    if rot is not None:

        is_rot = True

    if not is_soft:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_selector = check_str_list(comp_list = ["x", "y", "z"], input = pos_selection)
            
            if len(pos_selector) != 0:

                pos_cnstrnt = prb.createConstraint("pos_" + str(unique_id),\
                                                pos[pos_selector] - pos_ref[pos_selector],\
                                                nodes = nodes)

        if is_rot:
            
            rot_selector = check_str_list(comp_list = ["x", "y", "z"], input = rot_selection)

            if len(rot_selector) != 0:

                if rot_error_approach == "siciliano":
                    
                    rot_error_axis_sel_not_supp(rot_selector, rot_error_approach) # check if user tried to set axis --> in case, throw error

                    rot_cnstrnt = prb.createConstraint("rot_" + str(unique_id),\
                                        rot_error(rot_ref, rot, epsi)[rot_selector], nodes = nodes)

                if rot_error_approach == "arturo":
                    
                    rot_cnstrnt = prb.createConstraint("rot_" + str(unique_id),\
                                        rot_error2(rot_ref, rot, epsi)[rot_selector], nodes = nodes)
                
                if rot_error_approach == "traversaro":
                    
                    rot_cnstrnt = prb.createConstraint("rot_" + str(unique_id),\
                                        rot_error3(rot_ref, rot, epsi)[rot_selector], nodes = nodes)

                if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
                    and rot_error_approach != "siciliano":

                    raise Exception('Choose a valid rotation error computation approach')
            
    else:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_selector = check_str_list(comp_list = ["x", "y", "z"], input = pos_selection)
            
            if len(pos_selector) != 0:

                if pos[pos_selector].shape[0] != 0: # necessary only if using soft constraints
                
                    if pos[pos_selector].shape[0] == 1: # do not use cs.sumsqr
    
                        pos_cnstrnt = prb.createIntermediateCost("pos_soft_" + str(unique_id),\
                                                        weight_pos * cs.power(pos[pos_selector] - pos_ref[pos_selector], 2),\
                                                        nodes = nodes)

                    else:

                        pos_cnstrnt = prb.createIntermediateCost("pos_soft_" + str(unique_id),\
                                                        weight_pos * cs.sumsqr(pos[pos_selector] - pos_ref[pos_selector]),\
                                                        nodes = nodes)

        if is_rot:
            
            rot_selector = check_str_list(comp_list = ["x", "y", "z"], input = rot_selection)
            
            if len(rot_selector) != 0:

                if rot_error_approach == "siciliano":
                    
                    rot_error_axis_sel_not_supp(rot_selector, rot_error_approach) # check if user tried to set axis --> in case, throw error

                    rot_cnstrnt = prb.createIntermediateCost("rot_soft_" + str(unique_id),\
                                weight_rot * cs.sumsqr(rot_error(rot_ref, rot, epsi)[rot_selector]),\
                                nodes = nodes)

                if rot_error_approach == "arturo":
                    
                    if len(rot_selector) == 1: # do not use sumsqr
                        
                        rot_cnstrnt = prb.createIntermediateCost("rot_soft_" + str(unique_id),\
                                    weight_rot * (rot_error2(rot_ref, rot, epsi)[rot_selector]),\
                                    nodes = nodes)

                    else:

                        rot_cnstrnt = prb.createIntermediateCost("rot_soft_" + str(unique_id),\
                                    weight_rot * cs.sumsqr(rot_error2(rot_ref, rot, epsi)[rot_selector]),\
                                    nodes = nodes)
                
                if rot_error_approach == "traversaro":
                    
                    rot_error_axis_sel_not_supp(rot_selector, rot_error_approach) # check if user tried to set axis --> in case, throw error

                    rot_cnstrnt = prb.createIntermediateCost("rot_soft_" + str(unique_id),\
                                weight_rot * (rot_error3(rot_ref, rot, epsi)),\
                                nodes = nodes)

                if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
                    and rot_error_approach != "siciliano":

                    raise Exception('Choose a valid rotation error computation approach')
      

    return pos_cnstrnt, rot_cnstrnt

class SimpleCollHandler:

    def __init__(self,
                kindyn,
                q_p, 
                prb,
                nodes = None,
                collision_radii = None,
                tcp_contact_nodes = None, 
                link_names = [["arm_1_link_6", "arm_1_link_4", "arm_1_link_3"],\
                               ["arm_2_link_6", "arm_2_link_4", "arm_2_link_3"]]):

        self.kindyn = kindyn

        self.q_p = q_p

        self.link_names = link_names

        self.prb = prb

        self.nodes = nodes
        self.tcp_contact_nodes = []
        if (tcp_contact_nodes is not None) and (not len(tcp_contact_nodes) == 0):

            self.tcp_contact_nodes = tcp_contact_nodes

        self.filtered_nodes = self.remove_tcp_coll_nodes()

        # check dimension consistency between collision radii and link names
        if len(collision_radii[0]) != len(link_names[0]) or len(collision_radii[1]) != len(link_names[1]):

            raise Exception("SimpleCollHandler: dimesion mismatch between collision radii and link names!")

        # building collision radii dictionary (link_name -> collision radius)
        self.collision_radius_default = 0.05
        self.collision_radii = {}
        for i in range(len(link_names)): # iterate between the two arms

            for j in range(len(link_names[i])): # iterate through each link name 

                if collision_radii is None:

                    self.collision_radii[link_names[i][j]] = self.collision_radius_default

                else:

                    self.collision_radii[link_names[i][j]] = collision_radii[i][j]

        # building collision margins (two-level) dictionary (collision pair -> total collision margin)
        self.collision_margins = {}
        for i in range(len(link_names[0])):

            self.collision_margins[link_names[0][i]] = {}
            for j in range(len(link_names[1])):
                    
                    self.collision_margins[link_names[0][i]][link_names[1][j]] =\
                        self.collision_radii[link_names[0][i]] + \
                        self.collision_radii[link_names[1][j]]
        
        self.fks = {}

        self.coll_cnstrnts = []
        
        if not len(link_names) == 2:

            raise Exception("SimpleCollHandler: you have to specify exactly two collision groups")

        for i in range(len(link_names)):

            self.fks[i] = [None] * len(link_names[i])

        for i in range(len(link_names)):

            for j in range(len(link_names[i])):

                self.fks[link_names[i][j]] = self.get_link_abs_pos(link_names[i][j])
        
        self.collision_mask = {}
        for i in range(len(link_names[0])):
            
            self.collision_mask[link_names[0][i]] = {}
            for j in range(len(link_names[1])):

                self.collision_mask[link_names[0][i]][link_names[1][j]] = True
        
        # setting collision constraints on the collision pairs
        # defined by the collision mask
        for i in range(len(link_names[0])):

            for j in range(len(link_names[1])):
                
                if self.collision_mask[link_names[0][i]][link_names[1][j]]:
                    
                    if "tcp" in link_names[0][i] and "tcp" in link_names[1][j]: # tcps2tcps

                        self.coll_cnstrnts.append(self.add_p2p_coll_constr(self.prb,
                                            link_names[0][i],
                                            link_names[1][j],
                                            self.filtered_nodes))
                    
                    else: # all other pairs

                        self.coll_cnstrnts.append(self.add_p2p_coll_constr(self.prb,
                                            link_names[0][i],
                                            link_names[1][j],
                                            self.nodes))

    def remove_tcp_coll_nodes(self):

        n_nodes_prb = self.prb.getNNodes()

        nodes = None

        if self.nodes is None:
            
            self.nodes = [*range(n_nodes_prb)]

        nodes = self.nodes

        for i in range(len(self.tcp_contact_nodes)):

            nodes.remove(self.tcp_contact_nodes[i])

        return nodes

    def get_link_abs_pos(self, link_name):

        fk_link = cs.Function.deserialize(self.kindyn.fk(link_name)) 
        frame_pos = fk_link(q = self.q_p)["ee_pos"]

        return frame_pos

    def d_2(self, p1, p2):

        d_squared = cs.sumsqr(p2 - p1)

        return d_squared

    def add_p2p_coll_constr(self, prb,\
                        link1, link2, nodes):

        cnstrnt = prb.createConstraint(link1 + "_" + link2 + "coll",\
                self.d_2(self.fks[link1], self.fks[link2]),\
                nodes)

        cnstrnt.setBounds(self.collision_margins[link1][link2], cs.inf)

        return cnstrnt
        
