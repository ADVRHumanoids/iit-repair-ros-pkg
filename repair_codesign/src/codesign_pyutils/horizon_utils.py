import numpy as np

import casadi as cs

from codesign_pyutils.math_utils import quat2rot, rot_error, rot_error2, rot_error3, get_cocktail_matching_rot

from codesign_pyutils.miscell_utils import check_str_list, rot_error_axis_sel_not_supp

rot_error_approach = "arturo" # options: "siciliano", "arturo", "traversaro"

from codesign_pyutils.defaults_vals import epsi_default

###############
# obsolete

# def add_bartender_cnstrnt(index, prb, nodes, posl, posr, rotl, rotr, is_pos = True,\
#                           is_rot = True, weight_pos = 1.0, weight_rot = 1.0,\
#                           is_soft = False, epsi = epsi_default):

#     if not is_soft:

#         pos_cnstrnt = None
#         rot_cnstrnt = None

#         if is_pos:

#             pos_cnstrnt = prb.createConstraint("keep_baretender_pos" + str(index),\
#                                                 posr - posl, nodes = nodes)

#         if is_rot:
            
#             if rot_error_approach == "siciliano":

#                 rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index),\
#                                     rot_error( get_cocktail_matching_rot(rotr), rotl, epsi),\
#                                     nodes = nodes)

#             if rot_error_approach == "arturo":
                
#                 rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index),\
#                                     rot_error2( get_cocktail_matching_rot(rotr), rotl, epsi),\
#                                     nodes = nodes)
            
#             if rot_error_approach == "traversaro":

#                 rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index),\
#                                     rot_error3( get_cocktail_matching_rot(rotr), rotl, epsi),\
#                                     nodes = nodes)

#             if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
#                 and rot_error_approach != "siciliano":

#                 raise Exception('Choose a valid rotation error computation approach')

    
#     else:

#         pos_cnstrnt = None
#         rot_cnstrnt = None

#         if is_pos:
            
#             pos_cnstrnt = prb.createIntermediateCost("keep_baretender_pos_soft" + str(index),\
#                                                      weight_pos * cs.sumsqr(posr - posl),
#                                                      nodes = nodes)

#         if is_rot:
            
#             if rot_error_approach == "siciliano":
                
#                 rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot_soft" + str(index),\
#                     weight_rot * cs.sumsqr(rot_error( get_cocktail_matching_rot(rotr), rotl, epsi)),\
#                     nodes = nodes)

#             if rot_error_approach == "arturo":
                
#                 rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot_soft" + str(index),\
#                     weight_rot * cs.sumsqr(rot_error2( get_cocktail_matching_rot(rotr), rotl, epsi)),\
#                     nodes = nodes)
            
#             if rot_error_approach == "traversaro":

#                 rot_cnstrnt = prb.createIntermediateCost("keep_baretender_rot_soft" + str(index),\
#                     weight_rot * (rot_error3( get_cocktail_matching_rot(rotr), rotl, epsi)),\
#                     nodes = nodes)

#             if rot_error_approach != "traversaro" and rot_error_approach != "arturo"\
#                  and rot_error_approach != "siciliano":

#                 raise Exception('Choose a valid rotation error computation approach')
                

#     return pos_cnstrnt, rot_cnstrnt

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
