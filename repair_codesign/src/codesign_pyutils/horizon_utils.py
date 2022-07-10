import numpy as np

import casadi as cs

from codesign_pyutils.math_utils import rot_error, rot_error2, get_cocktail_aux_rot

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

            pos_cnstrnt = prb.createConstraint("init_pos" + str(index), pos - pos_ref, nodes = nodes)

        if is_rot:

            rot_cnstrnt = prb.createConstraint("init_rot" + str(index), rot_error2(rot, rot_ref, epsi), nodes = nodes)

  else:

        pos_cnstrnt = None
        rot_cnstrnt = None

        if is_pos:

            pos_cnstrnt = prb.createIntermediateCost("init_pos_soft" + str(index), weight_pos * cs.sumsqr(pos - pos_ref), nodes = nodes)

        if is_rot:

            rot_cnstrnt = prb.createIntermediateCost("init_soft" + str(index), weight_rot * cs.sumsqr(rot_error2(rot, rot_ref, epsi)), nodes = nodes)

  return pos_cnstrnt, rot_cnstrnt