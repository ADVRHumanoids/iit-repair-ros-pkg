import numpy as np

import casadi as cs

from codesign_pyutils.math_utils import rot_error, rot_error2, get_cocktail_aux_rot

def add_bartender_cnstrnt(index, prb, nodes, posl, posr, rotl, rotr, is_soft = False, epsi = 0.001):

    pos_cnstrnt = prb.createConstraint("keep_baretender_pos" + str(index), posr - posl, nodes = nodes)
    rot_cnstrnt = prb.createConstraint("keep_baretender_rot" + str(index), rot_error2( get_cocktail_aux_rot(rotr), rotl, epsi), nodes = nodes)
    
    return pos_cnstrnt, rot_cnstrnt

def add_lpose_cnstrnt(index, prb, nodes, pos, rot, pos_ref, rot_ref, weight_pos = 1.0, weight_rot = 1.0, is_only_pos = False, is_soft = False, epsi = 0.001):

  if is_soft:

      pos_cnstrnt = prb.createConstraint("init_left_tcp_pos" + str(index), pos - pos_ref, nodes = nodes)
  
      rot_cnstrnt = None

      if not is_only_pos:

          rot_cnstrnt = prb.createConstraint("init_left_tcp_rot" + str(index), rot_error2(rot, rot_ref, epsi), nodes = nodes)

  else:

      pos_cnstrnt = prb.createIntermediateCost("init_left_tcp_pos_soft" + str(index), weight_pos * cs.sumsqr(pos - pos_ref), nodes = nodes)

      rot_cnstrnt = None

      if not is_only_pos:

          rot_cnstrnt = prb.createIntermediateCost("init_left_tcp_soft" + str(index), weight_rot * cs.sumsqr(rot_error2(rot, rot_ref, epsi)), nodes = nodes)

  return pos_cnstrnt, rot_cnstrnt

def add_rpose_cnstrnt(index, prb, nodes, pos, rot, pos_ref, rot_ref, weight_pos = 1.0, weight_rot = 1.0, is_only_pos = False, is_soft = False, epsi = 0.001):

  if is_soft:

      pos_cnstrnt = prb.createConstraint("init_right_tcp_pos" + str(index), pos - pos_ref, nodes = nodes)
      
      rot_cnstrnt = None
      
      if not is_only_pos:

          rot_cnstrnt = prb.createConstraint("init_right_tcp_rot" + str(index), rot_error2(rot, rot_ref, epsi), nodes = nodes)

  else:

      pos_cnstrnt = prb.createIntermediateCost("init_right_tcp_pos_soft" + str(index), weight_pos * cs.sumsqr(pos - pos_ref), nodes = nodes)

      rot_cnstrnt = None

      if not is_only_pos:

          rot_cnstrnt = prb.createIntermediateCost("init_right_tcp_rot_soft" + str(index), weight_rot * cs.sumsqr(rot_error2(rot, rot_ref, epsi)), nodes = nodes)

  return pos_cnstrnt, rot_cnstrnt