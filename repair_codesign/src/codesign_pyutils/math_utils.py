import casadi as cs
import numpy as np

def quat2rot(Q):
    
    # first element = real component 

    q = cs.vertcat(Q[0], Q[1], Q[2], Q[3])

    q = q / cs.sqrt(q[0]* q[0] + q[1]* q[1] + q[2]* q[2] + q[3]* q[3]) # normalize input quat
    
    R = cs.vertcat(cs.horzcat(2 * (q[0] * q[0] + q[1] * q[1]) - 1, 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])),\
                   cs.horzcat(2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[0] * q[0] + q[2] * q[2]) - 1, 2 * (q[2] * q[3] - q[0] * q[1])),\
                   cs.horzcat(2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 2 * (q[0] * q[0] + q[3] * q[3]) - 1))
              
    return R

def rot2quat(R, epsi = 0.0):

    # convert matrix to quaternion representation

    # quaternion Q ={eta, Epsi}
    # where eta = cos(theta/2), with theta belogs [- PI, PI] 
    # Epsi = sin(theta/2) * r, where r is the axis of rotation
    
    eta = 1/2 * cs.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1 + epsi)
    epsi_1 = 1/2 * cs.sign(R[2, 1] - R[1, 2]) * cs.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1 + epsi) 
    epsi_2 = 1/2 * cs.sign(R[0, 2] - R[2, 0]) * cs.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1 + epsi) 
    epsi_3 = 1/2 * cs.sign(R[1, 0] - R[0, 1]) * cs.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1 + epsi)
    
    Q = cs.vertcat(eta, epsi_1, epsi_2, epsi_3) # real part is conventionally assigned to the first component

    return Q

def Skew(vector):

    S = cs.vertcat( cs.horzcat(          0, - vector[2],   vector[1]), \
                    cs.horzcat(  vector[2],           0, - vector[0]), \
                    cs.horzcat(- vector[1] , vector[0] ,          0 ))

    
    return S

def rot_error(R_trgt, R_actual, epsi = 0.0):

    Q_trgt = rot2quat(R_trgt, epsi)
    Q_actual = rot2quat(R_actual, epsi)
    
    rot_err = Q_trgt[0] * Q_actual[1:4] - Q_actual[0] * Q_trgt[1:4] - cs.mtimes(Skew(Q_actual[1:4]), Q_trgt[1:4])
    
    # rot_err1 = Q_trgt[0] * Q_actual[1] - Q_actual[0] * Q_trgt[1] + Q_actual[3] * Q_trgt[2] - Q_actual[2] * Q_trgt[3]
    # rot_err2 = Q_trgt[0] * Q_actual[2] - Q_actual[0] * Q_trgt[2] - Q_actual[3] * Q_trgt[1] + Q_actual[1] * Q_trgt[3]
    # rot_err3 = Q_trgt[0] * Q_actual[3] - Q_actual[0] * Q_trgt[3] + Q_actual[2] * Q_trgt[1] - Q_actual[1] * Q_trgt[2]
    # return cs.vertcat(rot_err1, rot_err2, rot_err3)

    return rot_err


def rot_error2(R_trgt, R_actual, epsi = 0.0):

    R_err = R_actual @ R_trgt.T

    S = (R_err - R_err.T) / 2

    r = cs.vertcat(S[2, 1], S[0, 2], S[1, 0])

    return r / cs.sqrt(epsi + 1 + cs.trace(R_err))

def get_cocktail_aux_rot(R):

    # Conventionally, given a target frame for one arm
    # ^z
    # |
    # |
    # o_ _ _ > x
    #
    # The target reference frame for the other arm is 
    # chosen with same origin and the following orientation:
    # 
    # o_ _ _ > x
    # |
    # |
    # ˇ z
    # This is done to make the renowed "baretender" pose
    

    cocktail_aux_rot = cs.DM([[0.0, 1.0, 0.0],\
                                   [1.0, 0.0, 0.0],\
                                   [0.0, 0.0, - 1.0]]) 

    R_aux = cs.mtimes(R, cocktail_aux_rot)

    return R_aux