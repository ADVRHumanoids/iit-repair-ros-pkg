import casadi as cs
import numpy as np

def rot2quat(R):

    # convert matrix to quaternion representation

    # quaternion Q ={eta, Epsi}
    # where eta = cos(theta/2), with theta belogs [- PI, PI] 
    # Epsi = sin(theta/2) * r, where r is the axis of rotation

    eta = 1/2 * cs.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
    epsi_1 = 1/2 * cs.sign(R[2, 1] - R[1, 2]) * cs.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1) 
    epsi_2 = 1/2 * cs.sign(R[0, 2] - R[2, 0]) * cs.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1) 
    epsi_3 = 1/2 * cs.sign(R[1, 0] - R[0, 1]) * cs.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)
    
    Q = cs.vertcat(eta, epsi_1, epsi_2, epsi_3)
    return Q

def rot_error(R_trgt, R_actual):

    Q_trgt = rot2quat(R_trgt)
    Q_actual = rot2quat(R_actual)

    rot_err1 = Q_trgt[0] * Q_actual[1] - Q_actual[0] * Q_trgt[1] + Q_actual[3] * Q_trgt[2] - Q_actual[2] * Q_trgt[3]
    rot_err2 = Q_trgt[0] * Q_actual[2] - Q_actual[0] * Q_trgt[2] - Q_actual[3] * Q_trgt[1] + Q_actual[1] * Q_trgt[3]
    rot_err3 = Q_trgt[0] * Q_actual[3] - Q_actual[0] * Q_trgt[3] + Q_actual[2] * Q_trgt[1] - Q_actual[1] * Q_trgt[2]

    return (rot_err1 * rot_err1 + rot_err2 * rot_err2 + rot_err3 * rot_err3)

def rot_error2(R_trgt, R_actual):

    R_err = R_trgt * cs.transpose(R_actual) # R_trgt * R_actual^T should be the identity matrix if error = 0

    I = np.zeros((3, 3))
    I[0, 0] = 1
    I[1, 1] = 1
    I[2, 2] = 1

    Err = R_err - I

    err = Err[0, 0] * Err[0, 0] + Err[1, 0] * Err[1, 0] + Err[2, 0] * Err[2, 0] + \
          Err[0, 1] * Err[0, 1] + Err[1, 1] * Err[1, 1] + Err[2, 1] * Err[2, 1] + \
          Err[0, 2] * Err[0, 2] + Err[1, 2] * Err[1, 2] + Err[2, 2] * Err[2, 2] 

    return err