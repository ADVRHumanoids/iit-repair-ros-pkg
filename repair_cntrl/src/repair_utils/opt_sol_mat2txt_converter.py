#!/usr/bin/env python3

## Script for converting mat files containing trajectories to CSV ##

import scipy.io 
import numpy as np
import csv	
from numpy import genfromtxt

mat_path = "/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/replay_directory/opt/"
mat_name = "prova.mat"

dump_path = mat_path

solution_names = ["q_p", "q_p_dot", "tau", "dt_opt"]

data = scipy.io.loadmat(mat_path + mat_name)

q_p = data[solution_names[0]]
q_p_dot = data[solution_names[1]]
tau = data[solution_names[2]]
dt_op = data[solution_names[3]]

np.savetxt(dump_path +'q_p.csv', q_p, delimiter=" ")
np.savetxt(dump_path +'q_p_dot.csv', q_p_dot, delimiter=" ")
np.savetxt(dump_path +'tau.csv', tau, delimiter=" ")
np.savetxt(dump_path +'dt_opt.csv', dt_op, delimiter=" ")

q_p = np.loadtxt(dump_path +'q_p.csv', ndmin = 2)
q_p_dot = np.loadtxt(dump_path +'q_p_dot.csv', ndmin = 2)
tau = np.loadtxt(dump_path +'tau.csv', ndmin = 2)
dt = np.loadtxt(dump_path +'dt_opt.csv', ndmin = 2)

print("dati:")
print(q_p.shape)
print(q_p_dot.shape)
print(tau.shape)
print(dt.shape)
