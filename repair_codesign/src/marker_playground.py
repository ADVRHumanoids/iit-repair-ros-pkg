#!/usr/bin/env python3

from horizon.utils.resampler_trajectory import resampler
from horizon.ros.replay_trajectory import *
from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver
import os, argparse
from os.path import exists

import numpy as np

from datetime import datetime
from datetime import date
import time

import subprocess

import rospkg

from codesign_pyutils.ros_utils import ReplaySol
from codesign_pyutils.miscell_utils import str2bool,\
                                        get_min_cost_index

from codesign_pyutils.task_utils import generate_ig     

from codesign_pyutils.tasks import DoubleArmCartTask

## getting some useful information to be used for data storage
today = date.today()
today_is = today.strftime("%d-%m-%Y")
now = datetime.now()
current_time = now.strftime("_%H_%M_%S")

# useful paths
rospackage = rospkg.RosPack() # Only for taking the path to the leg package
urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
codesign_path = rospackage.get_path("repair_codesign")
results_path = codesign_path + "/test_results"

file_name = os.path.splitext(os.path.basename(__file__))[0]

# task-specific options
filling_n_nodes = 10
rot_error_epsi = 0.0000001

# longitudinal extension of grasped object
cocktail_size = 0.08

# seed used for random number generation
ig_seed = 654 # 1 the luckiest one as of now

# single task execution time
t_exec_task = 5

# transcription options (if used)
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

def main(args):

    # preliminary ops
    if args.gen_urdf:

        try:

            xacro_gen = subprocess.check_call(["xacro", "-o",\
                                               urdf_full_path,\
                                               xacro_full_path])
            
        except:

            print('Failed to generate URDF.')

    if args.launch_rviz:

        try:

            rviz_window = subprocess.Popen(["roslaunch",\
                                            "repair_urdf",\
                                            "repair_full_markers.launch"])

        except:

            print('Failed to launch RViz.')
    
    if  (not os.path.isdir(results_path)):

        os.makedirs(results_path)
    
    # some initializations
    q_ig_main = [None] * 1
    q_dot_ig_main = [None] * 1

    cart_task = DoubleArmCartTask(rviz_window, filling_n_nodes = filling_n_nodes)

    cart_task.init_prb(urdf_full_path, args.base_weight_pos, args.base_weight_rot,\
                    args.weight_global_manip, args.soft_pose_cnstrnt, 4.0)
    
    cart_task.setup_prb(rot_error_epsi)
    
    # generating initial guesses, based on the script arguments
    q_ig_main, q_dot_ig_main =  generate_ig(args, "",\
                                            cart_task,\
                                            1, ig_seed,\
                                            True)

    if args.use_init_guess:

        cart_task.set_ig(q_ig_main[0], q_dot_ig_main[0])

    # clear generated urdf file
    if exists(urdf_full_path): 

        os.remove(urdf_full_path)

    cart_task.start_loop()

    # # closing all child processes and exiting
    # rviz_window.terminate()
    # exit()
    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--gen_urdf', '-g', type=str2bool,\
                        help = 'whether to generate urdf from xacro', default = True)
    parser.add_argument('--launch_rviz', '-rvz', type=str2bool,\
                        help = 'whether to launch rviz or not', default = True)
    parser.add_argument('--rviz_replay', '-rpl', type=str2bool,\
                        help = 'whether to replay the solution on RViz', default = True)
    parser.add_argument('--use_init_guess', '-ig', type=str2bool,\
                        help = 'whether to use initial guesses between solution loops', default = True)
    parser.add_argument('--soft_pose_cnstrnt', '-spc', type=str2bool,\
                        help = 'whether to use soft pose constraints or not', default = False)
    parser.add_argument('--base_weight_pos', '-wp', type = np.double,\
                        help = 'base weight for position tracking (if using soft constraints)', default = 0.001)
    parser.add_argument('--base_weight_rot', '-wr', type = np.double,\
                        help = 'base weight for orientation tracking (if using soft constraints)', default = 0.001)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 0.01)
    parser.add_argument('--load_initial_guess', '-lig', type=str2bool,\
                        help = 'whether to load the initial guess from file', default = False)

    args = parser.parse_args()

    main(args)