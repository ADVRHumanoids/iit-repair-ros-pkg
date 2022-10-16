#!/usr/bin/env python3

from numpy import True_
from horizon.utils.resampler_trajectory import resampler
from horizon.ros.replay_trajectory import *

import os, argparse
from os.path import exists

import subprocess

import rospkg

from codesign_pyutils.ros_utils import ReplaySol
from codesign_pyutils.miscell_utils import str2bool,\
                                        get_min_cost_index
            
from codesign_pyutils.tasks import TaskGen
from codesign_pyutils.load_utils import LoadSols

# useful paths
rospackage = rospkg.RosPack() # Only for taking the path to the leg package

urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"

codesign_path = rospackage.get_path("repair_codesign")

results_path = codesign_path + "/test_results"

replay_folder_name = "replay_directory" 
replay_base_path = results_path  + "/" + replay_folder_name

# resample solutions before replaying
refinement_scale = 10

def main(args):

    sliding_wrist_command = "is_sliding_wrist:=" + "true"
    gen_coll_command = "gen_coll:=" + str(args.gen_coll).lower()
    use_updated_meshes = "use_updated_wrist:=" + str(args.use_updated_meshes).lower()
    # preliminary ops

    try:

        
        # print(sliding_wrist_command)
        xacro_gen = subprocess.check_call(["xacro",\
                                        xacro_full_path, \
                                        sliding_wrist_command, \
                                        gen_coll_command, \
                                        use_updated_meshes, \
                                        "-o", 
                                        urdf_full_path])

    except:

        print('Failed to generate URDF.')


    try:

        rviz_window = subprocess.Popen(["roslaunch",\
                                        "repair_urdf",\
                                        "repair_full.launch", \
                                        sliding_wrist_command,\
                                        gen_coll_command, \
                                        use_updated_meshes])

    except:

        print('Failed to launch RViz.')
    
    # only used to parse urdf
    dummy_task = TaskGen()

    dummy_task.add_in_place_flip_task(0)

    # initialize problem
    dummy_task.init_prb(urdf_full_path)

    # clear generated urdf file
    if exists(urdf_full_path): 

        os.remove(urdf_full_path)

    sol_loader = LoadSols(replay_base_path)
    
    n_opt_sol = len(sol_loader.opt_data)

    q_replay = [None] * n_opt_sol

    opt_costs = np.zeros((n_opt_sol, 1)).flatten()

    for i in range(n_opt_sol):

        opt_costs[i] = sol_loader.opt_data[i]["opt_cost"]

    opt_index = np.where(opt_costs == min(opt_costs))[0][0]

    while True:

        if args.replay_only_best:

            if args.resample_sol:
                    
                    dt_res = sol_loader.task_info_data["dt"][0][0] / refinement_scale

                    q_replay = resampler(sol_loader.opt_data[opt_index]["q"], sol_loader.opt_data[opt_index]["q_dot"],\
                                            sol_loader.task_info_data["dt"][0][0], dt_res,\
                                            {'x': dummy_task.q, 'p': dummy_task.q_dot,\
                                            'ode': dummy_task.q_dot, 'quad': 0})

                    sol_replayer = ReplaySol(dt_res,
                                                joint_list = dummy_task.joint_names,
                                                q_replay = q_replay, \
                                                srt_msg = "\nReplaying solution...")

            else:
                
                q_replay = sol_loader.opt_data[opt_index]["q"]

                sol_replayer = ReplaySol(dt = sol_loader.task_info_data["dt"][0][0],\
                                            joint_list = dummy_task.joint_names,\
                                            q_replay = q_replay, \
                                            srt_msg = "\nReplaying solution...")
                        
            sol_replayer.sleep(0.5)
            sol_replayer.replay(is_floating_base = False, play_once = True)

        else:

            for i in range(n_opt_sol):

                if args.resample_sol:
                    
                    dt_res = sol_loader.task_info_data["dt"][0][0] / refinement_scale

                    q_replay = resampler(sol_loader.opt_data[i]["q"], sol_loader.opt_data[i]["q_dot"],\
                                            sol_loader.task_info_data["dt"][0][0], dt_res,\
                                            {'x': dummy_task.q, 'p': dummy_task.q_dot,\
                                            'ode': dummy_task.q_dot, 'quad': 0})

                    sol_replayer = ReplaySol(dt_res,
                                                joint_list = dummy_task.joint_names,
                                                q_replay = q_replay, \
                                                srt_msg = "\nReplaying solution...")

                else:
                    
                    q_replay = sol_loader.opt_data[i]["q"]

                    sol_replayer = ReplaySol(dt = sol_loader.task_info_data["dt"][0][0],\
                                                joint_list = dummy_task.joint_names,\
                                                q_replay = q_replay, \
                                                srt_msg = "\nReplaying solution...")
                        
                sol_replayer.sleep(0.5)
                sol_replayer.replay(is_floating_base = False, play_once = True)

    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = True)
    parser.add_argument('--replay_only_best', '-rob', type=str2bool,\
                        help = 'whether to only the best solution', default = True)
    parser.add_argument('--gen_coll', '-coll', type=str2bool,\
                        help = 'whether to generate collisions', default = True)
    parser.add_argument('--use_updated_meshes', '-uum', type=str2bool,\
                        help = 'whether to use updated meshes', default = False)

    args = parser.parse_args()

    main(args)