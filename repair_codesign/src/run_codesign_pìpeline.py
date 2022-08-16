#!/usr/bin/env python3
import os, argparse

import rospkg

from datetime import datetime
from datetime import date

import subprocess

from termcolor import colored

import numpy as np

from codesign_pyutils.miscell_utils import str2bool

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Pipeline script for the co-design of RePAIR project')

    # first level specific arguments
    parser.add_argument('--multistart_n_l1', '-msn_l1', type = int,\
                        help = '', default = 1296)
    parser.add_argument('--max_trials_factor_l1', '-mtfl1', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 5)

    parser.add_argument('--is_in_place_flip', '-iplf', type=str2bool,\
                        help = 'whether to use in place flip task', default = True)
    parser.add_argument('--is_biman_pick', '-ibp', type=str2bool,\
                        help = 'whether to use bimanual pick task', default = False)

    parser.add_argument('--ig_seed_l1', '-ig_l1', type = int,\
                        help = '', default = 1)
    parser.add_argument('--ipopt_verb_lev', '-ipopt_v', type = int,\
                        help = '', default = 1)
    parser.add_argument('--filling_nnodes', '-fnn', type = int,\
                        help = '', default = 0)
    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = False)
    parser.add_argument('--wrist_offset', '-wo', type = np.double,\
                        help = 'sliding_wrist_offset', default = 0.0)
    parser.add_argument('--is_sliding_wrist', '-isw', type = bool,\
                        help = 'is wrist off. is to be used as an additional codes variable', default = False)

    parser.add_argument('--n_y_samples_flip', '-nysf', type = int,\
                        help = 'number of y-axis samples on which tasks (flipping task) are placed', default = 4)
    parser.add_argument('--y_sampl_ub_flip', '-yubf', type = np.double,\
                        help = 'upper bound of the y sampling (bimanual task)', default = 0.4)
    parser.add_argument('--n_y_samples_biman', '-nysb', type = int,\
                        help = 'number of y-axis samples on which tasks(flipping task) are placed', default = 3)
    parser.add_argument('--y_sampl_ub_biman', '-yubb', type = np.double,\
                        help = 'upper bound of the y sampling (bimanual task)', default = 0.2)
                        
    # second level-specific arguments
    parser.add_argument('--multistart_n_l2', '-msn_l2', type=int,\
                        help = 'number of multistarts (per cluster) to use',
                        default = 72)
    parser.add_argument('--max_trials_factor_l2', '-mtfl2', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 20)
    parser.add_argument('--n_clust_l2', '-nc_l2', type=int,\
                        help = 'number of clusters to be generated', default = 15)
    parser.add_argument("--ig_seed_l2", '-ig_l2', type = int,\
                        help = '', default = 28)

    args = parser.parse_args()


    # unique id used for generation of results
    unique_id = date.today().strftime("%d-%m-%Y") + "-" +\
                        datetime.now().strftime("%H_%M_%S")

    # useful paths
    l1_dump_folder_name = "first_level"
    l2_dump_folder_name = "second_level"
    res_dir_basename = "test_results"
    res_dir_full_name = res_dir_basename + "_" + \
                unique_id
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package

    codesign_path = rospackage.get_path("repair_codesign")
    exec_path = codesign_path + "/src"

    l1_results_path = codesign_path + "/" + res_dir_basename + "/" + res_dir_full_name + "/" + l1_dump_folder_name
    l2_results_path = codesign_path + "/" + res_dir_basename + "/" + res_dir_full_name + "/" + l2_dump_folder_name

    os.chdir(exec_path) # change current path, so that executable can be run with check_call

    try:

        print(colored("\n--> STARTING FIRST LEVEL OPTIMIZATION....\n", "blue"))
        reset_term = subprocess.check_call(["reset"])
        # run first level (blocking --> we have to wait for data to be dumped to file)
        first_level_proc = subprocess.check_call(["./run_first_level_opt_on_workstation.py", \
                                    "-mst", \
                                    str(args.multistart_n_l1), \
                                    "-mtf", \
                                    str(args.max_trials_factor_l1), \
                                    "-igs", \
                                    str(args.ig_seed_l1),  \
                                    "-fnn", \
                                    str(args.filling_nnodes), \
                                    "-ipopt_v", \
                                    str(args.ipopt_verb_lev), \
                                    "-run_ext", \
                                    str(True), \
                                    "-id", \
                                    str(unique_id), \
                                    "-ma57", \
                                    str(args.use_ma57), \
                                    "-wo", \
                                    str(args.wrist_offset), \
                                    "-isw", \
                                    str(args.is_sliding_wrist), \
                                    "-nysf", \
                                    str(args.n_y_samples_flip),\
                                    "-nysb", \
                                    str(args.n_y_samples_biman),\
                                    "-yubf", \
                                    str(args.y_sampl_ub_flip), \
                                    "-yubb", \
                                    str(args.y_sampl_ub_biman), \
                                    "-dfn", \
                                    l1_dump_folder_name, \
                                    "-rdbs", \
                                    res_dir_basename, \
                                    "-iplf", \
                                    str(args.is_in_place_flip), \
                                    "-ibp", \
                                    str(args.is_biman_pick)])

        print(colored("\n--> FIRST LEVEL OPTIMIZATION FINISHED SUCCESSFULLY. \n", "blue"))

    except:

        print(colored('\n An exception occurred while running the first level of the codesign pipeline. Muy malo!!! \n', "red"))
    

    try:

        print(colored("\n--> STARTING SECOND LEVEL OPTIMIZATION....\n", "blue"))

        #run first level (blocking --> we have to wait for data to be dumped to file)
        second_level_proc = subprocess.check_call(["./run_second_level_opt_on_workstation.py", \
                                    "-d", \
                                    res_dir_full_name, \
                                    "-dfn", \
                                    l2_dump_folder_name,
                                    "-rdbs", \
                                    res_dir_basename, \
                                    "-ldn", \
                                    l1_dump_folder_name, \
                                    "-ipopt_v", \
                                    str(args.ipopt_verb_lev), \
                                    "-nc",\
                                    str(args.n_clust_l2), 
                                    "-ma57", \
                                    str(args.use_ma57), 
                                    "-igs", \
                                    str(args.ig_seed_l2),
                                    "-mtf", \
                                    str(args.max_trials_factor_l2), \
                                    "-mst",
                                    str(args.multistart_n_l2)
                                    ])

        print(colored("\n--> SECOND LEVEL OPTIMIZATION FINISHED SUCCESSFULLY. \n", "blue"))

    except:

        print(colored('\n An exception occurred while running the second level of the codesign pipeline. Muy malo!!! \n', "red"))
