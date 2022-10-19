#!/usr/bin/env python3
import os, argparse

import rospkg

from datetime import datetime
from datetime import date

import subprocess

from termcolor import colored

import numpy as np

from codesign_pyutils.miscell_utils import str2bool

from codesign_pyutils.post_proc_utils import PostProcS3

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Pipeline script for the co-design of RePAIR project')

    # first level specific arguments
    parser.add_argument('--multistart_n_s1', '-msn_s1', type = int,\
                        help = '', default = 432)
    parser.add_argument('--max_trials_factor_s1', '-mtfs1', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 10)

    parser.add_argument('--is_in_place_flip', '-iplf', type=str2bool,\
                        help = 'whether to use in place flip task', default = True)
    parser.add_argument('--is_biman_pick', '-ibp', type=str2bool,\
                        help = 'whether to use bimanual pick task', default = True)

    parser.add_argument('--ig_seed_s1', '-is_s1', type = int,\
                        help = '', default = 629)
    parser.add_argument('--ipopt_verb_lev', '-ipopt_v', type = int,\
                        help = '', default = 1)
    parser.add_argument('--filling_nnodes', '-fnn', type = int,\
                        help = '', default = 3)
    parser.add_argument('--use_ma57', '-ma57', type=str2bool,\
                        help = 'whether to use ma57 linear solver or not', default = False)
    parser.add_argument('--wrist_offset', '-wo', type = np.double,\
                        help = 'sliding_wrist_offset', default = 0.0)
    parser.add_argument('--is_sliding_wrist', '-isw', type = bool,\
                        help = 'is wrist off. is to be used as an additional codes variable', default = True)

    parser.add_argument('--n_y_samples_flip', '-nysf', type = int,\
                        help = 'number of y-axis samples on which tasks (flipping task) are placed', default = 3)
    parser.add_argument('--y_sampl_ub_flip', '-yubf', type = np.double,\
                        help = 'upper bound of the y sampling (bimanual task)', default = 0.3)
    parser.add_argument('--n_y_samples_biman', '-nysb', type = int,\
                        help = 'number of y-axis samples on which tasks(flipping task) are placed', default = 3)
    parser.add_argument('--y_sampl_ub_biman', '-yubb', type = np.double,\
                        help = 'upper bound of the y sampling (bimanual task)', default = 0.2)
    
    parser.add_argument('--use_static_tau', '-ustau', type=str2bool,\
                        help = 'whether to use the static tau minimization cost', default = True)
    parser.add_argument('--use_classical_man', '-ucm', type=str2bool,\
                        help = 'whether to use the classical man. cost', default = True)
    parser.add_argument('--weight_global_manip', '-wman', type = np.double,\
                        help = 'weight for global manipulability cost function', default = 1.0)
    parser.add_argument('--weight_class_manip', '-wclass', type = np.double,\
                        help = 'weight for classical manipulability cost function', default = 0.1)
    parser.add_argument('--weight_static_tau', '-wstau', type = np.double,\
                        help = 'weight static torque minimization term', default = 0.1)

    # second level-specific arguments
    parser.add_argument('--multistart_n_s3', '-msn_s3', type=int,\
                        help = 'number of multistarts (per cluster) to use',
                        default = 25)
    parser.add_argument('--max_trials_factor_s3', '-mtfl3', type=int,\
                        help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
                        default = 20)
    parser.add_argument('--n_clust_s2', '-nc_s2', type=int,\
                        help = 'number of clusters to be generated', default = 20)
    parser.add_argument("--ig_seed_s3", '-ig_s3', type = int,\
                        help = '', default = 731)
    
    # cl man reference generation-specific arguments
    # parser.add_argument('--gen_cl_man_ref', '-gen_clmr', type=bool,\
    #                     help = 'whether to run the cl. manipulability reference generation',
    #                     default = True)
    # parser.add_argument('--max_trials_factor_clmr', '-mtfl3', type=int,\
    #                     help = 'for each multistart node, at best max_trials_factor new solutions will be tried to obtain an optimal solution',
    #                     default = 20)

    args = parser.parse_args()


    # unique id used for generation of results
    unique_id = date.today().strftime("%d-%m-%Y") + "-" +\
                        datetime.now().strftime("%H_%M_%S")

    # useful paths
    s1_dump_folder_name = "first_step"
    s3_dump_folder_name = "second_step"
    res_dir_basename = "test_results"
    res_dir_full_name = res_dir_basename + "_" + \
                unique_id
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package

    codesign_path = rospackage.get_path("repair_codesign")
    exec_path = codesign_path + "/src"

    s1_results_path = codesign_path + "/" + res_dir_basename + "/" + res_dir_full_name + "/" + s1_dump_folder_name
    s3_results_path = codesign_path + "/" + res_dir_basename + "/" + res_dir_full_name + "/" + s3_dump_folder_name

    #generating urdf
    urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
    urdf_name = "repair_full"
    urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
    xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
    sliding_wrist_command = "is_sliding_wrist:=" + "true"
    gen_coll_command = "gen_coll:=" + "true"

    coll_yaml_name = "arm_coll.yaml"
    coll_yaml_path = rospackage.get_path("repair_urdf") + "/config/" + coll_yaml_name

    cost_weights_filename = "codes_cost_rel_weights.yaml"
    cost_weights_path = rospackage.get_path("repair_codesign") + "/config/" + cost_weights_filename

    try:

        print(colored("\n--> GENERATING URDF...\n", "blue"))
        xacro_gen = subprocess.check_call(["xacro",\
                                        xacro_full_path, \
                                        sliding_wrist_command, \
                                        gen_coll_command, \
                                        "-o", 
                                        urdf_full_path])

        print(colored("\n--> URDF GENERATED SUCCESSFULLY. \n", "blue"))

    except:

        print(colored('FAILED TO GENERATE URDF.', "red"))


    os.chdir(exec_path) # change current path, so that executable can be run with check_call

    if args.multistart_n_s1 > 0:

        # try:

        print(colored("\n--> STARTING FIRST LEVEL OPTIMIZATION....\n", "blue"))
        reset_term = subprocess.check_call(["reset"])
        # run first level (blocking --> we have to wait for data to be dumped to file)
        first_step_proc = subprocess.check_call(["./run_1st_step_opt_on_workstation.py", \
                                    "-mst", \
                                    str(args.multistart_n_s1), \
                                    "-mtf", \
                                    str(args.max_trials_factor_s1), \
                                    "-igs", \
                                    str(args.ig_seed_s1),  \
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
                                    s1_dump_folder_name, \
                                    "-rdbs", \
                                    res_dir_basename, \
                                    "-iplf", \
                                    str(args.is_in_place_flip), \
                                    "-ibp", \
                                    str(args.is_biman_pick), \
                                    "-urdf", \
                                    urdf_full_path, \
                                    "-coll", \
                                    coll_yaml_path, 
                                    "-weights", \
                                    cost_weights_path, 
                                    "-ustau", 
                                    str(args.use_static_tau),
                                    "-ucm", 
                                    str(args.use_classical_man), 
                                    "-wman", 
                                    str(args.weight_global_manip),
                                    "-wclass", 
                                    str(args.weight_class_manip),
                                    "-wstau", 
                                    str(args.weight_static_tau)])

        print(colored("\n--> FIRST LEVEL OPTIMIZATION FINISHED SUCCESSFULLY. \n", "blue"))

        # except:

            # print(colored('\n An exception occurred while running the first level of the codesign pipeline. Muy malo!!! \n', "red"))
    

    if args.multistart_n_s3 > 0:

        try:

            print(colored("\n--> STARTING SECOND LEVEL OPTIMIZATION....\n", "blue"))

            #run first level (blocking --> we have to wait for data to be dumped to file)
            second_step_proc = subprocess.check_call(["./run_2nd_3rd_step_opt_on_workstation.py", \
                                        "-d", \
                                        res_dir_full_name, \
                                        "-dfn", \
                                        s3_dump_folder_name,
                                        "-rdbs", \
                                        res_dir_basename, \
                                        "-ldn", \
                                        s1_dump_folder_name, \
                                        "-ipopt_v", \
                                        str(args.ipopt_verb_lev), \
                                        "-nc",\
                                        str(args.n_clust_s2), 
                                        "-ma57", \
                                        str(args.use_ma57), 
                                        "-igs", \
                                        str(args.ig_seed_s3),
                                        "-mtf", \
                                        str(args.max_trials_factor_s3), \
                                        "-mst",
                                        str(args.multistart_n_s3), \
                                        "-urdf", \
                                        urdf_full_path, \
                                        "-coll", \
                                        coll_yaml_path, \
                                        "-weights", \
                                        cost_weights_path])

            print(colored("\n--> SECOND LEVEL OPTIMIZATION FINISHED SUCCESSFULLY. \n", "blue"))

        except:

            print(colored('\n An exception occurred while running the second level of the codesign pipeline. Muy malo!!! \n', "red"))


    # if args.run_l3_postp:

    #     try:

    #         print(colored("\n--> PERFORMING POST-PROCESSING STEPS FOR 3rd STEP....\n", "blue"))

    #         postprs2s3 = PostProcS3(codesign_path + "/" + res_dir_basename)

    #     except:

    #         print(colored('\n An exception occurred while running 3rd step postprocessing. Muy malo!!! \n', "red"))

    # if args.gen_cl_man_ref:

    #     try:

    #         print(colored("\n--> PERFORMING POST-PROCESSING STEPS FOR 3rd STEP....\n", "blue"))

    #         postprs2s3 = PostProcS3(codesign_path + "/" + res_dir_basename)

    #         # generate cl man references
    #         second_step_proc = subprocess.check_call(["./gen_cl_man_ref.py", \
    #                                     "-mst", \
    #                                     args.max_trials_factor_clmr, \
    #                                     ])

    #     except:

    #         print(colored('\n An exception occurred while running 3rd step postprocessing. Muy malo!!! \n', "red"))

    
