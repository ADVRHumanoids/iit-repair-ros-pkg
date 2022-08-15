#!/usr/bin/env python3

import argparse

import rospkg
        
from codesign_pyutils.post_proc_utils import PostProc2ndLev

def main(args):

    # useful paths
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    codesign_path = rospackage.get_path("repair_codesign")

    results_path = codesign_path + "/test_results/" + args.res_dirname

    dump_dirname = "2nd_lev_postproc"

    post_proc = PostProc2ndLev(results_path, 
                    dump_dirname = dump_dirname) # loads 2nd level solutions and dumps general and postproc. info to file

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='2nd level post-process script')
    parser.add_argument('--res_dirname', '-d', type=str,\
                        help = 'directory name from where results are to be loaded', default = "load_dir")

    args = parser.parse_args()

    main(args)