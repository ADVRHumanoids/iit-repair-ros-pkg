#!/usr/bin/env python3

from horizon import problem
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

from codesign_pyutils.ros_utils import FramePub, ReplaySol
from codesign_pyutils.miscell_utils import str2bool,\
                                        wait_for_confirmation,\
                                        get_min_cost_index
from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.task_utils import do_one_solve_pass, \
                                        generate_ig              
from codesign_pyutils.tasks import FlippingTaskGen

    
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--gen_urdf', '-g', type=str2bool,\
                        help = 'whether to generate urdf from xacro', default = True)

    args = parser.parse_args()

    main(args)