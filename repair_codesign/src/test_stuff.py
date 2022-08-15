#!/usr/bin/env python3

import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib import colors
from pylab import cm

from collections import Counter
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.post_proc_utils import PostProcL1, PostProcL2

import os

import numpy as np

postprl1 = PostProcL1("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/load_dir", 
                        )
postprl1.print_sol_run_info()
