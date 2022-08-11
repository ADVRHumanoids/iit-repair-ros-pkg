#!/usr/bin/env python3

import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib import colors
from pylab import cm

from collections import Counter
from codesign_pyutils.load_utils import LoadSols

sol_loader = LoadSols("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/load_dir/first_level")

print(sol_loader.task_info_data["right_arm_picks"][0][0])