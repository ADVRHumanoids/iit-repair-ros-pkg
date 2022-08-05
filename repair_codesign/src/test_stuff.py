#!/usr/bin/env python3

import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib import colors
from pylab import cm

from collections import Counter

y_pred = np.array([1, 2, 1, 4, 3, 2, 1, 5, 8, 9, 3 ,5, 7, 7, 1,])
y_un = np.unique(y_pred)
print(np.where(y_pred == y_un[0])[0])
