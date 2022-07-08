#!/usr/bin/env python3

from codesign_pyutils.ros_utils import GenPosesFromRViz

marker = GenPosesFromRViz("repair_cage", "repair_urdf", marker_scale = 0.2, base_link_name = "working_surface_link")