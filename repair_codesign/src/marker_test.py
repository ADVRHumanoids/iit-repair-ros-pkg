#!/usr/bin/env python3

from codesign_pyutils.ros_utils import MarkerGen

import subprocess

rviz_window = subprocess.Popen(["roslaunch", "repair_urdf", "repair_full_markers.launch"])

init_pose_marker_topic = "init_pose"
trgt_pose_marker_topic = "trgt_pose"
rviz_int_marker = MarkerGen()
rviz_int_marker.add_marker("working_surface_link", [0, 0, 0.3], init_pose_marker_topic, "N1",  0.3) 
rviz_int_marker.add_marker("working_surface_link", [0, 0, 0.5], trgt_pose_marker_topic, "N2", 0.3) 
rviz_int_marker.spin()
rviz_int_marker.add_marker("working_surface_link", [0, 0, 0.1], "scibidibi", 0.3) # this will throw a warning
