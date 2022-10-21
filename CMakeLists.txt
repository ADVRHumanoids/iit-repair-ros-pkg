cmake_minimum_required(VERSION 3.0)

set(ROBOT_NAME repair)
project(iit-${ROBOT_NAME}-ros-pkg VERSION 1.0.0)

add_subdirectory(repair_codesign)
add_subdirectory(repair_urdf)
add_subdirectory(repair_srdf)
add_subdirectory(repair_cartesio)
add_subdirectory(repair_cntrl)
add_subdirectory(repair_gazebo)

