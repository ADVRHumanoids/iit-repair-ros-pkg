cmake_minimum_required(VERSION 3.0.2)
project(repair_gazebo)

find_package(catkin REQUIRED)

catkin_add_env_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.sh.in")