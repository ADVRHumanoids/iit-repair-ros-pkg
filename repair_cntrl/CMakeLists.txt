cmake_minimum_required(VERSION 3.1.3)
project(repair_cntrl)

find_package(catkin REQUIRED)

install(DIRECTORY
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_package()

