cmake_minimum_required(VERSION 3.0.2)
project(repair_gazebo)

find_package(catkin REQUIRED)

install(DIRECTORY
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)